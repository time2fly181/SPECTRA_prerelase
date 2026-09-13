"""Pre-norm learned-feature/time axial Transformer using native PyTorch SDPA."""

from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .common import _SDP_BACKEND_CHOICES, _sdp_kernel_context


def sdpa_backend_diagnostic(
    *,
    d_model: int = 128,
    nhead: int = 4,
    sequence_length: int = 630,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str = "cuda",
) -> dict[str, object]:
    """Report native SDPA fused-kernel eligibility for the axial layout."""
    resolved_device = torch.device(device)
    result: dict[str, object] = {
        "device": str(resolved_device),
        "dtype": str(dtype),
        "d_model": d_model,
        "nhead": nhead,
        "head_dim": d_model // nhead,
        "sequence_length": sequence_length,
        "flash_eligible": False,
        "memory_efficient_eligible": False,
        "expected_backend": "math",
    }
    if resolved_device.type != "cuda" or not torch.cuda.is_available():
        result["reason"] = "CUDA is unavailable"
        return result
    q = torch.empty(
        1,
        nhead,
        sequence_length,
        d_model // nhead,
        device=resolved_device,
        dtype=dtype,
    )
    params = torch.nn.attention.SDPAParams(q, q, q, None, 0.0, False, False)  # type: ignore[reportPrivateImportUsage]
    flash = bool(torch.nn.attention.can_use_flash_attention(params, debug=False))  # type: ignore[reportPrivateImportUsage]
    efficient = bool(
        torch.nn.attention.can_use_efficient_attention(params, debug=False)  # type: ignore[reportPrivateImportUsage]
    )
    result["flash_eligible"] = flash
    result["memory_efficient_eligible"] = efficient
    result["expected_backend"] = (
        "flash" if flash else "memory_efficient" if efficient else "math"
    )
    return result


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    even = x[..., 0::2]
    odd = x[..., 1::2]
    return torch.stack((-odd, even), dim=-1).flatten(-2)


class RotaryTemporalEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_len: int = 4096) -> None:
        super().__init__()
        if head_dim % 2:
            raise ValueError("RoPE head_dim must be even")
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        positions = torch.arange(max_len, dtype=torch.float32)
        angles = torch.outer(positions, inv_freq).repeat_interleave(2, dim=-1)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        length = q.size(-2)
        cos_tensor = cast(torch.Tensor, self.cos)
        sin_tensor = cast(torch.Tensor, self.sin)
        cos = cos_tensor[:length].to(device=q.device, dtype=q.dtype)[None, None]
        sin = sin_tensor[:length].to(device=q.device, dtype=q.dtype)[None, None]
        return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


class SDPAMultiheadAttention(nn.Module):
    """Self-attention with explicit QKV projections and fused SDPA dispatch."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        *,
        dropout: float,
        backend: str,
        use_rope: bool = False,
        max_len: int = 4096,
    ) -> None:
        super().__init__()
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")
        if backend not in _SDP_BACKEND_CHOICES:
            raise ValueError(f"Unsupported attention backend {backend!r}")
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.dropout = float(dropout)
        self.backend = backend
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.rope = (
            RotaryTemporalEmbedding(self.head_dim, max_len=max_len)
            if use_rope
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        local_window: int | None = None,
    ) -> torch.Tensor:
        batch, length, _ = x.shape
        resolved_valid_mask: torch.Tensor | None = None
        if valid_mask is not None:
            # Fused and math SDPA kernels are not guaranteed to produce finite
            # values when a sequence has no valid keys.  Supply one zero-valued
            # dummy key for those rows; the axial wrapper zeros invalid queries.
            mask = valid_mask.to(torch.bool)
            any_valid = mask.any(dim=-1)
            first_valid = mask[:, :1] | (~any_valid).unsqueeze(1)
            resolved_valid_mask = torch.cat([first_valid, mask[:, 1:]], dim=1)
            zero_first = torch.where(
                any_valid[:, None, None], x[:, :1], torch.zeros_like(x[:, :1])
            )
            x = torch.cat([zero_first, x[:, 1:]], dim=1)
        qkv = self.qkv(x).view(batch, length, 3, self.nhead, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        if self.rope is not None:
            q, k = self.rope(q, k)
        attn_mask = None
        if resolved_valid_mask is not None:
            attn_mask = resolved_valid_mask[:, None, None, :]
        if local_window is not None:
            if local_window < 0:
                raise ValueError("local_window must be non-negative")
            positions = torch.arange(length, device=x.device)
            local_mask = (positions[:, None] - positions[None, :]).abs() <= local_window
            local_mask = local_mask[None, None]
            attn_mask = local_mask if attn_mask is None else (attn_mask & local_mask)
        with _sdp_kernel_context(self.backend):
            attended = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )
        return self.out_proj(
            attended.transpose(1, 2).contiguous().view(batch, length, self.d_model)
        )


class SDPACrossAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        *,
        dropout: float,
        backend: str,
    ) -> None:
        super().__init__()
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.dropout = float(dropout)
        self.backend = backend
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        return (
            x.view(batch, length, self.nhead, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_valid_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        q = self._heads(self.q_proj(query))
        k = self._heads(self.k_proj(key_value))
        v = self._heads(self.v_proj(key_value))
        mask = (
            key_valid_mask.to(torch.bool)[:, None, None, :]
            if key_valid_mask is not None
            else None
        )
        with _sdp_kernel_context(self.backend):
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )
        batch, _, query_len, _ = out.shape
        return self.out_proj(
            out.transpose(1, 2).contiguous().view(batch, query_len, self.d_model)
        )


class FeatureAxialAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, *, dropout: float, backend: str):
        super().__init__()
        self.attention = SDPAMultiheadAttention(
            d_model, nhead, dropout=dropout, backend=backend
        )
        self.last_sequence_shape: tuple[int, ...] | None = None

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        batch, time, slots, dim = x.shape
        sequence = x.reshape(batch * time, slots, dim).contiguous()
        mask = valid.reshape(batch * time, slots).contiguous()
        self.last_sequence_shape = tuple(sequence.shape)
        out = self.attention(sequence, mask).view(batch, time, slots, dim)
        return out * valid.unsqueeze(-1).to(out.dtype)


class TemporalAxialAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        *,
        dropout: float,
        backend: str,
        max_time: int,
        local_window: int | None = None,
    ):
        super().__init__()
        self.attention = SDPAMultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            backend=backend,
            use_rope=True,
            max_len=max_time,
        )
        self.local_window = local_window
        self.last_sequence_shape: tuple[int, ...] | None = None

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        batch, time, slots, dim = x.shape
        sequence = x.permute(0, 2, 1, 3).reshape(batch * slots, time, dim).contiguous()
        mask = valid.permute(0, 2, 1).reshape(batch * slots, time).contiguous()
        self.last_sequence_shape = tuple(sequence.shape)
        out = self.attention(sequence, mask, local_window=self.local_window)
        out = out.view(batch, slots, time, dim).permute(0, 2, 1, 3).contiguous()
        return out * valid.unsqueeze(-1).to(out.dtype)


class PreNormAxialBlock(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        dim_feedforward: int = 512,
        *,
        dropout: float = 0.2,
        backend: str = "auto",
        max_time: int = 630,
        enable_feature_attention: bool = True,
        enable_temporal_attention: bool = True,
        temporal_local_window: int | None = None,
    ) -> None:
        super().__init__()
        self.feature_norm = nn.LayerNorm(d_model)
        self.feature_attention = FeatureAxialAttention(
            d_model, nhead, dropout=dropout, backend=backend
        )
        self.temporal_norm = nn.LayerNorm(d_model)
        self.temporal_attention = TemporalAxialAttention(
            d_model,
            nhead,
            dropout=dropout,
            backend=backend,
            max_time=max_time,
            local_window=temporal_local_window,
        )
        self.enable_feature_attention = bool(enable_feature_attention)
        self.enable_temporal_attention = bool(enable_temporal_attention)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        valid_f = valid.unsqueeze(-1).to(x.dtype)
        if self.enable_feature_attention:
            x = (x + self.feature_attention(self.feature_norm(x), valid)) * valid_f
        if self.enable_temporal_attention:
            x = (x + self.temporal_attention(self.temporal_norm(x), valid)) * valid_f
        x = (x + self.ffn(self.ffn_norm(x))) * valid_f
        return x


class CenterFeatureTimeReadout(nn.Module):
    """Learned query readout over one or all contextualized epoch grids."""

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        dim_feedforward: int = 512,
        *,
        dropout: float = 0.2,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.query_norm = nn.LayerNorm(d_model)
        self.kv_norm = nn.LayerNorm(d_model)
        self.cross_attention = SDPACrossAttention(
            d_model, nhead, dropout=dropout, backend=backend
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.last_key_value_shape: tuple[int, ...] | None = None

    def forward(
        self,
        grid: torch.Tensor,
        token_valid: torch.Tensor,
        epoch_valid: torch.Tensor,
        *,
        all_epochs: bool,
    ) -> torch.Tensor:
        # grid/token_valid: [B, L, S, K, D] / [B, L, S, K]
        batch, epochs, bins, slots, dim = grid.shape
        if all_epochs:
            kv = grid.reshape(batch * epochs, bins * slots, dim)
            mask = token_valid.reshape(batch * epochs, bins * slots)
            valid_epoch = epoch_valid.reshape(batch * epochs)
        else:
            center = epochs // 2
            kv = grid[:, center].reshape(batch, bins * slots, dim)
            mask = token_valid[:, center].reshape(batch, bins * slots)
            valid_epoch = epoch_valid[:, center]
        self.last_key_value_shape = tuple(kv.shape)

        # SDPA produces NaNs for a row with no valid key. Insert a zero dummy key
        # for those rows, then zero the readout again after attention.
        any_valid = mask.any(dim=-1)
        first_valid = mask[:, :1] | (~any_valid).unsqueeze(1)
        mask = torch.cat([first_valid, mask[:, 1:]], dim=1)
        zero_first = torch.where(
            any_valid[:, None, None], kv[:, :1], torch.zeros_like(kv[:, :1])
        )
        kv = torch.cat([zero_first, kv[:, 1:]], dim=1)

        query = self.query.expand(kv.size(0), -1, -1)
        query = query + self.cross_attention(
            self.query_norm(query), self.kv_norm(kv), mask
        )
        query = query + self.ffn(self.ffn_norm(query))
        result = self.output_norm(query[:, 0])
        result = result * valid_epoch.unsqueeze(-1).to(result.dtype)
        if all_epochs:
            return result.view(batch, epochs, dim)
        return result


class LearnedFeatureAxialBackbone(nn.Module):
    """Project local slots and apply continuous feature/time axial attention."""

    def __init__(
        self,
        *,
        num_slots: int = 16,
        slot_local_dim: int = 32,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.2,
        context_epochs: int = 21,
        time_bins_per_epoch: int = 30,
        slot_modality_ids: torch.Tensor,
        backend: str = "auto",
        gradient_checkpointing: bool = True,
        enable_feature_attention: bool = True,
        enable_temporal_attention: bool = True,
        temporal_attention_mode: str = "full",
        temporal_local_window: int = 90,
    ) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.slot_local_dim = slot_local_dim
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.context_epochs = context_epochs
        self.time_bins_per_epoch = time_bins_per_epoch
        if temporal_attention_mode not in {"full", "local_local_global"}:
            raise ValueError(
                "temporal_attention_mode must be 'full' or 'local_local_global'"
            )
        self.temporal_attention_mode = temporal_attention_mode
        self.temporal_local_window = int(temporal_local_window)
        self.slot_projection = nn.Linear(slot_local_dim, d_model)
        self.feature_slot_embedding = nn.Parameter(
            torch.randn(num_slots, d_model) * 0.02
        )
        self.modality_embedding = nn.Parameter(torch.randn(3, d_model) * 0.02)
        self.mask_content = nn.Parameter(torch.randn(1, 1, 1, d_model) * 0.02)
        self.register_buffer(
            "slot_modality_ids", slot_modality_ids.to(torch.long).clone()
        )
        max_time = context_epochs * time_bins_per_epoch
        self.blocks = nn.ModuleList(
            [
                PreNormAxialBlock(
                    d_model,
                    nhead,
                    dim_feedforward,
                    dropout=dropout,
                    backend=backend,
                    max_time=max_time,
                    enable_feature_attention=enable_feature_attention,
                    enable_temporal_attention=enable_temporal_attention,
                    temporal_local_window=(
                        self.temporal_local_window
                        if temporal_attention_mode == "local_local_global"
                        and layer_idx < max(1, num_layers // 2)
                        else None
                    ),
                )
                for layer_idx in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.gradient_checkpointing = bool(gradient_checkpointing)

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing = True
        return self

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False
        return self

    def forward(
        self,
        local_features: torch.Tensor,
        epoch_valid: torch.Tensor,
        slot_valid: torch.Tensor,
        center_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # local_features: [B, L, S, K, local_dim]
        batch, epochs, bins, slots, _ = local_features.shape
        if slots != self.num_slots:
            raise ValueError(f"Expected {self.num_slots} slots, got {slots}")
        x = self.slot_projection(local_features)
        x = x + self.feature_slot_embedding[None, None, None]
        modality_ids = cast(torch.Tensor, self.slot_modality_ids)
        x = x + self.modality_embedding[modality_ids][None, None, None]
        if center_mask is not None:
            center = epochs // 2
            center_content = x[:, center]
            # Remove all measured center-epoch content while retaining only
            # learned slot/modality identity.  The classification query cannot
            # access the masked center because readout validity is handled below.
            identity = (
                self.feature_slot_embedding + self.modality_embedding[modality_ids]
            )
            replacement = self.mask_content + identity[None, None]
            replacement = replacement.expand(batch, bins, slots, -1)
            x = x.clone()
            x[:, center] = torch.where(
                center_mask[:, None, None, None], replacement, center_content
            )

        time_valid = (
            epoch_valid[:, :, None].expand(-1, -1, bins).reshape(batch, epochs * bins)
        )
        if slot_valid.dim() == 2:
            expanded_slot_valid = slot_valid[:, None, :].expand(-1, epochs * bins, -1)
        elif slot_valid.dim() == 3:
            expanded_slot_valid = (
                slot_valid[:, :, None, :]
                .expand(-1, -1, bins, -1)
                .reshape(batch, epochs * bins, slots)
            )
        else:
            raise ValueError("slot_valid must have shape [B,K] or [B,L,K]")
        valid = time_valid[:, :, None] & expanded_slot_valid
        x = x.reshape(batch, epochs * bins, slots, self.d_model)
        x = x * valid.unsqueeze(-1).to(x.dtype)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, valid, use_reentrant=False)
            else:
                x = block(x, valid)
        x = cast(torch.Tensor, x)
        x = self.output_norm(x) * valid.unsqueeze(-1).to(x.dtype)
        return x, valid


class LearnedFeatureAxialV2Backbone(nn.Module):
    """Contextualize channel-preserving 1 Hz tokens with axial attention."""

    def __init__(
        self,
        *,
        num_slots: int,
        slot_local_dim: int,
        d_model: int = 160,
        nhead: int = 5,
        num_layers: int = 4,
        dim_feedforward: int = 640,
        dropout: float = 0.2,
        context_epochs: int = 35,
        time_bins_per_epoch: int = 30,
        slot_modality_ids: torch.Tensor,
        slot_channel_ids: torch.Tensor,
        num_channels: int = 5,
        backend: str = "auto",
        gradient_checkpointing: bool = True,
        temporal_local_window: int = 90,
    ) -> None:
        super().__init__()
        if d_model % nhead:
            raise ValueError("v2 d_model must be divisible by nhead")
        if num_layers < 1:
            raise ValueError("v2 num_layers must be at least one")
        self.num_slots = int(num_slots)
        self.d_model = int(d_model)
        self.context_epochs = int(context_epochs)
        self.time_bins_per_epoch = int(time_bins_per_epoch)
        self.slot_projection = nn.Linear(slot_local_dim, d_model)
        self.feature_slot_embedding = nn.Parameter(
            torch.randn(num_slots, d_model) * 0.02
        )
        self.channel_embedding = nn.Parameter(torch.randn(num_channels, d_model) * 0.02)
        self.modality_embedding = nn.Parameter(torch.randn(3, d_model) * 0.02)
        self.within_epoch_embedding = nn.Parameter(
            torch.randn(time_bins_per_epoch, d_model) * 0.02
        )
        self.register_buffer(
            "slot_modality_ids", slot_modality_ids.to(torch.long).clone()
        )
        self.register_buffer(
            "slot_channel_ids", slot_channel_ids.to(torch.long).clone()
        )
        max_time = context_epochs * time_bins_per_epoch
        self.blocks = nn.ModuleList(
            [
                PreNormAxialBlock(
                    d_model,
                    nhead,
                    dim_feedforward,
                    dropout=dropout,
                    backend=backend,
                    max_time=max_time,
                    temporal_local_window=(
                        temporal_local_window
                        if layer_index < max(1, num_layers // 2)
                        else None
                    ),
                )
                for layer_index in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.gradient_checkpointing = bool(gradient_checkpointing)

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing = True
        return self

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False
        return self

    def forward(
        self,
        coarse_features: torch.Tensor,
        epoch_valid: torch.Tensor,
        slot_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``[B,L*K,S,D]`` contextual tokens and their validity mask."""
        batch, epochs, bins, slots, _ = coarse_features.shape
        if slots != self.num_slots or bins != self.time_bins_per_epoch:
            raise ValueError("Unexpected v2 coarse feature geometry")
        if epochs != self.context_epochs:
            raise ValueError(
                f"Expected context_epochs={self.context_epochs}, got {epochs}"
            )
        modality_ids = cast(torch.Tensor, self.slot_modality_ids)
        channel_ids = cast(torch.Tensor, self.slot_channel_ids)
        identity = (
            self.feature_slot_embedding
            + self.modality_embedding[modality_ids]
            + self.channel_embedding[channel_ids]
        )
        x = self.slot_projection(coarse_features)
        x = x + identity[None, None, None]
        x = x + self.within_epoch_embedding[None, None, :, None]
        valid = (epoch_valid[:, :, None, None] & slot_valid[:, :, None, :]).expand(
            -1, -1, bins, -1
        )
        x = x.reshape(batch, epochs * bins, slots, self.d_model)
        valid = valid.reshape(batch, epochs * bins, slots)
        x = x * valid.unsqueeze(-1).to(x.dtype)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, valid, use_reentrant=False)
            else:
                x = block(x, valid)
        x = cast(torch.Tensor, x)
        x = self.output_norm(x) * valid.unsqueeze(-1).to(x.dtype)
        return x, valid


class _EpochSummaryBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        *,
        dropout: float,
        backend: str,
        max_epochs: int,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(d_model)
        self.attention = SDPAMultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            backend=backend,
            use_rope=True,
            max_len=max_epochs,
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        valid_f = valid.unsqueeze(-1).to(x.dtype)
        x = (x + self.attention(self.attention_norm(x), valid)) * valid_f
        return (x + self.ffn(self.ffn_norm(x))) * valid_f


class EpochSummaryTransformer(nn.Module):
    """Reason over a cheap sequence of masked per-epoch summaries."""

    def __init__(
        self,
        d_model: int = 160,
        nhead: int = 5,
        num_layers: int = 2,
        dim_feedforward: int = 640,
        *,
        dropout: float = 0.2,
        context_epochs: int = 35,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                _EpochSummaryBlock(
                    d_model,
                    nhead,
                    dim_feedforward,
                    dropout=dropout,
                    backend=backend,
                    max_epochs=context_epochs,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        coarse_grid: torch.Tensor,
        token_valid: torch.Tensor,
        epoch_valid: torch.Tensor,
    ) -> torch.Tensor:
        valid_f = token_valid.unsqueeze(-1).to(coarse_grid.dtype)
        denominator = valid_f.sum(dim=(2, 3)).clamp_min(1.0)
        summary = (coarse_grid * valid_f).sum(dim=(2, 3)) / denominator
        summary = summary * epoch_valid.unsqueeze(-1).to(summary.dtype)
        for block in self.blocks:
            summary = block(summary, epoch_valid)
        return self.output_norm(summary) * epoch_valid.unsqueeze(-1).to(summary.dtype)


class MultiEvidenceReadout(nn.Module):
    """Pool fine and coarse evidence with global and modality-specific queries."""

    def __init__(
        self,
        *,
        fine_dim: int,
        d_model: int = 160,
        output_dim: int = 512,
        fine_bins: int = 240,
        num_slots: int = 16,
        slot_modality_ids: torch.Tensor,
        slot_channel_ids: torch.Tensor,
        num_channels: int = 5,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.fine_dim = int(fine_dim)
        self.d_model = int(d_model)
        self.fine_bins = int(fine_bins)
        self.fine_queries = nn.Parameter(torch.randn(4, fine_dim) * 0.02)
        self.coarse_queries = nn.Parameter(torch.randn(4, d_model) * 0.02)
        self.fine_position_embedding = nn.Parameter(
            torch.randn(fine_bins, fine_dim) * 0.02
        )
        self.fine_slot_embedding = nn.Parameter(torch.randn(num_slots, fine_dim) * 0.02)
        self.fine_modality_embedding = nn.Parameter(torch.randn(3, fine_dim) * 0.02)
        self.fine_channel_embedding = nn.Parameter(
            torch.randn(num_channels, fine_dim) * 0.02
        )
        self.register_buffer(
            "slot_modality_ids", slot_modality_ids.to(torch.long).clone()
        )
        self.register_buffer(
            "slot_channel_ids", slot_channel_ids.to(torch.long).clone()
        )
        self.fine_projection = nn.Linear(fine_dim, d_model)
        self.query_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Sequential(
            nn.LayerNorm(5 * d_model),
            nn.Linear(5 * d_model, output_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.LayerNorm(output_dim),
        )

    @staticmethod
    def _pool(
        tokens: torch.Tensor,
        query: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        scores = torch.einsum("bnt,qt->bqn", tokens, query)
        scores = scores / math.sqrt(tokens.size(-1))
        masked_scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        weights = torch.softmax(masked_scores.float(), dim=-1).to(scores.dtype)
        has_value = valid.any(dim=-1, keepdim=True)
        weights = weights * has_value.to(weights.dtype)
        return torch.einsum("bqn,bnt->bqt", weights, tokens)

    def forward(
        self,
        fine_grid: torch.Tensor,
        coarse_grid: torch.Tensor,
        coarse_valid: torch.Tensor,
        slot_valid: torch.Tensor,
        epoch_summary: torch.Tensor,
        epoch_valid: torch.Tensor,
        *,
        all_epochs: bool,
    ) -> torch.Tensor:
        batch, epochs, fine_bins, slots, _ = fine_grid.shape
        if fine_bins != self.fine_bins:
            raise ValueError("Unexpected v2 fine feature geometry")
        modality_ids = cast(torch.Tensor, self.slot_modality_ids)
        channel_ids = cast(torch.Tensor, self.slot_channel_ids)
        fine_identity = (
            self.fine_slot_embedding
            + self.fine_modality_embedding[modality_ids]
            + self.fine_channel_embedding[channel_ids]
        )
        fine = fine_grid + fine_identity[None, None, None]
        fine = fine + self.fine_position_embedding[None, None, :, None]
        fine = fine.reshape(batch * epochs, fine_bins * slots, self.fine_dim)
        coarse = coarse_grid.reshape(
            batch * epochs, coarse_grid.size(2) * slots, self.d_model
        )

        fine_slot_valid = slot_valid[:, :, None, :].expand(-1, -1, fine_bins, -1)
        query_slot_valid = torch.stack(
            [
                torch.ones_like(modality_ids, dtype=torch.bool),
                modality_ids == 0,
                modality_ids == 1,
                modality_ids == 2,
            ]
        )
        fine_valid = (
            fine_slot_valid[:, :, None]
            & query_slot_valid[None, None, :, None, :]
            & epoch_valid[:, :, None, None, None]
        ).reshape(batch * epochs, 4, fine_bins * slots)
        coarse_query_valid = (
            coarse_valid[:, :, None] & query_slot_valid[None, None, :, None, :]
        ).reshape(batch * epochs, 4, coarse_grid.size(2) * slots)

        pooled_fine = self._pool(fine, self.fine_queries, fine_valid)
        pooled_coarse = self._pool(coarse, self.coarse_queries, coarse_query_valid)
        evidence = self.query_norm(pooled_coarse + self.fine_projection(pooled_fine))
        combined = torch.cat(
            [evidence.reshape(batch, epochs, 4 * self.d_model), epoch_summary],
            dim=-1,
        )
        result = self.output_projection(combined)
        result = result * epoch_valid.unsqueeze(-1).to(result.dtype)
        if all_epochs:
            return result
        return result[:, epochs // 2]
