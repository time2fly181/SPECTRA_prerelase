"""Transformer-based context model for PSG sleep staging."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from .channel_embedding import LegacyChannelEmbeddingState
from .common import (
    _SDP_BACKEND_CHOICES,
    _sdp_kernel_context,
    transformer_init_,
)
from .context_input import (
    ContextWaveformPreparer,
    PreparedContextWaveforms,
    WaveformInputs,
)
from .multirate_asymmetric_epoch_cnn import MultiRateAsymmetricEpochCNN


@dataclass
class ForwardOutput:
    """Structured output from ``TransformerContextNet.forward``.

    Attributes:
        logits: Class prediction logits, shape [B, num_classes] or [B, L, num_classes].
        features: Extracted features from the center epoch or all epochs.
            Shape [B, d_model] or [B, L, d_model]. None if not requested.
        attention_weights: Attention weights from the last transformer layer.
            Shape [B, H, L, L]. None if not requested.
        readout_attention_weights: Optional center-readout attention, shape [B, 1, 1, L].
        confidence: Predicted probability that the stage prediction is correct.
            Shape [B] or [B, L]. None if the confidence head is disabled.
    """

    logits: torch.Tensor
    features: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None
    attention_weights: torch.Tensor | None = None
    readout_attention_weights: torch.Tensor | None = None
    confidence: torch.Tensor | None = None


class ResidualMLPClassifier(nn.Module):
    """Default classifier head for contextualized epoch representations."""

    def __init__(
        self,
        d_model: int,
        num_classes: int,
        *,
        hidden_dim: int | None = None,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim or max(1, d_model // 2))
        self.input_norm = nn.LayerNorm(d_model)
        self.input_proj = nn.Sequential(
            nn.Linear(d_model, self.hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
        )
        self.res_norm = nn.LayerNorm(self.hidden_dim)
        self.res_linear1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.res_drop = nn.Dropout(dropout)
        self.res_linear2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.head = nn.Linear(self.hidden_dim, self.num_classes)
        nn.init.normal_(self.res_linear2.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.res_linear2.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        has_sequence = x.dim() == 3
        if has_sequence:
            x = x.reshape(-1, original_shape[-1])
        h = self.input_proj(self.input_norm(x))
        residual = self.res_linear2(
            self.res_drop(
                F.gelu(self.res_linear1(self.res_norm(h)), approximate="tanh")
            )
        )
        h_out = self.output_norm(h + residual)
        if has_sequence:
            h_out = h_out.view(*original_shape[:-1], self.hidden_dim)
        return h_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))

    def initialize_output_bias_uniform(self) -> None:
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def set_output_bias_from_log_priors(self, log_priors: torch.Tensor) -> None:
        if self.head.bias is not None:
            with torch.no_grad():
                self.head.bias.copy_(log_priors.to(self.head.bias.device))


class PerPositionSleepHead(nn.Module):
    """Per-position sleep-stage head for predict-all (all-positions) training.

    Maps ``[B, L, d] -> [B, L, C]`` with weights shared across positions, so the
    same epoch evaluated at different window offsets produces comparable logits
    (the precondition for valid offset-averaging at inference). It is
    deliberately position-agnostic: there is no position embedding inside the
    head. A 2-D input ``[B, d]`` (the legacy center-only path) is also accepted
    and returns ``[B, C]``.

    The inter-epoch transformer output is already context-rich at every position
    (that was :class:`CenterContextReadout`'s job for the single center token),
    so the head does not aggregate context and stays lean. Dropout is kept so
    MC-dropout at inference perturbs the per-position predictions and stacks with
    offset-averaging.

    Args:
        d_model: Feature dimension of the inter-epoch representation.
        num_classes: Number of sleep-stage classes.
        hidden_dim: Hidden width of the shared MLP. Defaults to ``d_model // 2``.
        dropout: Dropout probability inside the MLP.
        use_local_mix: If True, add an optional residual depthwise mix across the
            epoch axis (lets a position sharpen boundary/transition predictions
            using immediate neighbors). Gated by a zero-init parameter so the
            head is numerically identical to ``use_local_mix=False`` at init.
            Default OFF: it can also smooth across true boundaries, so it must be
            ablated against off, not assumed beneficial.
        local_kernel: Kernel size for the depthwise mixing conv (odd).
        norm: Must be ``"layernorm"``.
    """

    def __init__(
        self,
        d_model: int,
        num_classes: int,
        *,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
        use_local_mix: bool = False,
        local_kernel: int = 3,
        norm: str = "layernorm",
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim or max(1, d_model // 2))
        if norm != "layernorm":
            raise ValueError("Only norm='layernorm' is supported")
        self.norm = nn.LayerNorm(self.d_model)
        self.use_local_mix = bool(use_local_mix)
        if self.use_local_mix:
            if local_kernel % 2 == 0:
                raise ValueError(f"local_kernel must be odd, got {local_kernel}")
            self.local = nn.Conv1d(
                self.d_model,
                self.d_model,
                int(local_kernel),
                padding=int(local_kernel) // 2,
                groups=self.d_model,
                bias=False,
            )
            self.local_gate = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(self.d_model, self.hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.num_classes),
        )

    @property
    def head(self) -> nn.Linear:
        """Final classification ``Linear`` (for bias-init parity with peers)."""
        final = self.mlp[-1]
        assert isinstance(final, nn.Linear)
        return final

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map context features to per-position logits.

        Args:
            x: ``[B, L, d]`` (all positions) or ``[B, d]`` (center only).

        Returns:
            ``[B, L, C]`` or ``[B, C]`` matching the input rank.
        """
        h = self.norm(x)
        if self.use_local_mix and h.dim() == 3:
            mixed = self.local(h.transpose(1, 2)).transpose(1, 2)
            h = h + self.local_gate * mixed
        return self.mlp(h)

    def initialize_output_bias_uniform(self) -> None:
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def set_output_bias_from_log_priors(self, log_priors: torch.Tensor) -> None:
        if self.head.bias is not None:
            with torch.no_grad():
                self.head.bias.copy_(log_priors.to(self.head.bias.device))


class ConfidenceHead(nn.Module):
    """Predict probability that the chosen class prediction is correct."""

    def __init__(
        self, d_model: int, *, hidden_dim: int | None = None, dropout: float = 0.2
    ) -> None:
        super().__init__()
        hidden = int(hidden_dim or max(1, d_model // 2))
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return out.squeeze(-1)


class CenterContextReadout(nn.Module):
    """Fuse the center epoch with an explicit summary of neighboring context.

    Sleep-stage labels are assigned to the center epoch, but literature-aligned
    context models typically improve that prediction using nearby epochs.  This
    readout keeps the center token as the anchor while building a separate
    attention-weighted summary over the *neighboring* epochs only.
    """

    def __init__(
        self, d_model: int, *, dropout: float = 0.2, gate_bias_init: float = -2.0
    ):
        super().__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.fuse = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
        )
        self.gate = nn.Linear(d_model * 2, d_model)
        if self.gate.bias is not None:
            nn.init.constant_(self.gate.bias, gate_bias_init)
        self._last_attention_weights: torch.Tensor | None = None

    def forward(
        self,
        z: torch.Tensor,
        epoch_valid_mask: torch.Tensor | None = None,
        *,
        return_attention: bool = False,
    ) -> torch.Tensor:
        """Return a center representation enriched by valid neighbor context."""
        if z.dim() != 3:
            raise ValueError(f"Expected [B, L, D] input, got shape {tuple(z.shape)}")
        B, L, D = z.shape
        center_idx = L // 2
        center = z[:, center_idx, :]
        if epoch_valid_mask is None:
            valid = torch.ones(B, L, dtype=torch.bool, device=z.device)
        else:
            valid = epoch_valid_mask.to(device=z.device, dtype=torch.bool)
            if valid.shape != (B, L):
                raise ValueError(
                    f"epoch_valid_mask must have shape {(B, L)}, got {tuple(valid.shape)}"
                )
        if L <= 1:
            self._last_attention_weights = (
                z.new_zeros(B, 1, 1, L) if return_attention else None
            )
            return center
        q = self.query(center).unsqueeze(1)
        k = self.key(z)
        v = self.value(z)
        scores = torch.matmul(q, k.transpose(1, 2)).squeeze(1) / math.sqrt(D)
        valid_neighbors = valid.clone()
        valid_neighbors[:, center_idx] = False
        has_neighbor = valid_neighbors.any(dim=1, keepdim=True)
        scores = scores.masked_fill(~valid_neighbors, float("-inf"))
        safe_scores = torch.where(has_neighbor, scores, torch.zeros_like(scores))
        weights = F.softmax(safe_scores, dim=-1) * valid_neighbors.to(scores.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-08)
        self._last_attention_weights = (
            weights[:, None, None, :] if return_attention else None
        )
        context = torch.bmm(weights.unsqueeze(1), v).squeeze(1)
        fused_input = torch.cat([center, context], dim=-1)
        gate = torch.sigmoid(self.gate(fused_input))
        delta = self.fuse(fused_input)
        enriched = center + gate * delta
        return torch.where(has_neighbor, enriched, center)


def _safe_key_padding_mask(
    key_padding_mask: torch.Tensor | None,
) -> torch.Tensor | None:
    """Avoid fully masked attention rows while preserving ordinary padding."""
    if key_padding_mask is None:
        return None
    mask = key_padding_mask.to(dtype=torch.bool)
    fully_masked = mask.all(dim=-1)
    safe_first = mask[:, :1] & ~fully_masked.unsqueeze(1)
    return torch.cat((safe_first, mask[:, 1:]), dim=1)


class SwiGLUTransformerEncoderLayer(nn.Module):
    """Pre-norm encoder layer with a gated, bias-free SwiGLU FFN.

    ``dim_feedforward`` is the gate/up hidden width directly.

    Args:
        d_model: Token feature dimension.
        nhead: Number of self-attention heads.
        dim_feedforward: Hidden width of the gate and up projections.
        dropout: Attention, FFN, and residual dropout probability.
    """

    ffn_activation = "swiglu"

    def __init__(
        self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.w_gate = nn.Linear(d_model, dim_feedforward, bias=False)
        self.w_up = nn.Linear(d_model, dim_feedforward, bias=False)
        self.w_down = nn.Linear(dim_feedforward, d_model, bias=False)
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """Apply pre-norm self-attention followed by the SwiGLU FFN.

        Args:
            src: Token features shaped ``[batch, length, d_model]``.
            src_mask: Optional attention mask.
            src_key_padding_mask: Optional padding mask.
            is_causal: Whether ``src_mask`` represents causal attention.

        Returns:
            Contextual token features with the same shape as ``src``.
        """
        normed = self.norm1(src)
        attn_out, _ = self.self_attn(
            normed,
            normed,
            normed,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=False,
            is_causal=is_causal,
        )
        src = src + self.dropout1(attn_out)
        return src + _encoder_layer_feed_forward(self, self.norm2(src))


def _encoder_layer_feed_forward(layer: nn.Module, src: torch.Tensor) -> torch.Tensor:
    """Run either the checkpoint-compatible GELU or SwiGLU FFN sublayer."""
    typed_layer = cast(Any, layer)
    if getattr(layer, "ffn_activation", "gelu") == "swiglu":
        gated = F.silu(typed_layer.w_gate(src)) * typed_layer.w_up(src)
        return typed_layer.dropout2(typed_layer.w_down(typed_layer.ffn_dropout(gated)))
    return typed_layer.dropout2(
        typed_layer.linear2(
            typed_layer.dropout(typed_layer.activation(typed_layer.linear1(src)))
        )
    )


def _encoder_layer_forward_with_attention(
    layer: nn.Module,
    src: torch.Tensor,
    *,
    attn_mask: torch.Tensor | None,
    key_padding_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one encoder layer while retaining per-head self-attention weights."""
    typed_layer = cast(Any, layer)

    def self_attention(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attn_out, weights = typed_layer.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
            is_causal=False,
        )
        if weights is None:
            raise RuntimeError("MultiheadAttention did not return requested weights")
        return (typed_layer.dropout1(attn_out), weights)

    if getattr(layer, "norm_first", True):
        attn_out, weights = self_attention(typed_layer.norm1(src))
        src = src + attn_out
        src = src + _encoder_layer_feed_forward(layer, typed_layer.norm2(src))
    else:
        attn_out, weights = self_attention(src)
        src = typed_layer.norm1(src + attn_out)
        src = typed_layer.norm2(src + _encoder_layer_feed_forward(layer, src))
    return (src, weights)


class TransformerContextNet(nn.Module):
    """Multirate asymmetric CNN and context transformer for normalized waveforms.

    Each 30-second epoch becomes a CNN embedding; the transformer combines 21
    embeddings before stage classification. Inputs are floating ``[B, 21, 5, 3840]``
    tensors at 128 Hz, already normalized by the EDF preprocessing pipeline.
    The model does not perform EDF preprocessing. Checkpoint metadata determines
    encoder, attention, pooling, and classifier settings; use the runtime loader
    to reconstruct a saved model. See ``ContextWaveformPreparer`` for mask rules.
    """

    def _encoder_recording_kwargs(
        self, recording_index: torch.Tensor | None, batch: int, context_len: int
    ) -> dict[str, torch.Tensor]:
        if recording_index is None:
            return {}
        flat = recording_index.reshape(-1)
        if flat.numel() != batch:
            return {}
        return {"recording_index": flat.repeat_interleave(int(context_len))}

    def __init__(
        self,
        in_ch: int,
        time_len: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 2,
        dim_ff: int = 256,
        ffn_activation: str = "gelu",
        num_classes: int = 5,
        cnn_dropout: float = 0.1,
        head_dropout: float = 0.1,
        classifier_dropout: float | None = 0.2,
        *,
        norm: str = "bn",
        sdp_backend: str = "auto",
        fs: int = 128,
        classifier_temperature: float = 1.0,
        use_confidence_head: bool = False,
        confidence_head_weight: float = 0.1,
        confidence_head_detach_target: bool = True,
        confidence_head_hidden_dim: int | None = None,
        channel_names: list[str] | str | None = None,
        epoch_encoder_variant: str = "multirate_asymmetric",
        multirate_asymmetric_encoder_kwargs: dict[str, Any] | None = None,
        classifier_head: str = "residual_mlp",
        use_per_position_head: bool = False,
        head_use_local_mix: bool = False,
        head_local_kernel: int = 3,
        head_norm: str = "layernorm",
        fusion_gate_bias: float = -2.0,
        cnn_widths: tuple[int, ...] | None = None,
        context_epochs: int = 21,
        context_attention_mode: str = "legacy_absolute",
        context_readout_mode: str = "legacy_single",
    ) -> None:
        super().__init__()
        if epoch_encoder_variant != "multirate_asymmetric":
            raise ValueError("SPECTRA supports only multirate_asymmetric checkpoints")
        self.in_ch = int(in_ch)
        self.expected_input_channels = self.in_ch
        self.num_classes = num_classes
        self.epoch_encoder_variant = epoch_encoder_variant
        self.classifier_temperature = classifier_temperature
        self.use_confidence_head = bool(use_confidence_head)
        self.confidence_head_weight = float(confidence_head_weight)
        self.confidence_head_detach_target = bool(confidence_head_detach_target)
        self.time_len = time_len
        self.fs = fs
        self.nhead = nhead
        self.context_epochs = int(context_epochs)
        clf_dropout = head_dropout if classifier_dropout is None else classifier_dropout
        if context_attention_mode != "legacy_absolute":
            raise ValueError(
                "Only context_attention_mode='legacy_absolute' is supported"
            )
        if context_readout_mode != "legacy_single":
            raise ValueError("Only context_readout_mode='legacy_single' is supported")
        if classifier_head != "residual_mlp":
            raise ValueError("Only classifier_head='residual_mlp' is supported")
        if head_norm != "layernorm":
            raise ValueError("Only head_norm='layernorm' is supported")
        if context_epochs != 21:
            raise ValueError("SPECTRA requires context_half=10 (21 context epochs)")
        if fs != 128:
            raise ValueError("SPECTRA requires a fixed sample rate of 128 Hz")
        self.context_attention_mode = context_attention_mode
        self.context_readout_mode = context_readout_mode
        self.head_dropout = float(head_dropout)
        self.classifier_dropout = float(clf_dropout)
        if ffn_activation not in {"gelu", "swiglu"}:
            raise ValueError("ffn_activation must be 'gelu' or 'swiglu'")
        self.ffn_activation = ffn_activation
        if self.confidence_head_weight < 0.0:
            raise ValueError("confidence_head_weight must be >= 0")
        if confidence_head_hidden_dim is not None and confidence_head_hidden_dim <= 0:
            raise ValueError("confidence_head_hidden_dim must be > 0 when provided")
        head_width = d_model
        self.confidence_head_hidden_dim = (
            int(confidence_head_hidden_dim)
            if confidence_head_hidden_dim is not None
            else max(1, head_width // 2)
        )
        self.fusion_gate_bias = float(fusion_gate_bias)
        encoder_kwargs: dict = {
            "in_ch": in_ch,
            "time_len": time_len,
            "dropout": cnn_dropout,
            "norm": norm,
        }
        if cnn_widths is not None:
            encoder_kwargs["widths"] = cnn_widths
        passthrough = dict(multirate_asymmetric_encoder_kwargs or {})
        reserved = {
            "in_ch",
            "time_len",
            "dropout",
            "norm",
            "widths",
            "fs",
        }.intersection(passthrough)
        if reserved:
            raise ValueError(f"Encoder kwargs must not override {sorted(reserved)}")
        self.epoch_encoder = MultiRateAsymmetricEpochCNN(
            **encoder_kwargs, fs=fs, **passthrough
        )
        enc_dim = int(self.epoch_encoder.out_dim)
        self._last_attention_weights: torch.Tensor | None = None
        if sdp_backend not in _SDP_BACKEND_CHOICES:
            raise ValueError(f"Unsupported sdp_backend '{sdp_backend}'")
        self.sdp_backend = sdp_backend
        self.epoch_encoder_out_dim = enc_dim
        self.context_d_model = int(d_model)
        self.proj_is_identity = False
        self.proj = nn.Linear(enc_dim, d_model)
        transformer_init_(self.proj)
        if enc_dim == d_model:
            nn.init.eye_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)
        max_pe_len = 128
        self.pos_encoding = nn.Parameter(torch.randn(1, max_pe_len, d_model) * 0.02)
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1 for the transformer encoder")
        encoder_layer: nn.Module
        if ffn_activation == "swiglu":
            encoder_layer = SwiGLUTransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_ff,
                dropout=head_dropout,
            )
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_ff,
                dropout=head_dropout,
                batch_first=True,
                norm_first=True,
                activation=nn.GELU(approximate="tanh"),
            )
        self.transformer = nn.TransformerEncoder(
            cast(nn.TransformerEncoderLayer, encoder_layer),
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.transformer.apply(transformer_init_)
        self.output_norm = nn.LayerNorm(d_model)
        self.center_context_readout = CenterContextReadout(
            d_model=d_model,
            dropout=clf_dropout,
            gate_bias_init=self.fusion_gate_bias,
        )
        self.center_context_readout.apply(transformer_init_)
        self.feature_dim = d_model
        self.use_per_position_head = bool(use_per_position_head)
        self.classifier_head = str(classifier_head)
        if use_per_position_head:
            self.classifier = PerPositionSleepHead(
                d_model=d_model,
                num_classes=num_classes,
                hidden_dim=max(1, d_model // 2),
                dropout=clf_dropout,
                use_local_mix=head_use_local_mix,
                local_kernel=head_local_kernel,
                norm=head_norm,
            )
        else:
            self.classifier = ResidualMLPClassifier(
                d_model=d_model,
                num_classes=num_classes,
                hidden_dim=max(1, d_model // 2),
                dropout=clf_dropout,
            )
        if self.use_confidence_head:
            self.confidence_head: ConfidenceHead | None = ConfidenceHead(
                d_model=d_model,
                hidden_dim=self.confidence_head_hidden_dim,
                dropout=clf_dropout,
            )
        else:
            self.confidence_head = None
        self._last_features: torch.Tensor | None = None
        self._last_confidence_logits: torch.Tensor | None = None
        self._last_confidence: torch.Tensor | None = None
        self._last_confidence_pred_classes: torch.Tensor | None = None
        self.input_preparer = ContextWaveformPreparer(in_ch, time_len)
        self.channel_embedding = LegacyChannelEmbeddingState(in_ch)
        self._init_readout_gate_bias()
        self._init_classifier_bias_uniform()
        self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook_compat)

    def _load_state_dict_pre_hook_compat(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Fill compatible legacy state and discard removed positional parameters."""
        state_dict.pop(f"{prefix}pos_scale", None)
        if isinstance(self.output_norm, nn.LayerNorm):
            weight_key = f"{prefix}output_norm.weight"
            bias_key = f"{prefix}output_norm.bias"
            if weight_key not in state_dict:
                state_dict[weight_key] = torch.ones(self.feature_dim)
            if bias_key not in state_dict:
                state_dict[bias_key] = torch.zeros(self.feature_dim)
        center_context_state = self.center_context_readout.state_dict()
        for name, value in center_context_state.items():
            full_key = f"{prefix}center_context_readout.{name}"
            if full_key not in state_dict:
                state_dict[full_key] = value.detach().clone()
        if isinstance(self.classifier, ResidualMLPClassifier):
            classifier_state = self.classifier.state_dict()
            legacy_to_new = {
                f"{prefix}classifier.0.weight": f"{prefix}classifier.input_proj.0.weight",
                f"{prefix}classifier.0.bias": f"{prefix}classifier.input_proj.0.bias",
                f"{prefix}classifier.3.weight": f"{prefix}classifier.head.weight",
                f"{prefix}classifier.3.bias": f"{prefix}classifier.head.bias",
            }
            for legacy_key, new_key in legacy_to_new.items():
                if legacy_key in state_dict and new_key not in state_dict:
                    state_dict[new_key] = state_dict[legacy_key].detach().clone()
                state_dict.pop(legacy_key, None)
            for name, value in classifier_state.items():
                full_key = f"{prefix}classifier.{name}"
                if full_key not in state_dict:
                    state_dict[full_key] = value.detach().clone()
        if self.confidence_head is not None:
            confidence_state = self.confidence_head.state_dict()
            for name, value in confidence_state.items():
                full_key = f"{prefix}confidence_head.{name}"
                if full_key not in state_dict:
                    state_dict[full_key] = value.detach().clone()

    def _init_classifier_bias_uniform(self):
        """Initialize classifier output layer bias to uniform prior (all classes equally likely)."""
        if isinstance(
            self.classifier,
            (ResidualMLPClassifier, PerPositionSleepHead),
        ):
            self.classifier.initialize_output_bias_uniform()
        elif isinstance(self.classifier, nn.Sequential):
            final_linear = cast(nn.Linear, self.classifier[-1])
            if hasattr(final_linear, "bias") and final_linear.bias is not None:
                nn.init.zeros_(final_linear.bias)

    def _init_readout_gate_bias(self) -> None:
        """Preserve the trained center-readout gate initialization."""
        gate = self.center_context_readout.gate
        if gate.bias is not None:
            nn.init.constant_(gate.bias, self.fusion_gate_bias)

    @property
    def enc_layers(self) -> nn.ModuleList:
        """Backward-compatible alias for legacy code that expects ``enc_layers``."""
        assert self.transformer is not None
        return self.transformer.layers

    def _apply_scaled_position_encoding(self, z: torch.Tensor) -> torch.Tensor:
        """Add legacy absolute positional encodings.

        Args:
            z: Input tensor of shape ``[B, L, D]``.

        Returns:
            Tensor of shape ``[B, L, D]`` with positional encoding added.
        """
        L = z.size(1)
        if L > self.pos_encoding.size(1):
            raise ValueError(
                f"context length {L} exceeds positional table length {self.pos_encoding.size(1)}"
            )
        return z + self.pos_encoding[:, :L, :]

    def _apply_center_context_readout(
        self,
        z: torch.Tensor,
        epoch_valid_mask: torch.Tensor | None,
        *,
        return_attention: bool = False,
    ) -> torch.Tensor:
        return self.center_context_readout(
            z, epoch_valid_mask=epoch_valid_mask, return_attention=return_attention
        )

    def _run_encoder(
        self,
        z: torch.Tensor,
        *,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run context attention and optionally capture last-layer weights."""
        if self.transformer is None:
            raise RuntimeError("Legacy Transformer encoder is unavailable")
        transformer = self.transformer
        diagnostic_padding_mask = (
            key_padding_mask.to(dtype=torch.bool)
            if key_padding_mask is not None
            else None
        )
        key_padding_mask = _safe_key_padding_mask(key_padding_mask)
        attention_weights: torch.Tensor | None = None
        if return_attention:
            for idx, layer in enumerate(transformer.layers):
                is_last = idx == len(transformer.layers) - 1
                if is_last:
                    z, attention_weights = _encoder_layer_forward_with_attention(
                        layer,
                        z,
                        attn_mask=attn_mask,
                        key_padding_mask=key_padding_mask,
                    )
                else:
                    z = layer(
                        z, src_mask=attn_mask, src_key_padding_mask=key_padding_mask
                    )
            z = self.output_norm(z)
            if attention_weights is not None and diagnostic_padding_mask is not None:
                attention_weights = attention_weights.masked_fill(
                    diagnostic_padding_mask[:, None, None, :], 0.0
                )
                attention_weights = attention_weights.masked_fill(
                    diagnostic_padding_mask[:, None, :, None], 0.0
                )
            return (z, attention_weights)
        z = transformer(z, mask=attn_mask, src_key_padding_mask=key_padding_mask)
        z = self.output_norm(z)
        return (z, attention_weights)

    def _compute_standard_head_outputs(
        self, classifier_input: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the default classifier path and optional confidence head."""
        logits = self.classifier(classifier_input)
        confidence: torch.Tensor | None = None
        if self.confidence_head is not None:
            confidence_logits = self.confidence_head(classifier_input)
            self._last_confidence_logits = confidence_logits
            confidence = torch.sigmoid(confidence_logits)
            self._last_confidence = confidence
        else:
            self._last_confidence_logits = None
            self._last_confidence = None
        self._last_confidence_pred_classes = None
        return (logits, confidence)

    def _apply_temperature_scaling(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply the configured fixed classifier temperature."""
        if self.classifier_temperature != 1.0:
            return logits / self.classifier_temperature
        return logits

    def forward_features(
        self,
        inputs: WaveformInputs,
        *,
        all_positions: bool = False,
        return_attention: bool = False,
    ) -> torch.Tensor:
        """Prepare inputs and return backbone features before classification."""
        prepared = self.input_preparer(inputs)
        return self._forward_features_prepared(
            prepared,
            inputs,
            all_positions=all_positions,
            return_attention=return_attention,
        )

    def _forward_features_prepared(
        self,
        prepared: PreparedContextWaveforms,
        inputs: WaveformInputs,
        *,
        all_positions: bool = False,
        return_attention: bool = False,
    ) -> torch.Tensor:
        """Encode waveforms, attend across epochs, and select the readout."""
        B, L, _, _ = prepared.waveform.shape
        feats = self.epoch_encoder(
            prepared.cnn_waveform,
            channel_mask=prepared.cnn_presence_mask,
            **self._encoder_recording_kwargs(prepared.recording_index, B, L),
        ).view(B, L, -1)
        feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        z = self._apply_scaled_position_encoding(self.proj(feats))
        epoch_valid = prepared.epoch_valid_mask
        with _sdp_kernel_context(self.sdp_backend):
            z, attn_weights = self._run_encoder(
                z, key_padding_mask=~epoch_valid, return_attention=return_attention
            )
        z = z * epoch_valid.to(z.dtype).unsqueeze(-1)
        self._last_attention_weights = attn_weights
        return (
            z
            if all_positions
            else self._apply_center_context_readout(
                z, epoch_valid, return_attention=return_attention
            )
        )

    def forward(
        self,
        inputs: WaveformInputs,
        predict_all: bool = False,
        return_features: bool = False,
        return_attention: bool = False,
        stage_labels: torch.Tensor | None = None,
    ) -> ForwardOutput:
        """Return stage logits and optional features and attention tensors.

        Args:
            inputs: Preprocessed floating ``[B, L, C, T]`` tensor with
                ``L=21, C=5, T=3840``, or a mapping with ``wave`` and masks as
                documented by ``ContextWaveformPreparer``. Use the model device.
            predict_all: Return logits for every context position instead of
                the center readout. Padded positions still require caller masking.
            return_features: Populate the output's ``features`` field.
            return_attention: If True, return attention weights from the last
                transformer encoder layer. The attention weights have shape
                [B, n_heads, seq_len, seq_len] where seq_len is the context length.
                Useful for attention visualization and interpretability analysis.
            stage_labels: Compatibility argument; ignored.

        Returns:
            ForwardOutput with ``logits`` shaped ``[B, 5]`` or ``[B, L, 5]``.
            Optional ``features`` are ``[B, d_model]`` or ``[B, L, d_model]``
            for center or all-position output. Attention and confidence fields
            depend on the requested outputs and checkpoint configuration.
            This method does not disable gradients; inference callers select
            evaluation mode and a no-grad context.
        """
        del stage_labels
        self._last_confidence_logits = None
        self._last_confidence = None
        self._last_confidence_pred_classes = None
        if hasattr(self.center_context_readout, "_last_attention_weights"):
            cast(Any, self.center_context_readout)._last_attention_weights = None
        need_all_positions = predict_all or return_features
        prepared = self.input_preparer(inputs)
        feats_result = self._forward_features_prepared(
            prepared,
            inputs,
            all_positions=need_all_positions,
            return_attention=return_attention,
        )
        feats = feats_result
        self._last_features = feats if need_all_positions and feats.dim() == 3 else None
        confidence: torch.Tensor | None = None
        if predict_all:
            classifier_input = feats
            center_feat = feats
        else:
            if feats.dim() == 3:
                center = self._apply_center_context_readout(
                    feats, prepared.epoch_valid_mask, return_attention=return_attention
                )
            else:
                center = feats
            classifier_input = center
            center_feat = center
        if isinstance(self.classifier, ResidualMLPClassifier):
            logits, confidence = self._compute_standard_head_outputs(classifier_input)
        else:
            logits = self.classifier(classifier_input)
        logits = self._apply_temperature_scaling(logits)
        if self._last_confidence_logits is not None:
            pred_classes = logits.argmax(dim=-1)
            if self.confidence_head_detach_target:
                pred_classes = pred_classes.detach()
            self._last_confidence_pred_classes = pred_classes
        attn_weights = self._last_attention_weights if return_attention else None
        readout_attn_weights = (
            getattr(self.center_context_readout, "_last_attention_weights", None)
            if return_attention
            else None
        )
        return ForwardOutput(
            logits=logits,
            features=center_feat if return_features else None,
            attention_weights=attn_weights,
            readout_attention_weights=readout_attn_weights,
            confidence=confidence,
        )

    def clear_last_attention_weights(self) -> None:
        """Discard cached attention weights to release graph references."""
        self._last_attention_weights = None
        if hasattr(self.center_context_readout, "_last_attention_weights"):
            cast(Any, self.center_context_readout)._last_attention_weights = None
        self._last_confidence_logits = None
        self._last_confidence = None
        self._last_confidence_pred_classes = None
