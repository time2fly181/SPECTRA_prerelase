"""Unified epoch-patch grid tokenization for a single context transformer.

Every other supervised path collapses the epoch encoder's ``T' = 240`` temporal
frames to **one vector per epoch** before the context transformer runs, which
destroys sub-epoch structure: where inside the epoch an arousal sits, or whether
the last seconds of epoch ``l-1`` continue into epoch ``l``.

This module instead lays the whole context window out as an ``L x (P+1)`` token
grid and hands it to the host's *existing* transformer, so one attention
operation reasons **within** an epoch and **across** epochs. There is no second
transformer and no intermediate pooling bottleneck.

Token layout is epoch-major with the per-epoch summary token first::

    flat index = l * (P + 1) + s          s = 0        -> summary token
                                         s = 1..P     -> patch tokens

:class:`EpochPatchGrid` owns **geometry only** — the tokenizer, the projection,
the position tables, the summary token and the epoch pool. It deliberately owns
no transformer, no epoch position table and no output norm; the host supplies
those. That is what lets ``TransformerContextNet`` and both pretrainers share
transformer weights 1:1, so an SSL checkpoint transfers into supervised training
without any key remapping.

Patch length is chosen from the encoder's measured behavior rather than from
event durations alone. For ``FlexibleAsymmetricEpochCNN`` at 128 Hz, 50 % of a
single output frame's response mass comes from 0.25 s of input and 90 % from
about 3 s, so patches shorter than ~3 s increasingly re-read the same
convolution. ``P = 10`` (3.0 s) sits at that resolution and also makes the three
sub-epoch AASM rules integral: an arousal (>=3 s) is one patch, the N3 criterion
(>=20 % = 6 s) is two, and the Wake alpha criterion (>50 % = 15 s) is five.

Usage::

    >>> import torch
    >>> from spectra.models import EpochPatchGrid
    >>> grid = EpochPatchGrid(d_model=512, cnn_dim=256, num_patches=10,
    ...                       temporal_len=240)
    >>> feat = torch.randn(2 * 21, 240, 256)      # [B*L, T', d_cnn]
    >>> epoch_pos = torch.randn(21, 512)
    >>> flat = grid.to_grid(feat, 21, epoch_pos)  # [2, 231, 512]
    >>> epoch_repr, patches = grid.from_grid(flat, 21)
    >>> epoch_repr.shape, patches.shape
    (torch.Size([2, 21, 512]), torch.Size([2, 21, 10, 512]))
"""

from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import kaiming_init_
from .patch_tokenization import PatchTokenizer

__all__ = ["EpochPatchGrid", "GridRelativePositionEncoderLayer"]


class EpochPatchGrid(nn.Module):
    """Tokenize a context window into an ``L x (P+1)`` epoch-patch token grid.

    Consumes the epoch encoder's pre-pool temporal features for every epoch in a
    context window and produces one flat token sequence for the host
    transformer, then reads per-epoch representations back out of it.

    The read-out is an attention pool over each epoch's contextualized patch
    tokens plus a **gated** contribution from that epoch's summary token::

        epoch_repr[l] = attn_pool(patch_out[l]) + summary_gate * summary_out[l]

    ``summary_gate`` is initialized to zero, so at initialization the read-out is
    exactly the attention pool. That keeps the module numerically continuous with
    the pooling math used by the pretrainers, and lets the summary token grow in
    only if it earns gradient. The summary token still participates in attention
    from step one, so it shapes the patch tokens even while the gate is zero.

    Args:
        d_model: Host transformer width.
        cnn_dim: Channel dimension of the encoder's temporal features. This must
            be the encoder's ``temporal_dim`` (pre-pool width), not ``out_dim``.
        num_patches: Patches ``P`` per epoch. Must divide ``temporal_len``.
        temporal_len: Encoder temporal length ``T'`` (``final_temporal_len``).
        patch_token_dim: Tokenizer output width before projecting to ``d_model``
            (``None`` -> ``d_model``).
        dropout: Dropout applied inside the tokenizer and to the assembled grid.

    Raises:
        ValueError: If ``num_patches`` does not divide ``temporal_len``, or if
            either dimension is non-positive.
    """

    def __init__(
        self,
        *,
        d_model: int,
        cnn_dim: int,
        num_patches: int,
        temporal_len: int,
        patch_token_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model < 1:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if cnn_dim < 1:
            raise ValueError(f"cnn_dim must be positive, got {cnn_dim}")

        d_token = int(patch_token_dim or d_model)
        self.d_model = int(d_model)
        self.cnn_dim = int(cnn_dim)
        self.num_patches = int(num_patches)
        self.temporal_len = int(temporal_len)
        self.patch_token_dim = d_token
        # Slot 0 is the summary token; slots 1..P are the patch tokens.
        self.slots_per_epoch = self.num_patches + 1

        self.patch_tokenizer = PatchTokenizer(
            d_cnn=self.cnn_dim,
            d_token=d_token,
            temporal_len=self.temporal_len,
            num_patches=self.num_patches,
            dropout=dropout,
        )
        self.patch_proj = nn.Linear(d_token, self.d_model)
        self.patch_epoch_pool = nn.Linear(self.d_model, 1)
        self.apply(kaiming_init_)

        # Within-epoch slot identity. Kept absolute in both attention modes: the
        # summary token has to stay distinguishable from a patch, and a slot's
        # offset inside the epoch is well defined regardless of where the window
        # sits in the night.
        self.slot_pos = nn.Parameter(
            torch.randn(1, self.slots_per_epoch, self.d_model) * 0.02
        )
        self.summary_token = nn.Parameter(torch.randn(1, 1, 1, self.d_model) * 0.02)
        self.summary_gate = nn.Parameter(torch.zeros(()))

        self.grid_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, cnn_dim={self.cnn_dim}, "
            f"num_patches={self.num_patches}, temporal_len={self.temporal_len}, "
            f"slots_per_epoch={self.slots_per_epoch}"
        )

    @property
    def patch_seconds(self) -> float:
        """Patch duration in seconds, assuming a 30 s epoch."""
        return 30.0 / float(self.num_patches)

    def summary_indices(
        self,
        seq_len: int,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Flat token indices of the per-epoch summary tokens.

        Args:
            seq_len: Context length ``L``.
            device: Device for the returned index tensor.

        Returns:
            Long tensor of shape ``[L]``.
        """
        return torch.arange(seq_len, device=device) * self.slots_per_epoch

    def expand_padding_mask(self, epoch_valid: torch.Tensor) -> torch.Tensor:
        """Expand a per-epoch validity mask to the flat token grid.

        Args:
            epoch_valid: Boolean availability mask ``[B, L]`` where ``True``
                marks a usable epoch.

        Returns:
            Key-padding mask ``[B, L*(P+1)]`` where ``True`` marks a padded
            token, matching the ``src_key_padding_mask`` convention.
        """
        if epoch_valid.dim() != 2:
            raise ValueError(
                f"epoch_valid must be [B, L], got {tuple(epoch_valid.shape)}"
            )
        valid = epoch_valid.to(dtype=torch.bool)
        return ~valid.repeat_interleave(self.slots_per_epoch, dim=1)

    def tokenize(self, feat: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Tokenize encoder temporal features into per-epoch patch tokens.

        This is the first half of :meth:`to_grid`, split out so callers that need
        to mask between tokenization and position assembly (the temporal
        pretrainer's student/teacher split) can do so without duplicating the
        geometry.

        Args:
            feat: Encoder temporal features ``[B*L, T', cnn_dim]``.
            seq_len: Context length ``L``.

        Returns:
            Patch tokens ``[B, L, P, d_model]``. No summary token, no positions.

        Raises:
            ValueError: On a shape mismatch or an indivisible batch.
        """
        if feat.dim() != 3:
            raise ValueError(
                f"feat must be [B*L, T', cnn_dim], got {tuple(feat.shape)}"
            )
        if seq_len < 1:
            raise ValueError(f"seq_len must be positive, got {seq_len}")
        n = feat.size(0)
        if n % seq_len != 0:
            raise ValueError(f"feat batch {n} not divisible by seq_len {seq_len}")
        if feat.size(1) != self.temporal_len:
            raise ValueError(
                f"feat temporal length {feat.size(1)} does not match the "
                f"configured temporal_len {self.temporal_len}"
            )
        tokens = self.patch_proj(self.patch_tokenizer(feat))  # [B*L, P, d_model]
        return tokens.view(n // seq_len, seq_len, self.num_patches, self.d_model)

    def assemble(
        self,
        tokens: torch.Tensor,
        epoch_pos: torch.Tensor | None = None,
        *,
        center_mask: torch.Tensor | None = None,
        center_mask_token: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Prepend summary tokens, add positions, and flatten to one sequence.

        Args:
            tokens: Patch tokens ``[B, L, P, d_model]``.
            epoch_pos: Epoch position table ``[>=L, d_model]``, or ``None`` to add
                no absolute epoch position (used when a relative attention bias
                supplies the epoch axis instead).
            center_mask: Optional ``[B]`` boolean mask; ``True`` replaces **all**
                ``P+1`` slots of the center epoch with ``center_mask_token``.
                Masking only the summary slot would leak the center epoch
                through its own patch tokens.
            center_mask_token: Replacement token broadcastable to
                ``[B, 1, 1, d_model]``. Required when ``center_mask`` is given.

        Returns:
            Flat token grid ``[B, L*(P+1), d_model]`` in epoch-major order.

        Raises:
            ValueError: On a shape mismatch, or a mask without its token.
        """
        if tokens.dim() != 4 or tokens.size(2) != self.num_patches:
            raise ValueError(
                f"tokens must be [B, L, {self.num_patches}, {self.d_model}], "
                f"got {tuple(tokens.shape)}"
            )
        b, seq_len = tokens.shape[0], tokens.shape[1]
        d = self.d_model

        summary = self.summary_token.to(dtype=tokens.dtype).expand(b, seq_len, 1, d)
        grid = torch.cat((summary, tokens), dim=2)  # [B, L, P+1, d_model]

        if center_mask is not None:
            if center_mask_token is None:
                raise ValueError("center_mask requires center_mask_token")
            if center_mask.dim() != 1 or center_mask.size(0) != b:
                raise ValueError(
                    f"center_mask must be [{b}], got {tuple(center_mask.shape)}"
                )
            center = seq_len // 2
            # Replace every slot of the center epoch, summary included.
            selector = torch.zeros(
                (1, seq_len, 1, 1), dtype=torch.bool, device=grid.device
            )
            selector[:, center] = True
            replace = center_mask.view(b, 1, 1, 1) & selector
            grid = torch.where(
                replace,
                center_mask_token.to(dtype=grid.dtype).view(-1, 1, 1, d),
                grid,
            )

        grid = grid + self.slot_pos.to(dtype=grid.dtype).unsqueeze(0)
        if epoch_pos is not None:
            if epoch_pos.dim() != 2 or epoch_pos.size(0) < seq_len:
                raise ValueError(
                    "epoch_pos must be [>=L, d_model], got "
                    f"{tuple(epoch_pos.shape)} for seq_len {seq_len}"
                )
            grid = grid + epoch_pos[:seq_len].to(dtype=grid.dtype).view(
                1, seq_len, 1, d
            )

        flat = grid.reshape(b, seq_len * self.slots_per_epoch, d)
        return cast(torch.Tensor, self.grid_dropout(flat))

    def to_grid(
        self,
        feat: torch.Tensor,
        seq_len: int,
        epoch_pos: torch.Tensor | None = None,
        *,
        patch_mask: torch.Tensor | None = None,
        patch_mask_token: torch.Tensor | None = None,
        center_mask: torch.Tensor | None = None,
        center_mask_token: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Assemble the flat token grid from encoder temporal features.

        Composes :meth:`tokenize`, optional patch masking, and :meth:`assemble`.

        Args:
            feat: Encoder temporal features ``[B*L, T', cnn_dim]``.
            seq_len: Context length ``L``.
            epoch_pos: Epoch position table ``[>=L, d_model]``, or ``None``.
            patch_mask: Optional ``[B, L, P]`` boolean mask; ``True`` replaces
                that patch token with ``patch_mask_token``. Applied before
                positions are added. Summary tokens are never masked.
            patch_mask_token: Replacement token broadcastable to
                ``[B, L, P, d_model]``. Required when ``patch_mask`` is given.
            center_mask: Optional ``[B]`` boolean mask over the center epoch.
            center_mask_token: Replacement token for ``center_mask``.

        Returns:
            Flat token grid ``[B, L*(P+1), d_model]`` in epoch-major order.

        Raises:
            ValueError: On a shape mismatch, an indivisible batch, or a mask
                supplied without its replacement token.
        """
        tokens = self.tokenize(feat, seq_len)
        if patch_mask is not None:
            if patch_mask_token is None:
                raise ValueError("patch_mask requires patch_mask_token")
            expected = (tokens.size(0), seq_len, self.num_patches)
            if tuple(patch_mask.shape) != expected:
                raise ValueError(
                    f"patch_mask must be {expected}, got {tuple(patch_mask.shape)}"
                )
            tokens = torch.where(
                patch_mask.to(dtype=torch.bool).unsqueeze(-1),
                patch_mask_token.to(dtype=tokens.dtype),
                tokens,
            )
        return self.assemble(
            tokens,
            epoch_pos,
            center_mask=center_mask,
            center_mask_token=center_mask_token,
        )

    def from_grid(
        self,
        flat_out: torch.Tensor,
        seq_len: int,
        *,
        epoch_valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read per-epoch representations back out of the contextualized grid.

        Args:
            flat_out: Transformer output ``[B, L*(P+1), d_model]``.
            seq_len: Context length ``L``.
            epoch_valid: Optional ``[B, L]`` mask; invalid epochs are zeroed
                before pooling so padded epochs contribute nothing.

        Returns:
            Tuple of per-epoch representations ``[B, L, d_model]`` and the
            post-transformer patch tokens ``[B, L, P, d_model]``. The patch
            tensor excludes summary tokens, so it keeps the ``[B, L, P, ...]``
            shape that the pretraining losses and spectral targets expect.

        Raises:
            ValueError: If the sequence length is not ``L*(P+1)``.
        """
        if flat_out.dim() != 3:
            raise ValueError(
                f"flat_out must be [B, S, d_model], got {tuple(flat_out.shape)}"
            )
        expected = seq_len * self.slots_per_epoch
        if flat_out.size(1) != expected:
            raise ValueError(
                f"flat_out length {flat_out.size(1)} does not match "
                f"seq_len*{self.slots_per_epoch} = {expected}"
            )
        b = flat_out.size(0)
        grid = flat_out.view(b, seq_len, self.slots_per_epoch, self.d_model)
        summary_out = grid[:, :, 0, :]  # [B, L, d_model]
        patch_out = grid[:, :, 1:, :]  # [B, L, P, d_model]

        if epoch_valid is not None:
            keep = epoch_valid.to(dtype=patch_out.dtype)[:, :, None, None]
            patch_out = patch_out * keep

        weights = torch.softmax(
            self.patch_epoch_pool(patch_out).squeeze(-1), dim=-1
        ).unsqueeze(
            -1
        )  # [B, L, P, 1]
        pooled = (weights * patch_out).sum(dim=2)  # [B, L, d_model]
        epoch_repr = pooled + self.summary_gate.to(dtype=pooled.dtype) * summary_out
        return epoch_repr, patch_out

    @torch.no_grad()
    def load_transfer_state(
        self, state: dict[str, object]
    ) -> tuple[list[str], list[str]]:
        """Load a pretrainer's grid export into this module.

        Args:
            state: The ``epoch_grid`` sub-dictionary of a pretrainer's
                ``get_transformer_state_dict()`` export.

        Returns:
            ``(missing, unexpected)`` parameter-key lists.
        """
        own = self.state_dict()
        missing = [k for k in own if k not in state]
        unexpected = [k for k in state if k not in own]
        loadable = {k: cast(torch.Tensor, v) for k, v in state.items() if k in own}
        for key, tensor in loadable.items():
            if tuple(own[key].shape) != tuple(tensor.shape):
                raise ValueError(
                    f"epoch_grid.{key} shape mismatch: model has "
                    f"{tuple(own[key].shape)}, checkpoint has {tuple(tensor.shape)}"
                )
        self.load_state_dict(loadable, strict=False)
        return missing, unexpected


class GridRelativePositionEncoderLayer(nn.TransformerEncoderLayer):
    """Encoder layer with a factorized 2-D relative bias over the token grid.

    Adds a per-head additive bias that is translation-equivariant on the epoch
    axis and on the within-epoch slot axis::

        bias[h, i, j] = epoch_lag[h, dl] + slot_lag[h, ds]
                        + type_bias[h, is_summary(i), is_summary(j)]

    where ``dl`` and ``ds`` are the signed epoch and slot offsets between the
    query and key tokens. ``type_bias`` is required rather than decorative:
    without it a summary-to-patch-``p`` pair and a patch-0-to-patch-``p`` pair
    share the same ``slot_lag`` bucket, which conflates two structurally
    different relations.

    Only the bias tables are added; ``self_attn``, ``linear1``/``linear2`` (or
    ``w_gate``/``w_up``/``w_down``), ``norm1`` and ``norm2`` keep the names
    ``nn.TransformerEncoderLayer`` uses, so transformer weights transfer freely
    between this layer and the absolute-position layers.

    The bias is materialized as a ``[B*nhead, S, S]`` float attention mask, which
    forfeits the flash SDPA backend. At ``P = 10`` and ``L = 21`` that is about
    1.7 MB per sample per layer in fp32; at ``P = 20`` it is roughly four times
    that and should be measured before use.

    Args:
        d_model: Token feature dimension.
        nhead: Number of self-attention heads.
        dim_feedforward: GELU hidden width or SwiGLU gate/up hidden width.
        dropout: Attention, FFN, and residual dropout probability.
        num_patches: Patches ``P`` per epoch, so each epoch owns ``P+1`` slots.
        max_epochs: Maximum supported context-window length in epochs.
        activation: Activation used by the checkpoint-compatible GELU path.
        ffn_activation: Feed-forward architecture, ``gelu`` or ``swiglu``.
        batch_first: Whether inputs use ``[batch, length, feature]`` layout.
        norm_first: Whether to apply pre-norm Transformer residual blocks.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        *,
        num_patches: int,
        max_epochs: int = 21,
        activation: Any = F.gelu,
        ffn_activation: str = "gelu",
        batch_first: bool = True,
        norm_first: bool = True,
    ) -> None:
        if max_epochs < 1:
            raise ValueError("max_epochs must be positive")
        if num_patches < 1:
            raise ValueError("num_patches must be positive")
        if ffn_activation not in {"gelu", "swiglu"}:
            raise ValueError("ffn_activation must be 'gelu' or 'swiglu'")
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=batch_first,
            norm_first=norm_first,
        )
        self.ffn_activation = ffn_activation
        if ffn_activation == "swiglu":
            # Match the established pretraining key schema without carrying
            # unused GELU parameters.
            del self.linear1
            del self.linear2
            self.w_gate = nn.Linear(d_model, dim_feedforward, bias=False)
            self.w_up = nn.Linear(d_model, dim_feedforward, bias=False)
            self.w_down = nn.Linear(dim_feedforward, d_model, bias=False)
            self.ffn_dropout = nn.Dropout(dropout)
        self.nhead = int(nhead)
        self.num_patches = int(num_patches)
        self.slots_per_epoch = self.num_patches + 1
        self.max_epochs = int(max_epochs)
        self.epoch_lag_bias = nn.Parameter(torch.zeros(nhead, 2 * self.max_epochs - 1))
        self.slot_lag_bias = nn.Parameter(
            torch.zeros(nhead, 2 * self.slots_per_epoch - 1)
        )
        self.type_bias = nn.Parameter(torch.zeros(nhead, 2, 2))

    def _grid_mask(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None,
        src_key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch, length, _ = src.shape
        slots = self.slots_per_epoch
        if length % slots != 0:
            raise ValueError(
                f"grid length {length} is not a multiple of slots_per_epoch "
                f"{slots}; the sequence must be L*(P+1) tokens"
            )
        epochs = length // slots
        if epochs > self.max_epochs:
            raise ValueError(
                f"context length {epochs} epochs exceeds configured maximum "
                f"{self.max_epochs}"
            )

        index = torch.arange(length, device=src.device)
        epoch_index = index // slots
        slot_index = index % slots
        # Signed key-minus-query offsets, matching the epoch-mode convention.
        epoch_lag = epoch_index[None, :] - epoch_index[:, None]
        slot_lag = slot_index[None, :] - slot_index[:, None]
        bias = self.epoch_lag_bias[:, epoch_lag + self.max_epochs - 1]
        bias = bias + self.slot_lag_bias[:, slot_lag + self.num_patches]

        is_summary = (slot_index == 0).long()
        query_type = is_summary[:, None].expand(length, length)
        key_type = is_summary[None, :].expand(length, length)
        bias = bias + self.type_bias[:, query_type, key_type]

        mask = bias.to(dtype=src.dtype).unsqueeze(0).expand(batch, -1, -1, -1).clone()

        if src_mask is not None:
            supplied = src_mask.to(device=src.device)
            if supplied.dtype == torch.bool:
                supplied_float = torch.zeros_like(supplied, dtype=src.dtype)
                supplied_float.masked_fill_(supplied, float("-inf"))
                supplied = supplied_float
            else:
                supplied = supplied.to(dtype=src.dtype)
            if supplied.dim() == 2:
                mask = mask + supplied[None, None, :, :]
            elif supplied.dim() == 3 and supplied.size(0) == batch * self.nhead:
                mask = mask + supplied.view(batch, self.nhead, length, length)
            else:
                raise ValueError(
                    "src_mask must be [S,S] or [B*nhead,S,S] for grid attention"
                )

        # Imported lazily to avoid a circular import with context_transformer.
        from .context_transformer import _safe_key_padding_mask

        safe_padding = _safe_key_padding_mask(src_key_padding_mask)
        if safe_padding is not None:
            mask.masked_fill_(safe_padding[:, None, None, :], float("-inf"))
        return mask.reshape(batch * self.nhead, length, length)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        del is_causal
        from .context_transformer import _encoder_layer_forward_explicit

        grid_mask = self._grid_mask(src, src_mask, src_key_padding_mask)
        if self.ffn_activation == "swiglu" or not self.training:
            return _encoder_layer_forward_explicit(
                self,
                src,
                attn_mask=grid_mask,
                key_padding_mask=None,
            )
        return super().forward(
            src,
            src_mask=grid_mask,
            src_key_padding_mask=None,
            is_causal=False,
        )

    def forward_with_attention(
        self,
        src: torch.Tensor,
        *,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from .context_transformer import _encoder_layer_forward_with_attention

        grid_mask = self._grid_mask(src, src_mask, src_key_padding_mask)
        return _encoder_layer_forward_with_attention(
            self,
            src,
            attn_mask=grid_mask,
            key_padding_mask=None,
        )
