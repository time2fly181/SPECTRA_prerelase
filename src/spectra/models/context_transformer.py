"""
Transformer-based context model for PSG sleep staging.
"""

from __future__ import annotations

import inspect
import logging
import math
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

if TYPE_CHECKING:
    from .channel_config import CanonicalChannelSet
from .channel_embedding import LegacyChannelEmbeddingState
from .common import (
    _SDP_BACKEND_CHOICES,
    _sdp_kernel_context,
    resolve_sampling_params,
    transformer_init_,
)
from .context_input import (
    ContextWaveformPreparer,
    PreparedContextWaveforms,
    WaveformInputs,
)
from .epoch_patch_grid import EpochPatchGrid, GridRelativePositionEncoderLayer
from .feature_extraction import (
    BottleneckProjection,
    HybridFeatureExtractor,
    N1FocusedAttention,
    SequentialFeatureFusion,
    SleepFeatureExtractor,
)
from .multirate_asymmetric_epoch_cnn import MultiRateAsymmetricEpochCNN
from .objectives import (
    RELATIVE_SLOW_WAVE_TARGET_VERSION,
    RELATIVE_SLOW_WAVE_THRESHOLDS,
    RelativeSlowWaveOccupancyHead,
)

EPOCH_ENCODER_VARIANTS = {"multirate_asymmetric": MultiRateAsymmetricEpochCNN}
EPOCH_ENCODER_KWARG_PASSTHROUGH = {
    "multirate_asymmetric": "multirate_asymmetric_encoder_kwargs"
}
EPOCH_ENCODER_PASSTHROUGH_RESERVED = frozenset(
    {"in_ch", "time_len", "dropout", "norm", "widths", "fs"}
)


@contextmanager
def temporary_eval_mode(module: nn.Module):
    """Run a module in eval mode with gradients disabled, then restore its state."""
    was_training = module.training
    module.eval()
    try:
        with torch.no_grad():
            yield module
    finally:
        module.train(was_training)


def _infer_cnn_downsample_stages(
    epoch_encoder_variant: str, epoch_encoder: nn.Module
) -> int:
    """Infer how many ceil-div2 downsampling stages an encoder applies."""
    enc = epoch_encoder
    if hasattr(enc, "encoder"):
        enc = cast(nn.Module, enc.encoder)
    if hasattr(enc, "cnn_extractor"):
        enc = cast(nn.Module, enc.cnn_extractor)
    factor = getattr(enc, "total_downsample_factor", None)
    if isinstance(factor, int) and factor >= 1:
        return max(factor.bit_length() - 1, 0)
    return 3


@dataclass
class ForwardOutput:
    """
    Structured output from TransformerContextNet.forward().

    Using a dataclass instead of tuples provides:
    - Self-documenting field names
    - Consistent return structure regardless of which outputs are requested
    - Easier to extend with new output fields
    - Better IDE autocomplete and type checking

    Attributes:
        logits: Class prediction logits, shape [B, num_classes] or [B, L, num_classes].
        features: Extracted features from the center epoch or all epochs.
            Shape [B, d_model] or [B, L, d_model]. None if not requested.
        attention_weights: Attention weights from the last transformer layer.
            Shape [B, H, L, L]. None if not requested.
        reasoning: Chain-of-thought reasoning trace from CoT classifier.
            None if not using CoT or not requested.
        confidence: Predicted probability that the stage prediction is correct.
            Shape [B] or [B, L]. None if the confidence head is disabled.
        recurrent_logits: Per-step all-position predictions from recurrent
            refinement. None when refinement or ``predict_all`` is disabled.
    """

    logits: torch.Tensor
    features: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None
    attention_weights: torch.Tensor | None = None
    readout_attention_weights: torch.Tensor | None = None
    reasoning: torch.Tensor | None = None
    confidence: torch.Tensor | None = None
    recurrent_logits: tuple[torch.Tensor, ...] | None = None


class TemporalContextHead(nn.Module):
    """Predict neighbor stages from center representation.

    This auxiliary head encourages the transformer to encode temporal context
    by requiring the center epoch's representation to predict neighboring epochs'
    sleep stages. If the center position doesn't contain neighbor information,
    this task will fail.

    Args:
        d_model: Transformer feature dimension.
        num_classes: Number of sleep stage classes (default 5).
        n_neighbors: Number of neighbors on each side to predict (default 2).
            With n_neighbors=2, predicts epochs at positions -2, -1, +1, +2
            relative to center.
    """

    def __init__(self, d_model: int, num_classes: int = 5, n_neighbors: int = 2):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.num_classes = num_classes
        self.neighbor_heads = nn.ModuleList(
            [nn.Linear(d_model, num_classes) for _ in range(2 * n_neighbors)]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Predict neighbor stages from center representation.

        Args:
            z: Transformer output [B, L, d_model].

        Returns:
            Neighbor predictions [B, 2*n_neighbors, num_classes].
        """
        B, L, D = z.shape
        center = L // 2
        center_repr = z[:, center, :]
        preds = []
        for head in self.neighbor_heads:
            preds.append(head(center_repr))
        return torch.stack(preds, dim=1)


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
        norm: ``"layernorm"`` (default) or ``"rmsnorm"`` (variance-preserving).
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
        if norm == "rmsnorm":
            from spectra.models.feature_extraction import VariancePreservingRMSNorm

            self.norm: nn.Module = VariancePreservingRMSNorm(self.d_model)
        elif norm == "layernorm":
            self.norm = nn.LayerNorm(self.d_model)
        else:
            raise ValueError(
                f"Unknown norm '{norm}' (expected 'layernorm' or 'rmsnorm')"
            )
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


CLASSIFIER_HEAD_CHOICES: tuple[str, ...] = ("residual_mlp", "linear")


def validate_classifier_head_selection(
    classifier_head: str, use_per_position_head: bool
) -> None:
    """Validate a classifier-head selection against the per-position head flag.

    ``classifier_head`` and ``use_per_position_head`` both select *the* single
    classification head, so requesting the linear head together with the
    per-position head is ambiguous rather than additive.

    Args:
        classifier_head: Requested head family, one of
            :data:`CLASSIFIER_HEAD_CHOICES`.
        use_per_position_head: Whether the per-position head was requested.

    Raises:
        ValueError: If ``classifier_head`` is not a recognized choice, or if
            ``classifier_head="linear"`` is combined with
            ``use_per_position_head=True``.
    """
    if classifier_head not in CLASSIFIER_HEAD_CHOICES:
        raise ValueError(
            f"classifier_head must be one of {CLASSIFIER_HEAD_CHOICES}, got {classifier_head!r}"
        )
    if classifier_head == "linear" and use_per_position_head:
        raise ValueError(
            "classifier_head='linear' and use_per_position_head=True both select the classification head, so they cannot be combined. The linear head is already position-agnostic and weight-shared (it applies the same LayerNorm+Linear at every epoch position), so use it on its own with --all_positions_loss. Choose classifier_head='residual_mlp' if you want the per-position MLP head instead."
        )


class LinearSleepHead(nn.Module):
    """Linear-probe classification head for contextualized epoch representations.

    Maps ``[B, L, d] -> [B, L, C]`` (or ``[B, d] -> [B, C]``) with
    ``LayerNorm(d) -> Linear(d, C)`` and nothing else. This is the linear-probe
    readout protocol used across the sleep-staging and EEG foundation-model
    literature (SleepFM's logistic-regression probe, the EEG-FM benchmark
    protocol), and it is the shallow arm of a classifier-head depth ablation
    against :class:`ResidualMLPClassifier` (~74x more head parameters at
    ``d_model=512``: 266,245 vs 3,589).

    There is deliberately no dropout. A dropout layer immediately before a
    ``d -> C`` projection perturbs features rather than regularizing a decision
    boundary; literature linear probes omit it; and ~3.6k head parameters
    against millions of labeled epochs do not present the overfitting risk that
    dropout would address. Note the consequence for MC-dropout inference: this
    head contributes no stochastic layers, so predictive-variance magnitudes are
    not comparable against a :class:`ResidualMLPClassifier` run.

    The ``LayerNorm`` is retained for *conditioning*, not regularization: it
    holds the classifier's input scale fixed so that a head-depth ablation
    varies depth alone. It also does real work on the ``learned_feature_axial_v2``
    path, where the backbone's ``output_norm`` is an ``nn.Identity``.

    Like :class:`PerPositionSleepHead`, this head is position-agnostic and
    weight-shared, so it is valid under ``--all_positions_loss`` and under
    offset-averaged inference with no further configuration. Both ``LayerNorm``
    and ``Linear`` act on the trailing axis, so it is rank-polymorphic without
    the reshape :meth:`ResidualMLPClassifier.forward_features` needs.

    Args:
        d_model: Feature dimension of the inter-epoch representation.
        num_classes: Number of sleep-stage classes.

    Attributes:
        norm: ``LayerNorm(d_model)`` applied before the projection.
        head: The ``Linear(d_model, num_classes)`` projection.

    Note:
        The submodule names ``norm`` and ``head`` are load-bearing. Optimizer
        param grouping matches ``no_decay`` keywords as lowercased substrings of
        the full parameter name, so these names are what place
        ``classifier.head.weight`` in the decayed head group at the head LR
        multiplier while ``classifier.norm.*`` and ``classifier.head.bias`` stay
        undecayed -- the parity a depth-only ablation requires. A wrapper
        attribute containing ``norm``/``ln``/``pos``/``embed`` around the
        ``Linear`` would silently move its weight out of the decayed group.
        ``head`` is additionally the attribute the bias-initialization helpers
        look for.
    """

    def __init__(self, d_model: int, num_classes: int) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.num_classes = int(num_classes)
        self.norm = nn.LayerNorm(self.d_model)
        self.head = nn.Linear(self.d_model, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map context features to logits.

        Args:
            x: ``[B, L, d]`` (all positions) or ``[B, d]`` (center only).

        Returns:
            ``[B, L, C]`` or ``[B, C]``, matching the input rank.
        """
        return self.head(self.norm(x))

    def initialize_output_bias_uniform(self) -> None:
        """Zero the output bias, giving a uniform prior over classes."""
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def set_output_bias_from_log_priors(self, log_priors: torch.Tensor) -> None:
        """Set the output bias to log class priors.

        Args:
            log_priors: ``[num_classes]`` tensor of log prior probabilities.
        """
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

    ``dim_feedforward`` is the gate/up hidden width directly. Consequently,
    matching the parameter count of a two-projection GELU FFN requires a
    SwiGLU width near two thirds of the GELU width.

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


def _encoder_layer_forward_explicit(
    layer: nn.Module,
    src: torch.Tensor,
    *,
    attn_mask: torch.Tensor | None,
    key_padding_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Run an encoder layer without PyTorch's fused eval fast path.

    PyTorch 2.9's fused transformer path produces NaNs when a three-dimensional
    additive attention mask contains learned, nonzero relative-position biases.
    Calling the constituent attention and feed-forward blocks explicitly keeps
    the mask additive and preserves the ordinary TransformerEncoderLayer math.
    """
    typed_layer = cast(Any, layer)

    def self_attention(x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = typed_layer.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            is_causal=False,
        )
        return typed_layer.dropout1(attn_out)

    if getattr(layer, "norm_first", True):
        src = src + self_attention(typed_layer.norm1(src))
        return src + _encoder_layer_feed_forward(layer, typed_layer.norm2(src))
    src = typed_layer.norm1(src + self_attention(src))
    return typed_layer.norm2(src + _encoder_layer_feed_forward(layer, src))


class RelativePositionTransformerEncoderLayer(nn.TransformerEncoderLayer):
    """Pre-norm encoder layer with signed relative bias and GELU or SwiGLU.

    Args:
        d_model: Token feature dimension.
        nhead: Number of self-attention heads.
        dim_feedforward: GELU hidden width or SwiGLU gate/up hidden width.
        dropout: Attention, FFN, and residual dropout probability.
        max_context_epochs: Maximum supported context-window length.
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
        max_context_epochs: int = 21,
        activation: Any = F.gelu,
        ffn_activation: str = "gelu",
        batch_first: bool = True,
        norm_first: bool = True,
    ) -> None:
        if max_context_epochs < 1:
            raise ValueError("max_context_epochs must be positive")
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
            del self.linear1
            del self.linear2
            self.w_gate = nn.Linear(d_model, dim_feedforward, bias=False)
            self.w_up = nn.Linear(d_model, dim_feedforward, bias=False)
            self.w_down = nn.Linear(dim_feedforward, d_model, bias=False)
            self.ffn_dropout = nn.Dropout(dropout)
        self.nhead = int(nhead)
        self.max_context_epochs = int(max_context_epochs)
        self.relative_position_bias = nn.Parameter(
            torch.zeros(nhead, 2 * max_context_epochs - 1)
        )

    def _relative_mask(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None,
        src_key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch, length, _ = src.shape
        if length > self.max_context_epochs:
            raise ValueError(
                f"context length {length} exceeds configured maximum {self.max_context_epochs}"
            )
        positions = torch.arange(length, device=src.device)
        signed_lag = positions[None, :] - positions[:, None]
        indices = signed_lag + self.max_context_epochs - 1
        bias = self.relative_position_bias[:, indices].to(dtype=src.dtype)
        mask = bias.unsqueeze(0).expand(batch, -1, -1, -1).clone()
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
                    "src_mask must be [L,L] or [B*nhead,L,L] for relative attention"
                )
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
        relative_mask = self._relative_mask(src, src_mask, src_key_padding_mask)
        if self.ffn_activation == "swiglu" or not self.training:
            return _encoder_layer_forward_explicit(
                self, src, attn_mask=relative_mask, key_padding_mask=None
            )
        return super().forward(
            src, src_mask=relative_mask, src_key_padding_mask=None, is_causal=False
        )

    def forward_with_attention(
        self,
        src: torch.Tensor,
        *,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        relative_mask = self._relative_mask(src, src_mask, src_key_padding_mask)
        return _encoder_layer_forward_with_attention(
            self, src, attn_mask=relative_mask, key_padding_mask=None
        )


class RecurrentContextRefiner(nn.Module):
    """Iteratively refine context tokens with one shared Transformer cell.

    The base context representation is reintroduced at every step so the shared
    cell refines fixed evidence instead of repeatedly transforming an
    increasingly detached latent state. Invalid epochs remain zeroed throughout
    refinement.

    Args:
        d_model: Token feature dimension.
        nhead: Number of self-attention heads.
        dim_feedforward: Feed-forward hidden width.
        dropout: Attention, feed-forward, and residual dropout probability.
        steps: Number of recurrent refinement steps.
        max_context_epochs: Maximum supported context-window length.
        attention_mode: Absolute or signed-relative temporal attention.
        ffn_activation: Feed-forward architecture, ``gelu`` or ``swiglu``.
    """

    def __init__(
        self,
        *,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        steps: int,
        max_context_epochs: int,
        attention_mode: str,
        ffn_activation: str,
    ) -> None:
        super().__init__()
        if steps < 1:
            raise ValueError("steps must be positive")
        if attention_mode == "relative_full":
            cell: nn.Module = RelativePositionTransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
                activation=nn.GELU(approximate="tanh"),
                ffn_activation=ffn_activation,
                max_context_epochs=max_context_epochs,
            )
        elif ffn_activation == "swiglu":
            cell = SwiGLUTransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
        else:
            cell = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
                activation=nn.GELU(approximate="tanh"),
            )
        self.cell = cell
        self.output_norm = nn.LayerNorm(d_model)
        self.evidence_gate = nn.Parameter(torch.full((d_model,), -2.0))
        self.update_gate = nn.Parameter(torch.full((d_model,), -2.0))
        self.register_buffer(
            "configured_steps", torch.tensor(steps, dtype=torch.int64), persistent=True
        )
        self.steps = int(steps)
        self.apply(transformer_init_)

    def forward(
        self,
        evidence: torch.Tensor,
        epoch_valid_mask: torch.Tensor | None = None,
        steps: int | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Return the refined all-position representation after every step."""
        if evidence.dim() != 3:
            raise ValueError(f"Expected evidence [B,L,D], got {tuple(evidence.shape)}")
        batch, length, _ = evidence.shape
        valid = (
            torch.ones(batch, length, dtype=torch.bool, device=evidence.device)
            if epoch_valid_mask is None
            else epoch_valid_mask.to(device=evidence.device, dtype=torch.bool)
        )
        if valid.shape != (batch, length):
            raise ValueError(
                f"epoch_valid_mask must have shape {(batch, length)}, got {tuple(valid.shape)}"
            )
        valid_float = valid.unsqueeze(-1).to(dtype=evidence.dtype)
        fixed_evidence = evidence * valid_float
        state = fixed_evidence
        evidence_scale = torch.sigmoid(self.evidence_gate)
        update_scale = torch.sigmoid(self.update_gate)
        key_padding_mask = _safe_key_padding_mask(~valid)
        outputs: list[torch.Tensor] = []
        resolved_steps = self.steps if steps is None else int(steps)
        if resolved_steps < 1:
            raise ValueError("recurrent refinement steps must be positive")
        for _ in range(resolved_steps):
            cell_input = state + evidence_scale * fixed_evidence
            candidate = self.cell(cell_input, src_key_padding_mask=key_padding_mask)
            state = (state + update_scale * (candidate - state)) * valid_float
            outputs.append(self.output_norm(state) * valid_float)
        return tuple(outputs)


class RelativeMultiheadCenterReadout(nn.Module):
    """Relative-aware multi-head center-to-neighbor attention readout."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        *,
        max_context_epochs: int = 21,
        dropout: float = 0.2,
        gate_bias_init: float = -2.0,
    ) -> None:
        super().__init__()
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")
        self.nhead = int(nhead)
        self.max_context_epochs = int(max_context_epochs)
        self.input_norm = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.relative_position_bias = nn.Parameter(
            torch.zeros(nhead, 2 * max_context_epochs - 1)
        )
        self.fuse = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
        )
        self.gate = nn.Linear(d_model * 2, d_model)
        nn.init.constant_(self.gate.bias, gate_bias_init)
        self._last_attention_weights: torch.Tensor | None = None

    def forward(
        self,
        z: torch.Tensor,
        epoch_valid_mask: torch.Tensor | None = None,
        *,
        return_attention: bool = False,
    ) -> torch.Tensor:
        if z.dim() != 3:
            raise ValueError(f"Expected [B,L,D], got {tuple(z.shape)}")
        batch, length, _ = z.shape
        if length > self.max_context_epochs:
            raise ValueError(
                f"context length {length} exceeds configured maximum {self.max_context_epochs}"
            )
        center_idx = length // 2
        center = z[:, center_idx, :]
        valid = (
            torch.ones(batch, length, dtype=torch.bool, device=z.device)
            if epoch_valid_mask is None
            else epoch_valid_mask.to(device=z.device, dtype=torch.bool)
        )
        if valid.shape != (batch, length):
            raise ValueError(
                f"epoch_valid_mask must have shape {(batch, length)}, got {tuple(valid.shape)}"
            )
        valid_neighbors = valid.clone()
        valid_neighbors[:, center_idx] = False
        has_neighbor = valid_neighbors.any(dim=1, keepdim=True)
        center_fallback = torch.arange(length, device=z.device).eq(center_idx)
        safe_valid = valid_neighbors | ~has_neighbor & center_fallback.unsqueeze(0)
        key_positions = torch.arange(length, device=z.device)
        lag_indices = key_positions - center_idx + self.max_context_epochs - 1
        bias = self.relative_position_bias[:, lag_indices].to(dtype=z.dtype)
        attn_mask = bias[None, :, None, :].expand(batch, -1, -1, -1).clone()
        attn_mask.masked_fill_(~safe_valid[:, None, None, :], float("-inf"))
        attn_mask = attn_mask.reshape(batch * self.nhead, 1, length)
        normalized = self.input_norm(z)
        context, weights = self.cross_attn(
            normalized[:, center_idx : center_idx + 1, :],
            normalized,
            normalized,
            attn_mask=attn_mask,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        context = context.squeeze(1)
        if return_attention:
            if weights is None:
                raise RuntimeError("Center readout did not return requested weights")
            self._last_attention_weights = torch.where(
                has_neighbor[:, None, :, None], weights, torch.zeros_like(weights)
            )
        else:
            self._last_attention_weights = None
        fused_input = torch.cat((center, context), dim=-1)
        enriched = center + torch.sigmoid(self.gate(fused_input)) * self.fuse(
            fused_input
        )
        return torch.where(has_neighbor, enriched, center)


class TransformerContextNet(nn.Module):
    """
    CNN epoch encoder -> Transformer encoder over the context.

    This model processes multi-epoch sequences by first encoding each epoch
    independently with a CNN, then using a Transformer encoder to model
    temporal context across epochs.

    Args:
        in_ch: Number of input channels.
        time_len: Samples per epoch.
        d_model: Transformer width (project CNN features to this).
        nhead: Number of attention heads.
        num_layers: Number of transformer encoder layers.
        dim_ff: Feedforward width.
        num_classes: Number of output classes (default 5: W, N1, N2, N3, REM).
        source_convention_names: Ordered training source names for an optional
            N2/N3 observation layer. Forward always returns shared logits.
        source_convention_penalty: Mean-square source-offset penalty used by
            supervised training; ignored when source names are absent.
        cnn_dropout: Dropout inside CNN blocks.
        head_dropout: Dropout in transformer encoder layers.
        classifier_dropout: Dropout in classifier and auxiliary heads (default 0.2).
            Passing None ties it to *head_dropout* for backward compatibility.
        classifier_head: Classification head family, ``"residual_mlp"`` (default,
            the historical 3-layer residual MLP) or ``"linear"`` (a
            literature-style ``LayerNorm -> Linear`` probe with no dropout).
            Selects the head only when *use_per_position_head* is False;
            combining ``"linear"`` with *use_per_position_head* is an error.
        norm: Normalization type ('bn' or 'gn').
        sdp_backend: SDPA backend selection ('auto', 'flash', 'mem_efficient', 'math').
        use_feature_extraction: Whether to extract and concatenate sleep-specific
            engineered features (spectral bands, spindle/slow-wave proxies, etc.)
            alongside CNN features.
        fusion_mode: How to combine CNN and engineered features
            ('concat', 'gated', 'learned_weight', 'sequential').
            Only used if use_feature_extraction=True.
        feature_normalize: Normalization strategy for engineered features
            ('none', 'standardize', 'minmax'). Default: 'standardize'.
        fs: Sampling frequency in Hz (default: 100). Used for engineered feature extraction.
        classifier_temperature: Temperature for logits scaling (default: 1.0).
            Values > 1.0 produce softer predictions, < 1.0 produce sharper predictions.
            Can help prevent collapse by reducing overconfidence.
            Only used if learnable_temperature=False.
        learnable_temperature: Whether to learn per-class temperature parameters (default: False).
            When enabled, learns a separate temperature for each class to improve calibration.
            The learned temperatures are parameterized in log-space and clamped to [0.1, 10.0].
            This can help with class imbalance by allowing different smoothing per class.
        use_n1_attention: Whether to apply N1-focused attention mechanism (default: False).
            When enabled, extracts N1-specific features (alpha dropout, mixed frequency,
            vertex waves, K-complex precursors, theta/alpha power) and uses them to
            modulate the transformer features through an attention mechanism.
        epoch_encoder_variant: Which CNN architecture to use for epoch encoding.
            Options: 'flexible_asymmetric' (default), 'dilated_asymmetric',
            'deep_asymmetric', 'preact_asymmetric', 'sleep_staging_hybrid',
            'learned_feature_axial', 'learned_feature_axial_v2'.
            - 'flexible_asymmetric': FlexibleAsymmetricEpochCNN with
              FlexibleModalityAwareStem (modality-specific dilations) and
              multi-dilated blocks throughout.
            - 'dilated_asymmetric': DilatedAsymmetricEpochCNN with multi-dilated blocks
              and asymmetric physiological stem.
        pooling_mode: Temporal pooling mode for the asymmetric epoch encoders
            ('statistics', 'attentive_fusion', 'learned', 'intra_epoch', or
            'both'). Default: 'both'. 'intra_epoch' (flexible_asymmetric only)
            restores within-epoch temporal order with a small pre-norm attention
            stack over ``intra_epoch_patches`` patches before collapsing to one
            vector per epoch; the context transformer still sees ``[B, L, d]``.
        epoch_pool_output_dim: Width of the epoch vector produced by the pooling
            head. ``None`` (default) keeps the historical behavior: the encoder
            emits the last CNN width and a learned ``proj`` maps it to
            ``d_model``. When set explicitly and equal to ``d_model``, ``proj``
            becomes :class:`torch.nn.Identity` so no rank waist remains between
            the epoch encoder and the context transformer. Only the 'learned' and
            'intra_epoch' pooling heads can retarget their output width.
        intra_epoch_patches: Number of non-overlapping within-epoch patches for
            ``pooling_mode='intra_epoch'``. Must divide the encoder's pre-pool
            temporal length (240 for a 30 s epoch at 128 Hz with the default
            trunk, so 16 patches of 15 positions tile it exactly).
        intra_epoch_layers: Depth of the intra-epoch pre-norm attention stack.
        intra_epoch_dim: Width of the intra-epoch tokens (``None`` -> last CNN
            width). Kept separate from ``epoch_pool_output_dim`` so widening the
            epoch vector does not widen the intra-epoch attention.
        intra_epoch_heads: Attention heads per intra-epoch block.
        intra_epoch_ff_mult: Feed-forward expansion of the intra-epoch blocks.
        intra_epoch_dropout: Dropout inside the intra-epoch stack
            (``None`` -> ``cnn_dropout``).
        intra_epoch_aggregation: Aggregation over the within-epoch tokens. Only
            ``'attention'`` (learned softmax pooling) is supported.
        intra_epoch_pos_init_std: Initialization std of the learned within-epoch
            positions, as a fraction of the (normalized) token scale under
            ``intra_epoch_pool_version=2``. Too small and the head degenerates
            into permutation-invariant patch mean-pooling.
        intra_epoch_pool_version: Intra-epoch pool topology. ``2`` (default)
            normalizes the tokens before adding positions and learns an
            aggregation temperature; ``1`` reproduces the original topology for
            checkpoints written before those were added. See
            :class:`~spectra.models.patch_tokenization.IntraEpochAttentionPool`.
        cnn_feature_guidance: If True, pass engineered features into the epoch CNN and
            apply FiLM-style modulation after the first CNN stage (supported epoch
            encoders only).
        fusion_gate_bias: Initial bias for fusion gates (sigmoid), negative keeps
            gates mostly closed early in training.
        sleepfm_mode: When use_sleepfm_fusion=True, controls fusion behavior.
            'pool' (default): Full SleepFMInspiredFusion with temporal attention pooling.
                Returns [B, d_model] pooled center representation.
            'fuse_only': EpochFeatureFusion without temporal pooling.
                Returns [B, L, d_model] fused sequence, center extracted downstream.

    Input:
        seq: [B, L, C, T] or mapping with 'wave' key

    Output:
        logits [B, num_classes] (center epoch by default)
        If predict_all=True: [B, L, num_classes]

    Torch.compile compatibility (reduce-overhead mode):
        This model is fully compatible with torch.compile(model, mode="reduce-overhead").

        Key optimizations for torch.compile:
        1. N1 attention: Vectorized implementation (no Python loops over sequence length)
        2. Engineered features: STFT operations excluded via @torch_compile_disable
        3. Waveform preparation: Compile-safe shape and mask canonicalization
        4. No unnecessary clone() operations that break CUDA graphs

        Expected performance with torch.compile:
        - Without engineered features: ~95% speedup (full CUDA graph compilation)
        - With engineered features: ~80-90% speedup (partial compilation, STFT in eager mode)
        - With N1 attention: ~85-90% speedup (N1 feature extraction in eager mode)

        Usage:
            model = TransformerContextNet(...)
            compiled_model = torch.compile(model, mode="reduce-overhead")

    Note:
        STFT operations in SleepFeatureExtractor are excluded via @torch_compile_disable
        decorators on individual methods, allowing partial compilation with STFT running
        in eager mode.
    """

    def _encoder_recording_kwargs(
        self, recording_index: torch.Tensor | None, batch: int, context_len: int
    ) -> dict[str, torch.Tensor]:
        """Build the ``recording_index`` kwarg for encoders that accept one.

        Returns an empty dict for encoders without the parameter, so passing
        recording provenance never breaks a variant that does not use it. The
        index is expanded ``L``-fold because the encoder sees ``[B*L, C, T]``.
        """
        if recording_index is None:
            return {}
        encoder = getattr(self, "epoch_encoder", None)
        if encoder is None:
            return {}
        encoder = getattr(encoder, "cnn_extractor", encoder)
        forward = getattr(type(encoder), "forward", None)
        try:
            accepts = (
                forward is not None
                and "recording_index" in inspect.signature(forward).parameters
            )
        except (TypeError, ValueError):
            accepts = False
        if not accepts:
            return {}
        flat = recording_index.reshape(-1)
        if flat.numel() != batch:
            if getattr(encoder, "recording_conditioner", None) is not None:
                raise ValueError(f"recording_index must contain batch={batch} entries")
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
        use_feature_extraction: bool = False,
        simple_feature_projection: bool = False,
        fusion_mode: str = "concat",
        use_epoch_attention: bool = False,
        epoch_attention_position: str = "match",
        feature_normalize: str = "standardize",
        fs: int = 128,
        classifier_temperature: float = 1.0,
        learnable_temperature: bool = False,
        use_confidence_head: bool = False,
        confidence_head_weight: float = 0.1,
        confidence_head_detach_target: bool = True,
        confidence_head_hidden_dim: int | None = None,
        use_n1_attention: bool = False,
        channel_names: list[str] | str | None = None,
        epoch_encoder_variant: str = "multirate_asymmetric",
        use_sinc_stem: bool = True,
        pooling_mode: str = "both",
        epoch_pool_output_dim: int | None = None,
        intra_epoch_patches: int = 16,
        intra_epoch_layers: int = 2,
        intra_epoch_dim: int | None = None,
        intra_epoch_heads: int = 8,
        intra_epoch_ff_mult: float = 2.0,
        intra_epoch_dropout: float | None = None,
        intra_epoch_aggregation: str = "attention",
        intra_epoch_pos_init_std: float = 0.15,
        intra_epoch_pool_version: int = 2,
        epoch_tokenization: str = "epoch",
        patches_per_epoch: int = 10,
        patch_token_dim: int | None = None,
        grid_attention_mode: str = "absolute",
        transformer_checkpoint_threshold: int = 2,
        use_delta_continuity: bool = False,
        use_spindle_gating: bool = False,
        spindle_gate_strength: float = 0.5,
        deep_blocks_per_stage: tuple[int, ...] | None = None,
        deep_drop_path_rate: float = 0.1,
        preact_blocks_per_stage: tuple[int, ...] | None = None,
        preact_drop_path_rate: float = 0.1,
        modern_tcn_encoder_kwargs: dict[str, Any] | None = None,
        channel_indep_encoder_kwargs: dict[str, Any] | None = None,
        multirate_asymmetric_encoder_kwargs: dict[str, Any] | None = None,
        conformer_encoder_kwargs: dict[str, Any] | None = None,
        channel_config: CanonicalChannelSet | None = None,
        channel_config_path: str | None = None,
        eeg_ratio: float = 0.6,
        eog_ratio: float = 0.25,
        emg_ratio: float = 0.15,
        stem_eeg_out_ch: int | None = None,
        stem_eog_out_ch: int | None = None,
        stem_emg_out_ch: int | None = None,
        combination_aware_fusion: bool = False,
        cnn_feature_guidance: bool = False,
        classifier_head: str = "residual_mlp",
        use_per_position_head: bool = False,
        head_use_local_mix: bool = False,
        head_local_kernel: int = 3,
        head_norm: str = "layernorm",
        fusion_gate_bias: float = -2.0,
        cnn_widths: tuple[int, ...] | None = None,
        cnn_stage_strides: tuple[int, int, int, int] | None = None,
        cnn_stage1_dilations: tuple[int, ...] | None = None,
        cnn_stage2_dilations: tuple[int, ...] | None = None,
        cnn_stem_aa_num_taps: int | None = None,
        cnn_stage_aa_num_taps: tuple[int, ...] | None = None,
        cnn_anti_alias_dilated_branches: bool | None = None,
        cnn_legacy_global_branch: bool = False,
        cnn_aa_legacy_cutoff: bool = False,
        cnn_legacy_stem_allocation: bool = False,
        cnn_legacy_unmasked_stem: bool = False,
        cnn_legacy_batchnorm_checkpointing: bool = False,
        cnn_aa_cutoff_ratio: float | None = None,
        cnn_aa_beta: float | None = None,
        use_variance_preserving_norm: bool = True,
        feature_slots_total: int = 16,
        feature_slots_eeg: int = 8,
        feature_slots_eog: int = 4,
        feature_slots_emg: int = 4,
        slot_local_dim: int = 32,
        cnn_blocks_per_stage: int = 1,
        slot_width_mult: float = 1.0,
        feature_axial_d_model: int = 128,
        feature_axial_nhead: int = 4,
        feature_axial_num_layers: int = 4,
        feature_axial_dim_feedforward: int = 512,
        time_bins_per_epoch: int = 30,
        context_epochs: int = 21,
        context_attention_mode: str = "legacy_absolute",
        context_readout_mode: str = "legacy_single",
        recurrent_refinement_steps: int = 0,
        axial_readout_dim: int = 128,
        axial_output_dim: int = 512,
        feature_filter_init: str = "bandpass",
        filter_diversity_weight: float = 0.0001,
        filter_diversity_mode: str = "kernel_cosine",
        filter_diversity_warmup_epochs: int = 0,
        attention_backend: str | None = None,
        gradient_checkpoint_axial: bool = True,
        feature_axial_ablation: str = "full",
        feature_axial_temporal_attention_mode: str = "full",
        feature_axial_local_window: int = 90,
        axial_v2_halo_seconds: int = 5,
        axial_v2_slot_local_dim: int = 48,
        axial_v2_cnn_blocks_per_stage: int = 2,
        axial_v2_slot_width_mult: float = 1.5,
        axial_v2_d_model: int = 160,
        axial_v2_nhead: int = 5,
        axial_v2_num_layers: int = 4,
        axial_v2_dim_feedforward: int = 640,
        axial_v2_summary_layers: int = 2,
        axial_v2_local_window: int = 90,
        axial_v2_output_dim: int = 512,
        center_mask_prob: float = 0.0,
        use_neighbor_prediction: bool = False,
        neighbor_prediction_n: int = 2,
        use_reconstruction_loss: bool = False,
        cnn_reconstruction_weight: float = 0.5,
        eng_reconstruction_weight: float = 0.3,
        slow_wave_occupancy_config: Mapping[str, Any] | None = None,
        slow_wave_occupancy_loss_weight: float = 0.0,
        use_sleepfm_fusion: bool = False,
        sleepfm_temporal_pool_type: str = "attention",
        sleepfm_center_bias: str | None = "gaussian",
        sleepfm_center_bias_strength: float = 2.0,
        sleepfm_fusion_num_heads: int = 4,
        sleepfm_temperature: float = 1.0,
        sleepfm_debug_attention: bool = False,
        sleepfm_mode: str = "pool",
        use_engineered_features: bool | None = None,
        feature_fusion_mode: str | None = None,
        recording_conditioning: bool = False,
        recording_conditioning_samples: int = 64,
        recording_conditioning_dim: int = 64,
        source_convention_names: Sequence[str] | None = None,
        source_convention_penalty: float = 0.01,
    ) -> None:
        super().__init__()
        if epoch_encoder_variant != "multirate_asymmetric":
            raise ValueError("SPECTRA supports only multirate_asymmetric checkpoints")
        from .objectives.source_convention import SourceConvention

        if source_convention_names is not None and num_classes != 5:
            raise ValueError("source_convention requires the canonical five classes")
        self.source_convention = (
            SourceConvention(source_convention_names, source_convention_penalty)
            if source_convention_names is not None
            else None
        )
        self.recording_conditioning = bool(recording_conditioning)
        self.recording_conditioning_samples = int(recording_conditioning_samples)
        self.recording_conditioning_dim = int(recording_conditioning_dim)
        if recording_conditioning and epoch_encoder_variant not in (
            "flexible_asymmetric",
            "multirate_asymmetric",
        ):
            raise ValueError(
                "recording_conditioning supports only flexible_asymmetric and multirate_asymmetric"
            )
        if not 1 <= recording_conditioning_samples <= 64:
            raise ValueError("recording_conditioning_samples must be in [1, 64]")
        if recording_conditioning_dim < 1:
            raise ValueError("recording_conditioning_dim must be positive")
        recording_kwargs = {
            "recording_conditioning": bool(recording_conditioning),
            "recording_conditioning_samples": int(recording_conditioning_samples),
            "recording_conditioning_dim": int(recording_conditioning_dim),
        }
        if use_engineered_features is not None:
            import warnings

            warnings.warn(
                "Parameter 'use_engineered_features' is deprecated and will be removed in a future version. Use 'use_feature_extraction' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if use_feature_extraction is False:
                use_feature_extraction = use_engineered_features
        if feature_fusion_mode is not None:
            import warnings

            warnings.warn(
                "Parameter 'feature_fusion_mode' is deprecated and will be removed in a future version. Use 'fusion_mode' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if fusion_mode == "concat":
                fusion_mode = feature_fusion_mode
        if isinstance(channel_names, str):
            channel_names = [n.strip() for n in channel_names.split(",")]
        self.cross_attn_fusion_layers: nn.ModuleList | None = None
        if sleepfm_mode not in {"pool", "fuse_only"}:
            raise ValueError(
                f"sleepfm_mode must be one of {{'pool', 'fuse_only'}}, got {sleepfm_mode!r}"
            )
        if use_sleepfm_fusion:
            if not use_feature_extraction:
                raise ValueError(
                    "use_sleepfm_fusion=True requires use_feature_extraction=True. SleepFM fusion combines transformer output with engineered features."
                )
            if fusion_mode == "sequential":
                raise ValueError(
                    "use_sleepfm_fusion and fusion_mode='sequential' are mutually exclusive. SleepFM already performs post-transformer feature fusion."
                )
        self.in_ch = int(in_ch)
        self.expected_input_channels = self.in_ch
        self.num_classes = num_classes
        self.use_feature_extraction = use_feature_extraction
        self.fusion_mode = fusion_mode
        self.use_epoch_attention = use_epoch_attention
        if epoch_attention_position == "match":
            self.epoch_attention_position = "none"
        else:
            self.epoch_attention_position = epoch_attention_position
        self.cnn_feature_guidance = cnn_feature_guidance
        self.classifier_temperature = classifier_temperature
        self.learnable_temperature = learnable_temperature
        self.use_confidence_head = bool(use_confidence_head)
        self.confidence_head_weight = float(confidence_head_weight)
        self.confidence_head_detach_target = bool(confidence_head_detach_target)
        self.use_n1_attention = use_n1_attention
        self.epoch_encoder_variant = epoch_encoder_variant
        self.is_learned_feature_axial_v1 = (
            epoch_encoder_variant == "learned_feature_axial"
        )
        self.is_learned_feature_axial_v2 = (
            epoch_encoder_variant == "learned_feature_axial_v2"
        )
        self.is_learned_feature_axial = (
            self.is_learned_feature_axial_v1 or self.is_learned_feature_axial_v2
        )
        self.use_sinc_stem = use_sinc_stem
        self.pooling_mode = pooling_mode
        self.epoch_pool_output_dim = (
            None if epoch_pool_output_dim is None else int(epoch_pool_output_dim)
        )
        self.intra_epoch_patches = int(intra_epoch_patches)
        self.intra_epoch_layers = int(intra_epoch_layers)
        self.intra_epoch_dim = None if intra_epoch_dim is None else int(intra_epoch_dim)
        self.intra_epoch_heads = int(intra_epoch_heads)
        self.intra_epoch_ff_mult = float(intra_epoch_ff_mult)
        self.intra_epoch_dropout = (
            None if intra_epoch_dropout is None else float(intra_epoch_dropout)
        )
        self.intra_epoch_aggregation = intra_epoch_aggregation
        self.intra_epoch_pos_init_std = float(intra_epoch_pos_init_std)
        self.intra_epoch_pool_version = int(intra_epoch_pool_version)
        if pooling_mode == "intra_epoch":
            raise ValueError(
                f"pooling_mode='intra_epoch' is only implemented for epoch_encoder_variant='flexible_asymmetric', got {epoch_encoder_variant!r}"
            )
        if epoch_pool_output_dim is not None:
            raise ValueError(
                f"epoch_pool_output_dim is only implemented for epoch_encoder_variant='flexible_asymmetric', got {epoch_encoder_variant!r}"
            )
        engaged_legacy = sorted(
            (
                name
                for name, value, default in (
                    ("cnn_stem_aa_num_taps", cnn_stem_aa_num_taps, 47),
                    (
                        "cnn_stage_aa_num_taps",
                        (
                            tuple(cnn_stage_aa_num_taps)
                            if cnn_stage_aa_num_taps is not None
                            else None
                        ),
                        (31, 23, 23, 23),
                    ),
                    (
                        "cnn_anti_alias_dilated_branches",
                        cnn_anti_alias_dilated_branches,
                        False,
                    ),
                    ("cnn_aa_cutoff_ratio", cnn_aa_cutoff_ratio, 0.85),
                    ("cnn_aa_beta", cnn_aa_beta, 6.0),
                    ("cnn_legacy_global_branch", cnn_legacy_global_branch, False),
                    ("cnn_aa_legacy_cutoff", cnn_aa_legacy_cutoff, False),
                    ("cnn_legacy_stem_allocation", cnn_legacy_stem_allocation, False),
                    ("cnn_legacy_unmasked_stem", cnn_legacy_unmasked_stem, False),
                    (
                        "cnn_legacy_batchnorm_checkpointing",
                        cnn_legacy_batchnorm_checkpointing,
                        False,
                    ),
                )
                if value is not None and value != default
            )
        )
        if engaged_legacy:
            raise ValueError(
                f"the legacy CNN reconstruction knobs are only wired into epoch_encoder_variant='flexible_asymmetric', got {epoch_encoder_variant!r}; remove: {', '.join(engaged_legacy)}"
            )
        self.time_len = time_len
        self.fs = fs
        self.temporal_engineered_features: nn.Module | None = None
        self.nhead = nhead
        self.feature_slots_total = int(feature_slots_total)
        self.feature_slots_eeg = int(feature_slots_eeg)
        self.feature_slots_eog = int(feature_slots_eog)
        self.feature_slots_emg = int(feature_slots_emg)
        self.slot_local_dim = int(slot_local_dim)
        self.cnn_blocks_per_stage = int(cnn_blocks_per_stage)
        self.slot_width_mult = float(slot_width_mult)
        self.feature_axial_d_model = int(feature_axial_d_model)
        self.feature_axial_nhead = int(feature_axial_nhead)
        self.feature_axial_num_layers = int(feature_axial_num_layers)
        self.feature_axial_dim_feedforward = int(feature_axial_dim_feedforward)
        self.time_bins_per_epoch = int(time_bins_per_epoch)
        self.context_epochs = int(context_epochs)
        clf_dropout = head_dropout if classifier_dropout is None else classifier_dropout
        if context_attention_mode not in {"legacy_absolute", "relative_full"}:
            raise ValueError(
                "context_attention_mode must be 'legacy_absolute' or 'relative_full'"
            )
        if context_readout_mode not in {
            "legacy_single",
            "relative_multihead",
            "center_token",
        }:
            raise ValueError(
                "context_readout_mode must be 'legacy_single', 'relative_multihead', or 'center_token'"
            )
        if self.context_epochs < 1 or self.context_epochs % 2 == 0:
            raise ValueError("context_epochs must be a positive odd integer")
        self.context_attention_mode = context_attention_mode
        self.context_readout_mode = context_readout_mode
        self.recurrent_refinement_steps = int(recurrent_refinement_steps)
        if self.recurrent_refinement_steps < 0:
            raise ValueError("recurrent_refinement_steps must be non-negative")
        if self.recurrent_refinement_steps and self.is_learned_feature_axial:
            raise ValueError(
                "recurrent refinement currently requires an epoch-token CNN encoder"
            )
        if self.recurrent_refinement_steps and use_sleepfm_fusion:
            raise ValueError(
                "recurrent refinement is not compatible with SleepFM fusion"
            )
        self.head_dropout = float(head_dropout)
        self.classifier_dropout = float(clf_dropout)
        if ffn_activation not in {"gelu", "swiglu"}:
            raise ValueError("ffn_activation must be 'gelu' or 'swiglu'")
        self.ffn_activation = ffn_activation
        self.axial_readout_dim = int(axial_readout_dim)
        self.axial_output_dim = int(axial_output_dim)
        self.feature_filter_init = feature_filter_init
        self.filter_diversity_weight = float(filter_diversity_weight)
        self.filter_diversity_mode = filter_diversity_mode
        self.filter_diversity_warmup_epochs = int(filter_diversity_warmup_epochs)
        self.attention_backend = attention_backend or sdp_backend
        self.gradient_checkpoint_axial = bool(gradient_checkpoint_axial)
        self.feature_axial_ablation = feature_axial_ablation
        self.feature_axial_temporal_attention_mode = (
            feature_axial_temporal_attention_mode
        )
        self.feature_axial_local_window = int(feature_axial_local_window)
        self.axial_v2_halo_seconds = int(axial_v2_halo_seconds)
        self.axial_v2_slot_local_dim = int(axial_v2_slot_local_dim)
        self.axial_v2_cnn_blocks_per_stage = int(axial_v2_cnn_blocks_per_stage)
        self.axial_v2_slot_width_mult = float(axial_v2_slot_width_mult)
        self.axial_v2_d_model = int(axial_v2_d_model)
        self.axial_v2_nhead = int(axial_v2_nhead)
        self.axial_v2_num_layers = int(axial_v2_num_layers)
        self.axial_v2_dim_feedforward = int(axial_v2_dim_feedforward)
        self.axial_v2_summary_layers = int(axial_v2_summary_layers)
        self.axial_v2_local_window = int(axial_v2_local_window)
        self.axial_v2_output_dim = int(axial_v2_output_dim)
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
        needs_standalone_engineered = use_feature_extraction and (
            fusion_mode == "sequential" or use_sleepfm_fusion
        )
        engineered_feature_extractor: nn.Module | None = None
        engineered_feature_dim: int | None = None
        if use_feature_extraction and (
            needs_standalone_engineered or cnn_feature_guidance
        ):
            resolved_fs, resolved_epoch_sec = resolve_sampling_params(time_len, fs)
            engineered_feature_extractor = SleepFeatureExtractor(
                fs=resolved_fs,
                epoch_sec=resolved_epoch_sec,
                learnable_bands=True,
                num_channels=in_ch,
                normalize=feature_normalize,
                channel_names=channel_names,
            )
            engineered_feature_dim = int(
                getattr(engineered_feature_extractor, "out_dim", 0)
            )
        encoder_kwargs: dict = {
            "in_ch": in_ch,
            "time_len": time_len,
            "dropout": cnn_dropout,
            "norm": norm,
        }
        if cnn_widths is not None:
            encoder_kwargs["widths"] = cnn_widths
        passthrough_name = EPOCH_ENCODER_KWARG_PASSTHROUGH[epoch_encoder_variant]
        passthrough_by_name: dict[str, dict[str, Any] | None] = {
            "modern_tcn_encoder_kwargs": modern_tcn_encoder_kwargs,
            "channel_indep_encoder_kwargs": channel_indep_encoder_kwargs,
            "multirate_asymmetric_encoder_kwargs": multirate_asymmetric_encoder_kwargs,
        }
        passthrough = dict(passthrough_by_name[passthrough_name] or {})
        for key, value in recording_kwargs.items():
            if key in passthrough and passthrough[key] != value:
                raise ValueError(f"{key} must be configured at the context-model level")
            passthrough[key] = value
        reserved = EPOCH_ENCODER_PASSTHROUGH_RESERVED.intersection(passthrough)
        if reserved:
            raise ValueError(
                f"{passthrough_name} must not carry {sorted(reserved)}; those are supplied by the context model itself (see EPOCH_ENCODER_PASSTHROUGH_RESERVED)"
            )
        if pooling_mode not in (None, "occupancy"):
            logging.getLogger(__name__).info(
                "epoch_encoder_variant=%r owns its pooling head (set it via the %s passthrough, e.g. pooling_mode=...); ignoring the context-model pooling_mode=%r.",
                epoch_encoder_variant,
                passthrough_name,
                pooling_mode,
            )
        encoder_cls = EPOCH_ENCODER_VARIANTS[epoch_encoder_variant]
        base_encoder = encoder_cls(**encoder_kwargs, fs=fs, **passthrough)
        self._slow_wave_occupancy_encoder_dim = int(base_encoder.out_dim)
        if use_feature_extraction and (
            fusion_mode == "sequential" or use_sleepfm_fusion
        ):
            self.epoch_encoder = base_encoder
            enc_dim = cast(int, base_encoder.out_dim)
            if engineered_feature_extractor is None:
                resolved_fs, resolved_epoch_sec = resolve_sampling_params(time_len, fs)
                engineered_feature_extractor = SleepFeatureExtractor(
                    fs=resolved_fs,
                    epoch_sec=resolved_epoch_sec,
                    learnable_bands=True,
                    num_channels=in_ch,
                    normalize=feature_normalize,
                    channel_names=channel_names,
                )
                engineered_feature_dim = int(
                    getattr(engineered_feature_extractor, "out_dim", 0)
                )
            self.feature_extractor = engineered_feature_extractor
            feature_extractor = self.feature_extractor
            if feature_extractor is None:
                raise RuntimeError("Feature extractor expected but is not initialized.")
            feat_dim = int(engineered_feature_dim or feature_extractor.out_dim)
            if fusion_mode == "sequential":
                if simple_feature_projection:
                    self.feat_projection = nn.Linear(feat_dim, enc_dim)
                else:
                    self.feat_projection = BottleneckProjection(
                        in_dim=feat_dim,
                        out_dim=enc_dim,
                        expansion=2,
                        dropout=cnn_dropout,
                    )
                fusion_heads = max(1, nhead // 2)
                while fusion_heads > 1 and enc_dim % fusion_heads != 0:
                    fusion_heads -= 1
                if enc_dim % fusion_heads != 0:
                    fusion_heads = 1
                if use_epoch_attention:
                    from .enhanced_sequential_fusion import (
                        EnhancedSequentialFeatureFusion,
                    )

                    self.sequential_fusion = EnhancedSequentialFeatureFusion(
                        cnn_dim=enc_dim,
                        feat_dim=feat_dim,
                        num_heads=fusion_heads,
                        dropout=cnn_dropout,
                        use_epoch_context=True,
                        num_context_heads=fusion_heads,
                        position_encoding=self.epoch_attention_position,
                    )
                else:
                    self.sequential_fusion = SequentialFeatureFusion(
                        cnn_dim=enc_dim,
                        feat_dim=enc_dim,
                        num_heads=fusion_heads,
                        dropout=cnn_dropout,
                    )
            else:
                self.feat_projection = None
                self.sequential_fusion = None
            self.eng_token_projection = None
            self.cnn_type_embed = None
            self.eng_type_embed = None
            self.modality_gate = None
            self.modality_norm = None
            self.cross_attn_fusion_layers = None
        elif use_feature_extraction:
            resolved_fs, resolved_epoch_sec = resolve_sampling_params(time_len, fs)
            self.epoch_encoder = HybridFeatureExtractor(
                cnn_extractor=base_encoder,
                use_engineered=True,
                fs=resolved_fs,
                epoch_sec=resolved_epoch_sec,
                num_channels=in_ch,
                fusion_mode=fusion_mode,
                normalize=feature_normalize,
                channel_names=channel_names,
            )
            enc_dim = cast(int, self.epoch_encoder.out_dim)
            self.feature_extractor = None
            self.feat_projection = None
            self.sequential_fusion = None
            self.eng_token_projection = None
            self.cnn_type_embed = None
            self.eng_type_embed = None
            self.modality_gate = None
            self.modality_norm = None
            self.cross_attn_fusion_layers = None
        else:
            self.epoch_encoder = base_encoder
            enc_dim = cast(int, base_encoder.out_dim)
            self.feature_extractor = None
            self.feat_projection = None
            self.sequential_fusion = None
            self.eng_token_projection = None
            self.cnn_type_embed = None
            self.eng_type_embed = None
            self.modality_gate = None
            self.modality_norm = None
            self.cross_attn_fusion_layers = None
        self._last_attention_weights: torch.Tensor | None = None
        self._last_grid_attention_weights: torch.Tensor | None = None
        if transformer_checkpoint_threshold < 0:
            raise ValueError(
                f"transformer_checkpoint_threshold must be >= 0, got {transformer_checkpoint_threshold}"
            )
        self.transformer_checkpoint_threshold = int(transformer_checkpoint_threshold)
        if sdp_backend not in _SDP_BACKEND_CHOICES:
            raise ValueError(f"Unsupported sdp_backend '{sdp_backend}'")
        self.sdp_backend = sdp_backend
        self.epoch_encoder_out_dim = int(enc_dim)
        self.context_d_model = int(d_model)
        self.proj_is_identity = True
        self.axial_backbone = None
        self.center_feature_time_readout = None
        self.axial_output_projection = None
        self.axial_v2_summary = None
        self.axial_v2_readout = None
        self.epoch_encoder_out_dim = int(enc_dim)
        self.context_d_model = int(d_model)
        self.proj_is_identity = (
            self.epoch_pool_output_dim is not None and enc_dim == d_model
        )
        if self.proj_is_identity:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Linear(int(enc_dim), d_model)
            transformer_init_(self.proj)
            if enc_dim == d_model:
                nn.init.eye_(self.proj.weight)
                nn.init.zeros_(self.proj.bias)
        if self.context_attention_mode == "legacy_absolute":
            max_pe_len = 128
            self.pos_encoding = nn.Parameter(torch.randn(1, max_pe_len, d_model) * 0.02)
        else:
            self.register_parameter("pos_encoding", None)
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1 for the transformer encoder")
        encoder_layer: nn.Module
        if (
            epoch_tokenization == "grid"
            and grid_attention_mode == "relative_factorized"
        ):
            encoder_layer = GridRelativePositionEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_ff,
                dropout=head_dropout,
                batch_first=True,
                norm_first=True,
                activation=nn.GELU(approximate="tanh"),
                ffn_activation=ffn_activation,
                num_patches=int(patches_per_epoch),
                max_epochs=self.context_epochs,
            )
        elif self.context_attention_mode == "relative_full":
            encoder_layer = RelativePositionTransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_ff,
                dropout=head_dropout,
                batch_first=True,
                norm_first=True,
                activation=nn.GELU(approximate="tanh"),
                ffn_activation=ffn_activation,
                max_context_epochs=self.context_epochs,
            )
        elif ffn_activation == "swiglu":
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
        if self.context_readout_mode == "relative_multihead":
            self.center_context_readout = RelativeMultiheadCenterReadout(
                d_model=d_model,
                nhead=nhead,
                max_context_epochs=self.context_epochs,
                dropout=clf_dropout,
                gate_bias_init=self.fusion_gate_bias,
            )
        elif self.context_readout_mode == "legacy_single":
            self.center_context_readout = CenterContextReadout(
                d_model=d_model,
                dropout=clf_dropout,
                gate_bias_init=self.fusion_gate_bias,
            )
        else:
            self.center_context_readout = nn.Identity()
        self.center_context_readout.apply(transformer_init_)
        self.feature_dim = d_model
        self._gradient_checkpointing = False
        self.recurrent_refiner: RecurrentContextRefiner | None
        if self.recurrent_refinement_steps:
            self.recurrent_refiner = RecurrentContextRefiner(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_ff,
                dropout=head_dropout,
                steps=self.recurrent_refinement_steps,
                max_context_epochs=self.context_epochs,
                attention_mode=self.context_attention_mode,
                ffn_activation=self.ffn_activation,
            )
        else:
            self.recurrent_refiner = None
        self.epoch_tokenization = epoch_tokenization
        self.patches_per_epoch = int(patches_per_epoch)
        self.grid_attention_mode = grid_attention_mode
        self.epoch_grid: EpochPatchGrid | None = None
        if epoch_tokenization not in ("epoch", "grid"):
            raise ValueError(
                f"epoch_tokenization must be 'epoch' or 'grid', got {epoch_tokenization!r}"
            )
        if grid_attention_mode not in ("absolute", "relative_factorized"):
            raise ValueError(
                f"grid_attention_mode must be 'absolute' or 'relative_factorized', got {grid_attention_mode!r}"
            )
        if epoch_tokenization == "grid":
            self._build_epoch_grid(
                d_model=d_model, dropout=head_dropout, patch_token_dim=patch_token_dim
            )
            self.transformer_checkpoint_threshold = 0
        validate_classifier_head_selection(classifier_head, use_per_position_head)
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
        elif classifier_head == "linear":
            self.classifier = LinearSleepHead(d_model=d_model, num_classes=num_classes)
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
        if learnable_temperature:
            if classifier_temperature <= 0:
                raise ValueError(
                    f"classifier_temperature must be positive, got {classifier_temperature}"
                )
            init_val = (
                math.log(classifier_temperature)
                if classifier_temperature != 1.0
                else 0.0
            )
            self.log_temperature = nn.Parameter(
                torch.full((num_classes,), init_val, dtype=torch.float32)
            )
        else:
            self.log_temperature = None
        self._last_features: torch.Tensor | None = None
        self._sleepfm_pooled_center: torch.Tensor | None = None
        self._last_confidence_logits: torch.Tensor | None = None
        self._last_confidence: torch.Tensor | None = None
        self._last_confidence_pred_classes: torch.Tensor | None = None
        self.use_reconstruction_loss = use_reconstruction_loss
        self.cnn_reconstruction_weight = cnn_reconstruction_weight
        self.eng_reconstruction_weight = eng_reconstruction_weight
        self._last_cnn_targets: torch.Tensor | None = None
        self._last_eng_targets: torch.Tensor | None = None
        if use_reconstruction_loss:
            cnn_dim = self.epoch_encoder_out_dim
            self.cnn_reconstruction_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.SiLU(),
                nn.Dropout(clf_dropout),
                nn.Linear(d_model, cnn_dim),
            )
            if self.use_feature_extraction and self.feature_extractor is not None:
                eng_out_dim = int(getattr(self.feature_extractor, "out_dim", 0))
                if eng_out_dim > 0:
                    self.eng_reconstruction_head = nn.Sequential(
                        nn.Linear(d_model, d_model // 2),
                        nn.SiLU(),
                        nn.Dropout(clf_dropout),
                        nn.Linear(d_model // 2, eng_out_dim),
                    )
                else:
                    self.eng_reconstruction_head = None
            else:
                self.eng_reconstruction_head = None
        else:
            self.cnn_reconstruction_head = None
            self.eng_reconstruction_head = None
        self.input_preparer = ContextWaveformPreparer(in_ch, time_len)
        self.channel_embedding = LegacyChannelEmbeddingState(in_ch)
        if use_n1_attention:
            resolved_fs, resolved_epoch_sec = resolve_sampling_params(time_len, fs)
            n1_heads = max(1, nhead // 2)
            while n1_heads > 1 and d_model % n1_heads != 0:
                n1_heads -= 1
            if d_model % n1_heads != 0:
                n1_heads = 1
            self.n1_feature_extractor = SleepFeatureExtractor(
                fs=resolved_fs,
                epoch_sec=resolved_epoch_sec,
                learnable_bands=True,
                num_channels=in_ch,
                normalize="standardize",
            )
            self.n1_attention = N1FocusedAttention(
                d_model=d_model,
                n1_feature_dim=self.n1_feature_extractor._n1_feature_dim,
            )
        else:
            self.n1_feature_extractor = None
            self.n1_attention = None
        self.center_mask_prob = center_mask_prob
        if center_mask_prob > 0 and (not self.is_learned_feature_axial):
            self.mask_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        else:
            self.register_buffer("mask_token", None)
        self.use_neighbor_prediction = use_neighbor_prediction
        self.neighbor_prediction_n = neighbor_prediction_n
        if use_neighbor_prediction:
            self.neighbor_head = TemporalContextHead(
                d_model=d_model,
                num_classes=num_classes,
                n_neighbors=neighbor_prediction_n,
            )
        else:
            self.neighbor_head = None
        self.use_sleepfm_fusion = use_sleepfm_fusion
        self.sleepfm_mode = sleepfm_mode
        self._sleepfm_debug_attention = sleepfm_debug_attention
        self._sleepfm_attention_weights: dict | None = None
        self._sleepfm_engineered_core_dim: int = 0
        self._sleepfm_yasa_dim: int = 0
        self._sleepfm_source_entropy: torch.Tensor | None = None
        self._sleepfm_source_mean_weights: torch.Tensor | None = None
        self.sleepfm_source_min_entropy_ratio: float = 0.35
        self.epoch_feature_fusion: nn.Module | None = None
        if use_sleepfm_fusion:
            from .sleepfm_inspired_modules import (
                EpochFeatureFusion,
                SleepFMInspiredFusion,
            )

            if self.feature_extractor is not None:
                eng_dim = int(getattr(self.feature_extractor, "out_dim", 0))
                yasa_dim = max(
                    0, int(getattr(self.feature_extractor, "yasa_feature_dim", 0))
                )
                core_dim = int(
                    getattr(
                        self.feature_extractor, "core_feature_dim", eng_dim - yasa_dim
                    )
                )
                core_dim = max(0, min(core_dim, eng_dim))
                self._sleepfm_engineered_core_dim = core_dim
                self._sleepfm_yasa_dim = max(0, eng_dim - core_dim)
            else:
                raise ValueError(
                    "use_sleepfm_fusion=True but no feature_extractor available. Ensure use_feature_extraction=True is set."
                )
            if sleepfm_mode == "fuse_only":
                self.epoch_feature_fusion = EpochFeatureFusion(
                    cnn_dim=d_model,
                    engineered_dim=self._sleepfm_engineered_core_dim,
                    output_dim=d_model,
                    yasa_dim=self._sleepfm_yasa_dim,
                    num_heads=sleepfm_fusion_num_heads,
                    dropout=clf_dropout,
                )
                self.sleepfm_fusion = None
            else:
                self.sleepfm_fusion = SleepFMInspiredFusion(
                    cnn_dim=d_model,
                    engineered_dim=self._sleepfm_engineered_core_dim,
                    output_dim=d_model,
                    yasa_dim=self._sleepfm_yasa_dim,
                    num_heads=sleepfm_fusion_num_heads,
                    dropout=clf_dropout,
                    temporal_pool_type=sleepfm_temporal_pool_type,
                    center_bias=cast(
                        Literal["learned", "gaussian", "linear"],
                        (
                            sleepfm_center_bias
                            if sleepfm_center_bias is not None
                            else "gaussian"
                        ),
                    ),
                    center_bias_strength=sleepfm_center_bias_strength,
                    temperature=sleepfm_temperature,
                )
                self.epoch_feature_fusion = None
        else:
            self.sleepfm_fusion = None
            self.epoch_feature_fusion = None
        self.slow_wave_occupancy_head: RelativeSlowWaveOccupancyHead | None = None
        self.slow_wave_occupancy_config: dict[str, Any] | None = None
        self.slow_wave_occupancy_loss_weight = 0.0
        self._last_slow_wave_occupancy_aux: (
            tuple[torch.Tensor, dict[str, torch.Tensor]] | None
        ) = None
        if slow_wave_occupancy_config is not None:
            self.configure_slow_wave_occupancy_head(
                slow_wave_occupancy_config, loss_weight=slow_wave_occupancy_loss_weight
            )
        self._init_fusion_gate_biases()
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
        if self.pos_encoding is None:
            state_dict.pop(f"{prefix}pos_encoding", None)
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
            (ResidualMLPClassifier, PerPositionSleepHead, LinearSleepHead),
        ):
            self.classifier.initialize_output_bias_uniform()
        elif isinstance(self.classifier, nn.Sequential):
            final_linear = cast(nn.Linear, self.classifier[-1])
            if hasattr(final_linear, "bias") and final_linear.bias is not None:
                nn.init.zeros_(final_linear.bias)

    def _init_fusion_gate_biases(self) -> None:
        """Initialize fusion gate biases to keep engineered features conservative early."""

        def _init_gate(gate: nn.Module | None) -> None:
            if gate is None:
                return
            linear: nn.Linear | None = None
            if isinstance(gate, nn.Sequential) and len(gate) > 0:
                first_layer = gate[0]
                if isinstance(first_layer, nn.Linear):
                    linear = first_layer
            elif isinstance(gate, nn.Linear):
                linear = gate
            if linear is not None and linear.bias is not None:
                nn.init.constant_(linear.bias, self.fusion_gate_bias)

        _init_gate(self.modality_gate)
        center_readout = getattr(self, "center_context_readout", None)
        if isinstance(
            center_readout, (CenterContextReadout, RelativeMultiheadCenterReadout)
        ):
            _init_gate(center_readout.gate)
        if self.sequential_fusion is not None:
            _init_gate(getattr(self.sequential_fusion, "gate", None))
        if self.cross_attn_fusion_layers is not None:
            for layer in self.cross_attn_fusion_layers:
                _init_gate(getattr(layer, "gate", None))

    def configure_slow_wave_occupancy_head(
        self,
        config: Mapping[str, Any],
        *,
        loss_weight: float,
        state_dict: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        """Install and optionally restore a SupCon occupancy head.

        The head reads the same pooled epoch-encoder representation used during
        SupCon pretraining. Its auxiliary loss is computed inside this module so
        supervised callers do not need to reproduce target extraction, masking,
        or checkpoint validation logic.

        Args:
            config: Versioned reconstruction metadata from the SupCon export.
            loss_weight: Non-negative supervised auxiliary-loss weight. Zero
                retains the head in the model/checkpoint without training it.
            state_dict: Optional learned head state to restore strictly.
        """
        resolved = dict(config)
        version = int(resolved.get("version", -1))
        if version != RELATIVE_SLOW_WAVE_TARGET_VERSION:
            raise ValueError(
                f"Unsupported slow-wave occupancy target version {version}; expected {RELATIVE_SLOW_WAVE_TARGET_VERSION}"
            )
        thresholds = tuple(float(value) for value in resolved.get("thresholds", ()))
        if thresholds != RELATIVE_SLOW_WAVE_THRESHOLDS:
            raise ValueError(
                f"Slow-wave occupancy threshold contract mismatch: checkpoint={thresholds}, runtime={RELATIVE_SLOW_WAVE_THRESHOLDS}"
            )
        in_dim = int(resolved.get("in_dim", -1))
        if in_dim != self._slow_wave_occupancy_encoder_dim:
            raise ValueError(
                f"Slow-wave occupancy head input mismatch: checkpoint in_dim={in_dim}, epoch encoder output={self._slow_wave_occupancy_encoder_dim}"
            )
        eeg_channels = int(resolved.get("eeg_channels", 0))
        eeg_indices = tuple(
            int(value) for value in resolved.get("eeg_channel_indices", ())
        )
        if eeg_channels <= 0 or len(eeg_indices) != eeg_channels:
            raise ValueError(
                f"Slow-wave occupancy config must provide one EEG input index per head channel, got eeg_channels={eeg_channels}, indices={eeg_indices}"
            )
        if any(index < 0 or index >= self.in_ch for index in eeg_indices):
            raise ValueError(
                f"Slow-wave occupancy EEG indices {eeg_indices} are incompatible with a {self.in_ch}-channel supervised model"
            )
        fs = float(resolved.get("fs", 0.0))
        if fs <= 4.0:
            raise ValueError(f"Slow-wave occupancy fs must exceed 4 Hz, got {fs}")
        weight = float(loss_weight)
        if weight < 0.0:
            raise ValueError(
                f"slow_wave_occupancy_loss_weight must be non-negative, got {weight}"
            )
        if weight > 0.0:
            if self.epoch_tokenization != "epoch":
                raise ValueError(
                    "Slow-wave occupancy fine-tuning requires epoch_tokenization='epoch'"
                )
            if (
                self.use_feature_extraction
                and self.fusion_mode != "sequential"
                and (not self.use_sleepfm_fusion)
            ):
                raise ValueError(
                    "Slow-wave occupancy fine-tuning with engineered features requires sequential or SleepFM fusion so the original CNN embedding remains observable"
                )
        head = RelativeSlowWaveOccupancyHead(
            in_dim, eeg_channels=eeg_channels, thresholds=thresholds
        )
        if state_dict is not None:
            head.load_state_dict(dict(state_dict), strict=True)
        reference_parameter = next(self.epoch_encoder.parameters(), None)
        if reference_parameter is not None:
            head.to(device=reference_parameter.device)
        self.slow_wave_occupancy_head = head
        resolved.update(
            {
                "version": version,
                "in_dim": in_dim,
                "eeg_channels": eeg_channels,
                "thresholds": thresholds,
                "fs": fs,
                "eeg_channel_indices": eeg_indices,
            }
        )
        self.slow_wave_occupancy_config = resolved
        self.slow_wave_occupancy_loss_weight = weight

    @property
    def enc_layers(self) -> nn.ModuleList:
        """Backward-compatible alias for legacy code that expects ``enc_layers``."""
        assert self.transformer is not None
        return self.transformer.layers

    def gradient_checkpointing_enable(
        self, cnn_only: bool = False, cnn_granularity: str = "block"
    ):
        """Enable gradient checkpointing for memory efficiency.

        Args:
            cnn_only: If True, only enable gradient checkpointing on the CNN
                (epoch encoder), not the transformer layers.
            cnn_granularity: CNN checkpoint boundary policy. ``"stage"`` and
                ``"full"`` are supported by the preact-asymmetric encoder.
        """

        def enable_epoch_encoder() -> None:
            enable_fn = getattr(
                self.epoch_encoder, "gradient_checkpointing_enable", None
            )
            if not callable(enable_fn):
                return
            if cnn_granularity == "block":
                enable_fn()
            else:
                enable_fn(granularity=cnn_granularity)

        if cnn_only:
            enable_epoch_encoder()
            return self
        self._gradient_checkpointing = True
        if self.axial_backbone is not None:
            self.axial_backbone.gradient_checkpointing_enable()
        enable_epoch_encoder()
        return self

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self._gradient_checkpointing = False
        if self.axial_backbone is not None:
            self.axial_backbone.gradient_checkpointing_disable()
        if hasattr(self.epoch_encoder, "gradient_checkpointing_disable"):
            cast(Any, self.epoch_encoder).gradient_checkpointing_disable()
        return self

    def _apply_scaled_position_encoding(self, z: torch.Tensor) -> torch.Tensor:
        """Add legacy absolute positions or leave relative-attention inputs unchanged.

        Args:
            z: Input tensor of shape ``[B, L, D]``.

        Returns:
            Tensor of shape ``[B, L, D]`` with positional encoding added.
        """
        if self.context_attention_mode == "relative_full":
            return z
        L = z.size(1)
        if self.pos_encoding is None:
            raise RuntimeError("Legacy positional encoding is unavailable")
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
        if isinstance(self.center_context_readout, nn.Identity):
            return z[:, z.size(1) // 2, :]
        readout = cast(
            CenterContextReadout | RelativeMultiheadCenterReadout,
            self.center_context_readout,
        )
        return readout(
            z, epoch_valid_mask=epoch_valid_mask, return_attention=return_attention
        )

    def _run_encoder(
        self,
        z: torch.Tensor,
        *,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        cross_tokens: torch.Tensor | None = None,
        cross_key_padding_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run transformer encoder with optional gradient checkpointing.

        Args:
            z: Input tensor of shape ``[B, L, D]``.
            attn_mask: Optional attention bias or mask.
            key_padding_mask: Optional padding mask for attention.
            cross_tokens: Optional context tokens for cross-attention fusion.
            cross_key_padding_mask: Optional padding mask for cross-attention keys.

        Returns:
            Output tensor ``[B, L, D]`` and optional last-layer per-head
            attention weights ``[B, H, L, L]``.
        """
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
                if (
                    self.cross_attn_fusion_layers is not None
                    and cross_tokens is not None
                ):
                    z = self.cross_attn_fusion_layers[idx](
                        z, cross_tokens, key_padding_mask=cross_key_padding_mask
                    )
                is_last = idx == len(transformer.layers) - 1
                if is_last:
                    if isinstance(layer, RelativePositionTransformerEncoderLayer):
                        z, attention_weights = layer.forward_with_attention(
                            z, src_mask=attn_mask, src_key_padding_mask=key_padding_mask
                        )
                    else:
                        z, attention_weights = _encoder_layer_forward_with_attention(
                            cast(nn.TransformerEncoderLayer, layer),
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
        if self.cross_attn_fusion_layers is not None and cross_tokens is not None:
            checkpoint_threshold = self.transformer_checkpoint_threshold
            for idx, layer in enumerate(transformer.layers):
                z = self.cross_attn_fusion_layers[idx](
                    z, cross_tokens, key_padding_mask=cross_key_padding_mask
                )
                if (
                    self._gradient_checkpointing
                    and self.training
                    and (idx >= checkpoint_threshold)
                ):

                    def _layer_forward(
                        x: torch.Tensor, _layer: nn.Module = layer
                    ) -> torch.Tensor:
                        return cast(
                            torch.Tensor,
                            _layer(
                                x,
                                src_mask=attn_mask,
                                src_key_padding_mask=key_padding_mask,
                            ),
                        )

                    z = cast(
                        torch.Tensor, checkpoint(_layer_forward, z, use_reentrant=False)
                    )
                else:
                    z = layer(
                        z, src_mask=attn_mask, src_key_padding_mask=key_padding_mask
                    )
        elif self._gradient_checkpointing and self.training:
            checkpoint_threshold = self.transformer_checkpoint_threshold
            for idx, layer in enumerate(transformer.layers):
                if idx >= checkpoint_threshold:
                    if attn_mask is None and key_padding_mask is None:
                        z = cast(
                            torch.Tensor, checkpoint(layer, z, use_reentrant=False)
                        )
                    else:

                        def _layer_forward(
                            x: torch.Tensor, _layer: nn.Module = layer
                        ) -> torch.Tensor:
                            return cast(
                                torch.Tensor,
                                _layer(
                                    x,
                                    src_mask=attn_mask,
                                    src_key_padding_mask=key_padding_mask,
                                ),
                            )

                        z = cast(
                            torch.Tensor,
                            checkpoint(_layer_forward, z, use_reentrant=False),
                        )
                else:
                    z = layer(
                        z, src_mask=attn_mask, src_key_padding_mask=key_padding_mask
                    )
        else:
            z = transformer(z, mask=attn_mask, src_key_padding_mask=key_padding_mask)
        z = self.output_norm(z)
        return (z, attention_weights)

    def init_projection_as_identity(self) -> None:
        """Initialize projection layer as identity transformation.

        Call this method after loading a pretrained encoder to preserve the
        pretrained CNN features through the projection layer. This is only
        valid when the projection input and output dimensions match.

        Raises:
            ValueError: If projection input and output dimensions don't match.
        """
        if isinstance(self.proj, nn.Identity):
            return
        if not isinstance(self.proj, nn.Linear):
            raise ValueError("This variant has no legacy projection layer")
        if self.proj.in_features == self.proj.out_features:
            nn.init.eye_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)
        else:
            raise ValueError(
                f"Cannot use identity init: proj is {self.proj.in_features} -> {self.proj.out_features}"
            )

    def _apply_center_masking(self, z: torch.Tensor) -> torch.Tensor:
        """Randomly mask center epoch to force context learning (BERT-style).

        During training, with probability `center_mask_prob`, replaces the center
        epoch's features with a learned mask token. This forces the model to learn
        attention patterns that gather context from neighboring epochs, since it
        cannot rely on the center epoch's original features.

        Args:
            z: Input tensor of shape [B, L, D] after projection.

        Returns:
            Tensor of shape [B, L, D] with center epoch potentially masked.
        """
        if not self.training or self.center_mask_prob <= 0:
            return z
        if self.mask_token is None:
            return z
        B, L, D = z.shape
        center = L // 2
        mask = torch.rand(B, device=z.device) < self.center_mask_prob
        z = z.clone()
        center_slice = z[:, center : center + 1, :]
        masked_center = torch.where(
            mask.view(B, 1, 1), self.mask_token.expand(B, 1, D), center_slice
        )
        z[:, center : center + 1, :] = masked_center
        return z

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
        """Apply the configured learnable or fixed classifier temperature."""
        if self.log_temperature is not None:
            min_log_temp = math.log(0.1)
            max_log_temp = math.log(10.0)
            log_temp = torch.nan_to_num(
                self.log_temperature, nan=0.0, posinf=max_log_temp, neginf=min_log_temp
            ).clamp(min=min_log_temp, max=max_log_temp)
            return logits / log_temp.exp()
        if self.classifier_temperature != 1.0:
            return logits / self.classifier_temperature
        return logits

    def _build_epoch_grid(
        self, *, d_model: int, dropout: float, patch_token_dim: int | None
    ) -> None:
        """Construct the epoch-patch grid tokenizer and validate the encoder.

        Grid mode drives ``self.transformer`` directly, so every incompatibility
        is rejected here rather than surfacing as a shape error mid-forward.

        Args:
            d_model: Context transformer width.
            dropout: Dropout for the tokenizer and the assembled grid.
            patch_token_dim: Tokenizer width before projecting to ``d_model``.

        Raises:
            ValueError: If the configuration cannot support grid tokenization.
        """
        if self.transformer is None:
            raise ValueError(
                "epoch_tokenization='grid' drives the standard transformer encoder, which this configuration did not build."
            )
        if self.use_feature_extraction:
            raise ValueError(
                "epoch_tokenization='grid' is incompatible with use_feature_extraction=True; engineered features are epoch-level and have no place in a patch grid."
            )
        if self.use_n1_attention:
            raise ValueError(
                "epoch_tokenization='grid' does not support use_n1_attention."
            )
        if self.context_attention_mode == "relative_full":
            raise ValueError(
                f"context_attention_mode='relative_full' cannot be used with epoch_tokenization='grid': its relative bias table is sized to context_epochs={self.context_epochs} and cannot index a {self.context_epochs * (int(self.patches_per_epoch) + 1)}-token grid. Use grid_attention_mode='relative_factorized' instead."
            )
        encoder = self.epoch_encoder
        temporal_dim = getattr(encoder, "temporal_dim", None)
        temporal_len = getattr(encoder, "final_temporal_len", None)
        if not hasattr(encoder, "forward_temporal") or temporal_dim is None:
            raise ValueError(
                f"epoch_tokenization='grid' requires an epoch encoder exposing forward_temporal() and temporal_dim; {type(encoder).__name__} does not."
            )
        if temporal_len is None:
            raise ValueError(
                "epoch_tokenization='grid' requires the epoch encoder to expose final_temporal_len."
            )
        patches = int(self.patches_per_epoch)
        if patches < 1:
            raise ValueError(f"patches_per_epoch must be >= 1, got {patches}")
        if int(temporal_len) % patches != 0:
            raise ValueError(
                f"patches_per_epoch={patches} must divide the encoder's final_temporal_len={int(temporal_len)}."
            )
        if getattr(encoder, "pool", None) is not None:
            logging.getLogger(__name__).warning(
                "epoch_tokenization='grid' consumes forward_temporal(), so the epoch encoder's pooling head (pooling_mode=%r) is never called. A pretrained encoder still transfers its trunk, but its pooled representation is unused.",
                getattr(encoder, "pooling_mode", "unknown"),
            )
        self.epoch_grid = EpochPatchGrid(
            d_model=d_model,
            cnn_dim=int(temporal_dim),
            num_patches=patches,
            temporal_len=int(temporal_len),
            patch_token_dim=patch_token_dim,
            dropout=dropout,
        )

    def _reduce_grid_attention(
        self, attention: torch.Tensor, seq_len: int
    ) -> torch.Tensor:
        """Collapse grid attention ``[B,H,S,S]`` to the epoch contract ``[B,H,L,L]``.

        Sums over key slots so attention mass is preserved, then averages over
        query slots so each entry reads as "how much epoch i attended to epoch j".

        Args:
            attention: Per-head weights over the flat grid ``[B, H, S, S]``.
            seq_len: Context length ``L``.

        Returns:
            Epoch-level attention ``[B, H, L, L]``.
        """
        grid = self.epoch_grid
        assert grid is not None
        slots = grid.slots_per_epoch
        b, h, _, _ = attention.shape
        blocked = attention.view(b, h, seq_len, slots, seq_len, slots)
        return blocked.sum(dim=5).mean(dim=3)

    def _forward_grid_encoder(
        self,
        feat: torch.Tensor,
        batch: int,
        seq_len: int,
        epoch_valid: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the shared transformer over the epoch-patch token grid.

        Args:
            feat: Encoder temporal features ``[B*L, T', d_cnn]``.
            batch: Batch size ``B``.
            seq_len: Context length ``L``.
            epoch_valid: Availability mask ``[B, L]``.
            return_attention: Whether to export epoch-level attention weights.

        Returns:
            Per-epoch representations ``[B, L, d_model]`` and optional
            epoch-level attention ``[B, H, L, L]``.
        """
        grid = self.epoch_grid
        assert grid is not None
        epoch_pos: torch.Tensor | None = None
        if self.grid_attention_mode == "absolute":
            if self.pos_encoding is None:
                raise RuntimeError(
                    "grid_attention_mode='absolute' needs the learned epoch position table, which this configuration did not build."
                )
            if seq_len > self.pos_encoding.size(1):
                raise ValueError(
                    f"context length {seq_len} exceeds positional table length {self.pos_encoding.size(1)}"
                )
            epoch_pos = self.pos_encoding[0, :seq_len, :]
        center_mask: torch.Tensor | None = None
        if (
            self.training
            and self.center_mask_prob > 0
            and (self.mask_token is not None)
        ):
            center_mask = torch.rand(batch, device=feat.device) < self.center_mask_prob
        flat = grid.to_grid(
            feat,
            seq_len,
            epoch_pos,
            center_mask=center_mask,
            center_mask_token=self.mask_token if center_mask is not None else None,
        )
        padding_mask = grid.expand_padding_mask(epoch_valid)
        flat_out, attention = self._run_encoder(
            flat, key_padding_mask=padding_mask, return_attention=return_attention
        )
        z, _patch_tokens = grid.from_grid(flat_out, seq_len, epoch_valid=epoch_valid)
        if attention is not None:
            self._last_grid_attention_weights = attention
            attention = self._reduce_grid_attention(attention, seq_len)
        return (z, attention)

    def forward_features(
        self,
        inputs: WaveformInputs,
        *,
        all_positions: bool = False,
        return_engineered_features: bool = False,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Prepare inputs and return backbone features before classification."""
        prepared = self.input_preparer(inputs)
        return self._forward_features_prepared(
            prepared,
            inputs,
            all_positions=all_positions,
            return_engineered_features=return_engineered_features,
            return_attention=return_attention,
        )

    def _forward_features_prepared(
        self,
        prepared: PreparedContextWaveforms,
        inputs: WaveformInputs,
        *,
        all_positions: bool = False,
        return_engineered_features: bool = False,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """
        Return backbone features before the classifier.

        Args:
            inputs: Input tensor or mapping with 'wave' key.
            all_positions: If True, return features for all positions [B, L, F].
                          If False, return only center position [B, F].
            return_engineered_features: If True, also return engineered features
                aligned with the returned positions (center or all).

        Returns:
            If return_engineered_features is False:
                Feature tensor of shape [B, d_model] or [B, L, d_model].
                When use_sleepfm_fusion=True, sleepfm_mode='pool', and all_positions=True, returns
                (feats_3d, sleepfm_pooled_center) tuple to keep the gradient
                path explicit.
            If return_engineered_features is True:
                Tuple of (features, engineered_features).
        """
        if self.epoch_tokenization == "grid" and return_engineered_features:
            raise ValueError("grid tokenization does not expose engineered features")
        if (
            self.use_feature_extraction
            and self.training
            and (not torch.compiler.is_compiling())
        ):
            if (
                hasattr(self, "feature_extractor")
                and self.feature_extractor is not None
                and hasattr(self.feature_extractor, "stats_initialized")
            ):
                if not bool(cast(Any, self.feature_extractor).stats_initialized.item()):
                    logger = logging.getLogger(__name__)
                    logger.info(
                        "Feature statistics will be initialized on this first forward pass. This is expected for the first training batch."
                    )
        precomputed_features_raw = None
        if isinstance(inputs, Mapping) and "precomputed_features" in inputs:
            precomputed_features_raw = inputs["precomputed_features"]
        seq = prepared.waveform
        B, L, C, T = seq.shape
        x = prepared.cnn_waveform
        epoch_valid = prepared.epoch_valid_mask
        channel_mask = prepared.cnn_presence_mask
        has_wave_raw = isinstance(inputs, Mapping) and "wave_raw" in inputs
        x_raw = prepared.feature_waveform if has_wave_raw else None
        engineered_feats: torch.Tensor | None = None
        engineered_feats_raw: torch.Tensor | None = None
        attn_weights: torch.Tensor | None = None
        grid_temporal: torch.Tensor | None = None
        feats: torch.Tensor | None = None
        precomputed_features: dict[str, torch.Tensor] | None = None
        if precomputed_features_raw is not None:
            precomputed_features = {}
            precomputed_mapping = cast(
                dict[str, torch.Tensor | None], precomputed_features_raw
            )
            for key, val in precomputed_mapping.items():
                if val is None:
                    continue
                if val.dim() == 4:
                    precomputed_features[key] = val.reshape(
                        B * L, val.size(2), val.size(3)
                    )
                elif val.dim() == 3:
                    precomputed_features[key] = val.reshape(B * L, val.size(2))
                else:
                    precomputed_features[key] = val
        if self.use_feature_extraction and (
            self.fusion_mode == "sequential" or self.use_sleepfm_fusion
        ):
            if self.feature_extractor is None:
                raise RuntimeError(
                    "Standalone engineered feature path requires self.feature_extractor"
                )
            wave_for_features = x_raw if x_raw is not None else x
            eng_flat = self.feature_extractor(
                wave_for_features,
                channel_mask=channel_mask,
                precomputed_features=precomputed_features,
            )
            eng_flat = torch.nan_to_num(eng_flat, nan=0.0, posinf=0.0, neginf=0.0)
            engineered_feats = eng_flat.view(B, L, -1)
            engineered_feats_raw = engineered_feats
            guidance = eng_flat if self.cnn_feature_guidance else None
            cnn_flat = self.epoch_encoder(
                x,
                channel_mask=channel_mask,
                engineered_features=guidance,
                **self._encoder_recording_kwargs(prepared.recording_index, B, L),
            )
            cnn_flat = torch.nan_to_num(cnn_flat, nan=0.0, posinf=0.0, neginf=0.0)
            cnn_feats = cnn_flat.view(B, L, -1)
            if self.use_reconstruction_loss and self.training:
                self._last_cnn_targets = cnn_feats.detach()
                if engineered_feats is not None:
                    self._last_eng_targets = engineered_feats.detach()
            if self.use_sleepfm_fusion:
                feats = cnn_feats
                del cnn_flat, eng_flat
            else:
                if self.sequential_fusion is None:
                    raise RuntimeError(
                        "Sequential fusion mode requires sequential_fusion module"
                    )
                if not self.use_epoch_attention and self.feat_projection is not None:
                    engineered_feats = self.feat_projection(engineered_feats)
                feats = self.sequential_fusion(cnn_feats, engineered_feats)
                del cnn_feats, cnn_flat, eng_flat
        else:
            if x_raw is not None and self.use_feature_extraction:
                if isinstance(self.epoch_encoder, HybridFeatureExtractor):
                    encoder_input = {
                        "wave": x,
                        "wave_raw": x_raw,
                        "precomputed_features": precomputed_features,
                    }
                    feats = cast(
                        torch.Tensor,
                        self.epoch_encoder(
                            encoder_input,
                            channel_mask=channel_mask,
                            **self._encoder_recording_kwargs(
                                prepared.recording_index, B, L
                            ),
                        ),
                    )
                else:
                    feats = cast(
                        torch.Tensor,
                        self.epoch_encoder(
                            x,
                            channel_mask=channel_mask,
                            wave_raw=x_raw,
                            precomputed_features=precomputed_features,
                        ),
                    )
            elif self.epoch_tokenization == "grid":
                grid_temporal = cast(
                    torch.Tensor,
                    cast(Any, self.epoch_encoder).forward_temporal(
                        x,
                        channel_mask=channel_mask,
                        **self._encoder_recording_kwargs(
                            prepared.recording_index, B, L
                        ),
                    ),
                )
            else:
                feats = cast(
                    torch.Tensor,
                    self.epoch_encoder(
                        x,
                        channel_mask=channel_mask,
                        **self._encoder_recording_kwargs(
                            prepared.recording_index, B, L
                        ),
                    ),
                )
            if feats is not None:
                feats = feats.view(B, L, -1)
            if return_engineered_features and self.use_feature_extraction:
                extractor = getattr(self.epoch_encoder, "sleep_features", None)
                if extractor is not None:
                    wave_for_features = x_raw if x_raw is not None else x
                    eng_flat = extractor(
                        wave_for_features,
                        channel_mask=channel_mask,
                        precomputed_features=precomputed_features,
                    )
                    eng_flat = torch.nan_to_num(
                        eng_flat, nan=0.0, posinf=0.0, neginf=0.0
                    )
                    engineered_feats = eng_flat.view(B, L, -1)
                    engineered_feats_raw = engineered_feats
                    del eng_flat
        channel_mask_for_n1 = channel_mask if self.use_n1_attention else None
        del x, channel_mask
        if x_raw is not None:
            del x_raw
        if grid_temporal is not None:
            if (
                self.use_reconstruction_loss
                and self.training
                and (self._last_cnn_targets is None)
            ):
                pool = getattr(self.epoch_encoder, "pool", None)
                if pool is not None:
                    pooled = pool(grid_temporal.transpose(1, 2))
                    self._last_cnn_targets = pooled.view(B, L, -1).detach()
            with _sdp_kernel_context(self.sdp_backend):
                z, attn_weights = self._forward_grid_encoder(
                    grid_temporal, B, L, epoch_valid, return_attention=return_attention
                )
            del grid_temporal
        else:
            assert feats is not None
            feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
            if (
                self.use_reconstruction_loss
                and self.training
                and (self._last_cnn_targets is None)
            ):
                self._last_cnn_targets = feats.detach()
            z = self.proj(feats)
            if z.size(-1) != self.context_d_model:
                raise RuntimeError(
                    f"Context transformer input must have width d_model={self.context_d_model}, got {z.size(-1)} (epoch encoder out_dim={self.epoch_encoder_out_dim}, proj={('Identity' if self.proj_is_identity else 'Linear')}). Check epoch_pool_output_dim against d_model."
                )
            del feats
            z = self._apply_center_masking(z)
            z = self._apply_scaled_position_encoding(z)
            with _sdp_kernel_context(self.sdp_backend):
                z, attn_weights = self._run_encoder(
                    z, key_padding_mask=~epoch_valid, return_attention=return_attention
                )
        z = z * epoch_valid.to(z.dtype).unsqueeze(-1)
        self._last_attention_weights = attn_weights
        if self.use_n1_attention:
            x_for_n1 = prepared.feature_waveform
            if self.n1_attention is not None and self.n1_feature_extractor is not None:
                n1_features = self.n1_feature_extractor.extract_n1_features(
                    x_for_n1, channel_mask=channel_mask_for_n1
                )
                n1_features = n1_features.view(B, L, -1)
                center_n1_features = n1_features[:, L // 2, :]
                if all_positions:
                    z_flat = z.reshape(B * L, -1)
                    n1_flat = n1_features.reshape(B * L, -1)
                    z_modulated_flat = self.n1_attention(z_flat, n1_flat)
                    z = z_modulated_flat.view(B, L, -1)
                else:
                    z_center = z[:, L // 2, :]
                    z_center_modulated = self.n1_attention(z_center, center_n1_features)
                    center_idx = L // 2
                    z = torch.cat(
                        [
                            z[:, :center_idx, :],
                            z_center_modulated.unsqueeze(1),
                            z[:, center_idx + 1 :, :],
                        ],
                        dim=1,
                    )
        if (
            self.use_sleepfm_fusion
            and self.sleepfm_mode == "fuse_only"
            and (self.epoch_feature_fusion is not None)
        ):
            if engineered_feats is None:
                raise RuntimeError(
                    "Epoch feature fusion enabled but engineered features not available. Ensure use_feature_extraction=True."
                )
            engineered_feats = torch.clamp(engineered_feats, min=-50.0, max=50.0)
            engineered_core, yasa_features = self._split_sleepfm_engineered_sources(
                engineered_feats
            )
            return_weights = not self.training or self._sleepfm_debug_attention
            fused, sleepfm_weights = self.epoch_feature_fusion(
                cnn_features=z,
                engineered_features=engineered_core,
                yasa_features=yasa_features,
                return_weights=return_weights,
            )
            if sleepfm_weights is not None:
                self._sleepfm_attention_weights = sleepfm_weights
            z_out = fused
            if return_engineered_features:
                engineered_source = (
                    engineered_feats_raw
                    if engineered_feats_raw is not None
                    else engineered_feats
                )
                if engineered_source is None:
                    raise RuntimeError("Engineered features requested but unavailable.")
                eng_out = (
                    engineered_source
                    if all_positions
                    else engineered_source[:, L // 2, :]
                )
                eng_out = torch.nan_to_num(eng_out, nan=0.0, posinf=0.0, neginf=0.0)
                return (z_out, eng_out)
            return z_out
        if self.use_sleepfm_fusion and self.sleepfm_fusion is not None:
            if engineered_feats is None:
                raise RuntimeError(
                    "SleepFM fusion enabled but engineered features not available. Ensure use_feature_extraction=True."
                )
            engineered_feats = torch.clamp(engineered_feats, min=-50.0, max=50.0)
            return_weights = not self.training or self._sleepfm_debug_attention
            engineered_core, yasa_features = self._split_sleepfm_engineered_sources(
                engineered_feats
            )
            z_pooled, sleepfm_weights = self.sleepfm_fusion(
                cnn_features=z,
                engineered_features=engineered_core,
                yasa_features=yasa_features,
                center_idx=L // 2,
                return_weights=return_weights,
            )
            if sleepfm_weights is not None:
                self._sleepfm_attention_weights = sleepfm_weights
            if all_positions:
                z_out = torch.cat(
                    [z[:, : L // 2, :], z_pooled.unsqueeze(1), z[:, L // 2 + 1 :, :]],
                    dim=1,
                )
                self._sleepfm_pooled_center = z_pooled.detach()
            else:
                z_out = z_pooled
                self._sleepfm_pooled_center = None
            if return_engineered_features:
                engineered_source = (
                    engineered_feats_raw
                    if engineered_feats_raw is not None
                    else engineered_feats
                )
                if engineered_source is None:
                    raise RuntimeError("Engineered features requested but unavailable.")
                eng_out = (
                    engineered_source
                    if all_positions
                    else engineered_source[:, L // 2, :]
                )
                eng_out = torch.nan_to_num(eng_out, nan=0.0, posinf=0.0, neginf=0.0)
                if all_positions:
                    return (z_out, eng_out, z_pooled)
                return (z_out, eng_out)
            if all_positions:
                return (z_out, z_pooled)
            return z_out
        z_out = (
            z
            if all_positions
            else self._apply_center_context_readout(
                z, epoch_valid, return_attention=return_attention
            )
        )
        if return_engineered_features:
            if engineered_feats is None:
                raise RuntimeError(
                    "Requested engineered features but none were computed"
                )
            engineered_source = (
                engineered_feats_raw
                if engineered_feats_raw is not None
                else engineered_feats
            )
            eng_out = (
                engineered_source if all_positions else engineered_source[:, L // 2, :]
            )
            if torch.isnan(eng_out).any() or torch.isinf(eng_out).any():
                if not torch.compiler.is_compiling():
                    logger = logging.getLogger(__name__)
                    logger.error(
                        "NaN/Inf detected in engineered features. Sanitizing. Check feature extraction normalization."
                    )
                eng_out = torch.nan_to_num(eng_out, nan=0.0, posinf=0.0, neginf=0.0)
            return (z_out, eng_out)
        return z_out

    def forward(
        self,
        inputs: WaveformInputs,
        predict_all: bool = False,
        return_features: bool = False,
        return_reasoning: bool = False,
        return_attention: bool = False,
        stage_labels: torch.Tensor | None = None,
        recurrent_refinement_steps: int | None = None,
    ) -> ForwardOutput:
        """
        Forward pass.

        Args:
            inputs: Input tensor or mapping with 'wave' key.
            predict_all: If True, return predictions for all epochs.
            return_features: If True, return tuple (logits, features).
            return_reasoning: If True and using CoT classifier, return the reasoning
                trace with hierarchical decision probabilities and feature predictions.
            return_attention: If True, return attention weights from the last
                transformer encoder layer. The attention weights have shape
                [B, n_heads, seq_len, seq_len] where seq_len is the context length.
                Useful for attention visualization and interpretability analysis.
            stage_labels: Optional stage labels (currently unused).
            recurrent_refinement_steps: Optional inference-time override for the
                trained recurrent depth. Requires an enabled recurrent refiner.

        Returns:
            logits: [B, num_classes] or [B, L, num_classes] if predict_all=True.
            If return_features=True, returns (logits, features).
            If return_attention=True, attention weights are appended as the last element
                of the returned tuple. Shape: [B, n_heads, seq_len, seq_len].
        """
        del stage_labels
        self._sleepfm_pooled_center = None
        self._last_cnn_targets = None
        self._last_eng_targets = None
        self._last_confidence_logits = None
        self._last_confidence = None
        self._last_confidence_pred_classes = None
        self._last_slow_wave_occupancy_aux = None
        if hasattr(self.center_context_readout, "_last_attention_weights"):
            cast(Any, self.center_context_readout)._last_attention_weights = None
        need_all_positions = (
            predict_all
            or return_features
            or self.use_neighbor_prediction
            or (self.recurrent_refiner is not None)
        )
        prepared = self.input_preparer(inputs)
        feats_result = self._forward_features_prepared(
            prepared,
            inputs,
            all_positions=need_all_positions,
            return_engineered_features=False,
            return_attention=return_attention,
        )
        sleepfm_center: torch.Tensor | None = None
        if isinstance(feats_result, tuple):
            if not feats_result or not isinstance(feats_result[0], torch.Tensor):
                raise RuntimeError(
                    "forward_features returned an empty or invalid tuple"
                )
            feats = feats_result[0]
            last = (
                feats_result[len(feats_result) - 1] if len(feats_result) > 0 else None
            )
            has_explicit_sleepfm_center = len(feats_result) >= 2
            if has_explicit_sleepfm_center:
                last_tensor = last if isinstance(last, torch.Tensor) else None
                if (
                    last_tensor is not None
                    and last_tensor.dim() == 2
                    and self.use_sleepfm_fusion
                ):
                    sleepfm_center = last_tensor
        else:
            feats = feats_result
        assert isinstance(feats, torch.Tensor)
        recurrent_states: tuple[torch.Tensor, ...] | None = None
        if self.recurrent_refiner is not None:
            if feats.dim() != 3:
                raise RuntimeError(
                    "recurrent refinement requires all-position context features"
                )
            refined_states = self.recurrent_refiner(
                feats, prepared.epoch_valid_mask, steps=recurrent_refinement_steps
            )
            recurrent_states = refined_states
            feats = refined_states[-1]
        elif recurrent_refinement_steps is not None:
            raise ValueError(
                "recurrent_refinement_steps was provided at inference, but this checkpoint has no trained recurrent refiner"
            )
        self._last_features = feats if need_all_positions and feats.dim() == 3 else None
        confidence: torch.Tensor | None = None
        if predict_all:
            classifier_input = feats
            center_feat = feats
        else:
            if sleepfm_center is not None:
                center = sleepfm_center
            elif feats.dim() == 3:
                center = self._apply_center_context_readout(
                    feats, prepared.epoch_valid_mask, return_attention=return_attention
                )
            else:
                center = feats
            classifier_input = center
            center_feat = center
        if isinstance(self.classifier, (ResidualMLPClassifier, LinearSleepHead)):
            logits, confidence = self._compute_standard_head_outputs(classifier_input)
        else:
            logits = self.classifier(classifier_input)
        logits = self._apply_temperature_scaling(logits)
        recurrent_logits: tuple[torch.Tensor, ...] | None = None
        if predict_all and recurrent_states is not None:
            earlier_logits = tuple(
                self._apply_temperature_scaling(self.classifier(state))
                for state in recurrent_states[:-1]
            )
            recurrent_logits = (*earlier_logits, logits)
        if self._last_confidence_logits is not None:
            pred_classes = logits.argmax(dim=-1)
            if self.confidence_head_detach_target:
                pred_classes = pred_classes.detach()
            self._last_confidence_pred_classes = pred_classes
        if (
            self.is_learned_feature_axial or self.epoch_tokenization == "grid"
        ) and predict_all:
            logits = logits * prepared.epoch_valid_mask.to(
                device=logits.device, dtype=logits.dtype
            ).unsqueeze(-1)
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
            reasoning=None,
            confidence=confidence,
            recurrent_logits=recurrent_logits,
        )

    def _split_sleepfm_engineered_sources(
        self, engineered_feats: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Split engineered features into core and optional YASA branches."""
        total_dim = engineered_feats.size(-1)
        core_dim = int(max(0, min(self._sleepfm_engineered_core_dim, total_dim)))
        core_features = engineered_feats[..., :core_dim]
        if self._sleepfm_yasa_dim > 0:
            yasa_features = engineered_feats[..., core_dim:]
            if yasa_features.size(-1) == 0:
                raise RuntimeError(
                    "SleepFM 3-source fusion expects YASA features but split produced an empty YASA branch. Check feature extractor dimensions."
                )
            return (core_features, yasa_features)
        return (core_features, None)

    def clear_last_attention_weights(self) -> None:
        """Discard cached attention weights to release graph references."""
        self._last_attention_weights = None
        if hasattr(self.center_context_readout, "_last_attention_weights"):
            cast(Any, self.center_context_readout)._last_attention_weights = None
        self._last_confidence_logits = None
        self._last_confidence = None
        self._last_confidence_pred_classes = None
        self._sleepfm_attention_weights = None
        self._sleepfm_source_entropy = None
        self._sleepfm_source_mean_weights = None
        if self.sleepfm_fusion is not None:
            self.sleepfm_fusion._last_feature_attention_weights = None
            self.sleepfm_fusion._last_temporal_attention_weights = None
            self.sleepfm_fusion.feature_fusion._last_source_weights = None
