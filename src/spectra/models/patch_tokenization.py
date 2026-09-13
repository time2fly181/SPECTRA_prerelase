"""Patch geometry primitives shared by the grid tokenizer and the epoch pool.

This module turns a CNN's pre-pool temporal feature map (the output of
``forward_temporal`` on the asymmetric epoch encoders) into a short sequence of
*patch tokens* per 30 s epoch.

The trunk is **16× single-resolution**: a 30 s epoch at 128 Hz produces a
feature map of shape ``[N, T', D]`` with ``T' = 240`` and ``D = 256``. With
``patches_per_epoch = 10`` the tokenizer uses ``kernel = stride = T' // P = 24``,
yielding exactly ``P = 10`` tokens of 3.0 s that tile the epoch without overlap.

There are two consumers of this geometry, and they are not interchangeable:

:class:`spectra.models.EpochPatchGrid`
    The **tokenization mode**. It carries patch tokens across the epoch
    boundary so one transformer attends over the whole ``L x (P+1)`` grid.
    Positions are factorized 2D — an epoch position plus a within-epoch slot
    position, combined the way :func:`factorized_patch_positions` describes.

:class:`IntraEpochAttentionPool`
    An epoch-encoder **pooling head**, not a tokenization mode. It consumes the
    pre-pool feature map of one epoch, restores within-epoch temporal order with
    learned patch positions and pre-norm attention blocks, then aggregates back
    to exactly **one vector per epoch**. The downstream context model still sees
    ``[B, L, d]`` and is completely unchanged — no patch token ever crosses the
    epoch boundary. This is what ``pooling_mode="intra_epoch"`` selects, and it
    is orthogonal to the grid.

Usage::

    >>> import torch
    >>> from spectra.models import PatchTokenizer, factorized_patch_positions
    >>> feat = torch.randn(8, 240, 256)  # [B*L, T', D] from forward_temporal
    >>> tok = PatchTokenizer(d_cnn=256, d_token=512, temporal_len=240, num_patches=12)
    >>> patches = tok(feat)  # [8, 12, 512]
    >>> epoch_pos = torch.randn(4, 512)  # L=4 epoch positions
    >>> patch_pos = torch.randn(12, 512)  # P=12 within-epoch positions
    >>> pos = factorized_patch_positions(epoch_pos, patch_pos)  # [48, 512]
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .common import kaiming_init_, transformer_init_

__all__ = [
    "PatchTokenizer",
    "PatchPreNormBlock",
    "IntraEpochAttentionPool",
    "factorized_patch_positions",
]


class PatchTokenizer(nn.Module):
    """Tokenize a CNN temporal feature map into per-epoch patches.

    Applies a depthwise strided convolution (one filter per CNN channel) followed
    by a pointwise (1x1) convolution that mixes channels into the token
    dimension. Kernel and stride both equal ``temporal_len // num_patches`` so the
    patches tile the epoch without overlap and the output length is exactly
    ``num_patches``. Shapes are fully static, so the module is
    ``torch.compile(fullgraph=True)``-safe.

    Args:
        d_cnn: Channel dimension of the CNN feature map (``D`` in ``[N, T', D]``).
        d_token: Output token dimension produced by the pointwise convolution.
        temporal_len: Temporal length ``T'`` of the CNN feature map.
        num_patches: Number of patches ``P`` per epoch. Must divide ``temporal_len``.
        dropout: Dropout applied to the resulting token sequence.

    Raises:
        ValueError: If ``num_patches`` does not evenly divide ``temporal_len`` or
            either is non-positive.
    """

    def __init__(
        self,
        *,
        d_cnn: int,
        d_token: int,
        temporal_len: int,
        num_patches: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_patches <= 0:
            raise ValueError(f"num_patches must be > 0, got {num_patches}")
        if temporal_len <= 0:
            raise ValueError(f"temporal_len must be > 0, got {temporal_len}")
        if temporal_len % num_patches != 0:
            raise ValueError(
                f"temporal_len ({temporal_len}) must be divisible by "
                f"num_patches ({num_patches}) for non-overlapping patches."
            )

        self.d_cnn = int(d_cnn)
        self.d_token = int(d_token)
        self.temporal_len = int(temporal_len)
        self.num_patches = int(num_patches)
        self.patch_size = self.temporal_len // self.num_patches

        # Depthwise: one filter per CNN channel, strided to downsample T' -> P.
        self.depthwise = nn.Conv1d(
            d_cnn,
            d_cnn,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            groups=d_cnn,
        )
        # Pointwise: mix channels into the token dimension.
        self.pointwise = nn.Conv1d(d_cnn, d_token, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.apply(kaiming_init_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Tokenize a temporal feature map.

        Args:
            x: CNN temporal features of shape ``[N, T', d_cnn]`` where ``N`` is the
                flattened batch-times-epoch dimension.

        Returns:
            Patch tokens of shape ``[N, P, d_token]``.
        """
        if x.dim() != 3:
            raise ValueError(
                f"PatchTokenizer expected [N, T', D], got {tuple(x.shape)}"
            )
        x = x.transpose(1, 2)  # [N, d_cnn, T']
        x = self.depthwise(x)  # [N, d_cnn, P]
        x = self.pointwise(x)  # [N, d_token, P]
        x = x.transpose(1, 2)  # [N, P, d_token]
        return self.dropout(x)


class PatchPreNormBlock(nn.Module):
    """Pre-norm transformer encoder block with optional attention export.

    Args:
        d_model: Feature dimension.
        nhead: Number of attention heads.
        dim_ff: Feed-forward hidden dimension.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_ff: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the pre-norm block.

        Args:
            x: Input tokens of shape ``[N, S, d_model]``.
            return_attention: If True, also return attention weights.

        Returns:
            Tuple of the updated tokens and optional attention weights of shape
            ``[N, nhead, S, S]`` (``None`` when ``return_attention`` is False).
        """
        attn_input = self.norm1(x)
        attn_out, attn_weights = self.self_attn(
            attn_input,
            attn_input,
            attn_input,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        x = x + self.dropout1(attn_out)
        x = x + self.mlp(self.norm2(x))
        return x, (attn_weights if return_attention else None)


class IntraEpochAttentionPool(nn.Module):
    """Order-aware intra-epoch pooling head for asymmetric epoch encoders.

    Drop-in replacement for :class:`~spectra.models.learnable_pooling.LatentQueryAttentionPool`
    in an epoch encoder's pooling stage. It accepts the same channels-first
    ``[N, C, T']`` feature map and returns the same ``[N, out_dim]`` epoch vector,
    but the collapse from ``T'`` positions to one vector is no longer
    permutation-invariant:

    1. :class:`PatchTokenizer` tiles ``T'`` into ``P`` non-overlapping patches,
    2. a learned within-epoch position embedding is added to each patch,
    3. ``num_layers`` **pre-norm** blocks (:class:`PatchPreNormBlock`) attend
       across the ``P`` patches of that epoch only,
    4. a terminal :class:`~torch.nn.LayerNorm` closes the pre-norm stack, and
    5. a single learned score per token drives a softmax aggregation over the
       ``P`` tokens, followed by an output projection.

    The same weights are applied to every epoch in the context window, and no
    patch token ever leaves the epoch: the module returns exactly one vector per
    epoch, so a host context model still sees ``[B, L, out_dim]``. This is
    deliberately *not* the grid path — see :class:`spectra.models.EpochPatchGrid`,
    which is a whole context front-end and replaces the host transformer.

    Every epoch is processed independently, so no key-padding mask is needed or
    accepted; invalid epochs are handled by the host model's ``[B, L]`` mask
    exactly as they are for the order-blind pools. The softmax always runs over
    ``P >= 1`` real tokens, so the output is finite in both ``train()`` and
    ``eval()`` regardless of epoch validity.

    Args:
        channels: Channel dimension ``C`` of the encoder feature map.
        temporal_len: Temporal length ``T'`` of the encoder feature map. Must be
            divisible by ``num_patches``.
        num_patches: Number of non-overlapping within-epoch patches ``P``.
        num_layers: Number of intra-epoch pre-norm transformer blocks.
        token_dim: Width of the intra-epoch tokens (``None`` -> ``channels``).
        nhead: Attention heads in each intra-epoch block. Must divide ``token_dim``.
        ff_mult: Feed-forward expansion multiple of ``token_dim``.
        dropout: Dropout probability used by the tokenizer, blocks, and output.
        out_dim: Epoch-vector width produced by the output projection
            (``None`` -> ``token_dim``).
        aggregation: How the ``P`` within-epoch tokens collapse to one vector.

            * ``"attention"`` -- learned softmax weights, then a weighted sum.
              This is **not** permutation-invariant: permuting the patches
              re-pairs content with position, so the summands themselves change
              rather than merely being reordered. The order sensitivity is
              carried entirely by how much ``patch_pos`` contributes to each
              token's representation, which is small at initialization. Given a
              loss that needs position, training grows ``patch_pos`` and this
              readout solves position-dependent tasks exactly -- measured at
              100% on a synthetic which-half-of-the-epoch task, with
              ``patch_pos`` RMS growing 0.15 -> 0.40 and the aggregation weights
              staying near-uniform throughout. Given a loss that does not need
              position (SupCon on whole-epoch embeddings), it decays instead.
            * ``"concat"`` -- concatenate the ``P`` tokens in order and project
              ``P * token_dim -> out_dim``. Position-indexed by construction, so
              it is order-sensitive from step 0 and learns position-dependent
              tasks several times faster (the same synthetic task: solved in
              <400 steps versus ~2000). Costs ``P`` times more projection
              parameters (16 x 256 -> 512 is 2.1M versus 0.13M).

            Neither is strictly better. ``attention`` is the cheaper default and
            is capable of order; ``concat`` removes the need to learn its way
            there.
        pos_init_std: Initialization std of the learned patch positions. Only
            meaningful relative to the token scale -- see ``version``.
        version: Pool topology.

            * ``1`` -- the original topology: raw tokenizer output, positions
              added directly, plain softmax aggregation. Retained so existing
              ``intra_epoch`` checkpoints reconstruct exactly.
            * ``2`` (default) -- adds a :class:`~torch.nn.LayerNorm` on the
              tokens before the positions are added, and a learned temperature
              on the aggregation scores.

            Version 1 has a measured failure mode: the tokenizer is an
            unnormalized convolution, so its output scale is set by whatever the
            CNN trunk happens to produce. On a trained trunk the tokens reached
            an RMS ~50x the ``0.02``-std positions, leaving the positional signal
            at ~2% of the token magnitude -- far too weak to influence the
            output, which was ~0.999 cosine-invariant to permuting the patches.
            ``token_norm`` makes the token scale ~1 per dimension regardless of
            the trunk, so ``pos_init_std`` expresses a real position-to-token
            ratio. The learned temperature is a second, independent degree of
            freedom for the aggregation; note that a model learning to use
            position does **not** necessarily sharpen it (in the synthetic test
            above the weights stayed near-uniform while ``patch_pos`` grew), so
            ``patch_pos`` RMS is the diagnostic to watch, not entropy.

    Raises:
        ValueError: If sizes are non-positive, ``temporal_len`` is not divisible
            by ``num_patches``, ``token_dim`` is not divisible by ``nhead``,
            ``aggregation`` is unknown, or ``version`` is not 1 or 2.
    """

    def __init__(
        self,
        *,
        channels: int,
        temporal_len: int,
        num_patches: int = 16,
        num_layers: int = 2,
        token_dim: int | None = None,
        nhead: int = 8,
        ff_mult: float = 2.0,
        dropout: float = 0.1,
        out_dim: int | None = None,
        aggregation: str = "attention",
        pos_init_std: float = 0.15,
        version: int = 2,
    ) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError(f"channels must be > 0, got {channels}")
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if aggregation not in ("attention", "concat"):
            raise ValueError(
                f"IntraEpochAttentionPool supports aggregation in "
                f"('attention', 'concat'), got {aggregation!r}"
            )
        if version not in (1, 2):
            raise ValueError(f"version must be 1 or 2, got {version}")
        if pos_init_std <= 0:
            raise ValueError(f"pos_init_std must be > 0, got {pos_init_std}")

        resolved_token_dim = int(token_dim or channels)
        if resolved_token_dim % nhead != 0:
            raise ValueError(
                f"token_dim ({resolved_token_dim}) must be divisible by nhead ({nhead})"
            )
        if temporal_len % num_patches != 0:
            raise ValueError(
                f"Encoder temporal length ({temporal_len}) is not divisible by "
                f"num_patches ({num_patches}); non-overlapping patches would drop "
                f"{temporal_len % num_patches} trailing positions. Choose a patch "
                f"count that divides {temporal_len}."
            )

        self.channels = int(channels)
        self.temporal_len = int(temporal_len)
        self.num_patches = int(num_patches)
        self.num_layers = int(num_layers)
        self.token_dim = resolved_token_dim
        self.nhead = int(nhead)
        self.ff_mult = float(ff_mult)
        self.aggregation = aggregation
        self.out_dim = int(out_dim or resolved_token_dim)
        self.patch_size = self.temporal_len // self.num_patches
        self.version = int(version)
        self.pos_init_std = float(pos_init_std)

        self.tokenizer = PatchTokenizer(
            d_cnn=self.channels,
            d_token=self.token_dim,
            temporal_len=self.temporal_len,
            num_patches=self.num_patches,
            dropout=dropout,
        )
        # The tokenizer is an unnormalized convolution, so its output scale is
        # whatever the CNN trunk produces. Normalizing here makes the token scale
        # ~1 per dimension for any trunk, which is what lets ``pos_init_std``
        # express a meaningful position-to-token ratio.
        self.token_norm: nn.Module = (
            nn.LayerNorm(self.token_dim) if self.version >= 2 else nn.Identity()
        )
        # Learned within-epoch positions, added *before* self-attention so patch
        # order is part of the attention computation rather than a post-hoc tag.
        self.patch_pos = nn.Parameter(
            torch.randn(1, self.num_patches, self.token_dim) * self.pos_init_std
        )
        self.input_dropout = nn.Dropout(dropout)

        dim_ff = max(self.token_dim, int(math.ceil(self.token_dim * ff_mult)))
        self.blocks = nn.ModuleList(
            [
                PatchPreNormBlock(self.token_dim, self.nhead, dim_ff, dropout)
                for _ in range(self.num_layers)
            ]
        )
        # Terminal norm: required to close a pre-norm stack, whose residual
        # stream is otherwise unnormalized at the output.
        self.output_norm = nn.LayerNorm(self.token_dim)

        if self.aggregation == "attention":
            self.aggregation_head: nn.Module | None = nn.Linear(self.token_dim, 1)
            # Learned temperature on the aggregation scores. Init 0 -> temperature
            # 1, i.e. exactly the version-1 softmax, so a version-1 checkpoint
            # loaded into a version-2 module (this key missing) behaves
            # identically. Without it the scores stay near-uniform and the
            # weighted sum degenerates into a permutation-invariant mean.
            if self.version >= 2:
                self.log_temperature = nn.Parameter(torch.zeros(()))
            else:
                self.register_parameter("log_temperature", None)
            self.project = nn.Linear(self.token_dim, self.out_dim)
        else:
            # Concat readout: the token's index in the flattened vector IS its
            # position, so no separate scoring or temperature is needed.
            self.aggregation_head = None
            self.register_parameter("log_temperature", None)
            self.project = nn.Linear(self.num_patches * self.token_dim, self.out_dim)
        self.norm = nn.LayerNorm(self.out_dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._last_attn_weights: torch.Tensor | None = None
        # transformer_init_ keeps Kaiming for Conv1d, so the tokenizer's own
        # kaiming_init_ is preserved while the attention/linear stack gets Xavier.
        self.apply(transformer_init_)
        nn.init.normal_(self.patch_pos, mean=0.0, std=self.pos_init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool one epoch's feature map into a single order-aware vector.

        Args:
            x: Encoder features of shape ``[N, C, T']`` (channels-first, matching
                the other epoch-encoder pooling heads).

        Returns:
            Epoch vectors of shape ``[N, out_dim]``.

        Raises:
            ValueError: If the input rank, channel count, or temporal length does
                not match the configured geometry.
        """
        if x.dim() != 3:
            raise ValueError(
                f"IntraEpochAttentionPool expected [N, C, T'], got {tuple(x.shape)}"
            )
        if x.size(1) != self.channels:
            raise ValueError(
                f"IntraEpochAttentionPool expected {self.channels} channels, got "
                f"{x.size(1)}"
            )
        if x.size(2) != self.temporal_len:
            raise ValueError(
                f"IntraEpochAttentionPool expected temporal length "
                f"{self.temporal_len}, got {x.size(2)}"
            )

        # PatchTokenizer takes [N, T', C] and transposes back internally; the
        # round trip is a pure stride change, so no copy and no .view() on a
        # non-contiguous tensor is involved.
        tokens = self.tokenizer(x.transpose(1, 2))  # [N, P, token_dim]
        # Normalize before adding position (identity in version 1) so the
        # position-to-token magnitude ratio is set by pos_init_std, not by the
        # trunk's output scale.
        tokens = self.token_norm(tokens)
        tokens = tokens + self.patch_pos
        tokens = self.input_dropout(tokens)

        for block in self.blocks:
            tokens, _ = block(tokens)
        tokens = self.output_norm(tokens)

        if self.aggregation_head is None:
            # Concat: flatten in patch order, so the projection sees each token
            # at a fixed, position-specific slice of its input.
            self._last_attn_weights = None
            pooled = tokens.reshape(tokens.size(0), -1)  # [N, P * token_dim]
        else:
            scores = self.aggregation_head(tokens)  # [N, P, 1]
            if self.log_temperature is not None:
                # Clamped so the softmax cannot be driven to a hard argmax (which
                # would starve every other patch of gradient) or flattened away.
                temperature = self.log_temperature.exp().clamp(min=0.05, max=20.0)
                scores = scores / temperature
            weights = torch.softmax(scores.clamp(min=-50.0, max=50.0), dim=1)
            self._last_attn_weights = weights.squeeze(-1).detach()
            pooled = (weights * tokens).sum(dim=1)  # [N, token_dim]
        return self.drop(self.norm(self.project(pooled)))

    def get_attention_stats(self) -> dict[str, float] | None:
        """Return aggregation entropy and peak weight over the ``P`` patches.

        The diagnostic that matters is ``patch_pos_rms``: it is the magnitude of
        the positional signal relative to the (normalized) tokens, and it is what
        grows when a model learns to use within-epoch order. Entropy near
        ``log(num_patches)`` is *not* by itself evidence of order-blindness -- a
        model can reach exact position-dependent behavior with near-uniform
        aggregation weights, because permuting the patches changes the summands
        rather than merely reordering them.
        """
        if self.aggregation == "concat":
            # No aggregation weights exist; report the positional scale, which is
            # still the quantity worth watching.
            return {
                "patch_pos_rms": float(
                    self.patch_pos.detach().pow(2).mean().sqrt().item()
                )
            }
        if self._last_attn_weights is None:
            return None
        weights = self._last_attn_weights.float()
        entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=-1).mean()
        stats = {
            "patch_aggregation_entropy": float(entropy.item()),
            "patch_aggregation_entropy_uniform": float(math.log(self.num_patches)),
            "patch_aggregation_max_attn": float(
                weights.max(dim=-1).values.mean().item()
            ),
        }
        if self.log_temperature is not None:
            stats["patch_aggregation_temperature"] = float(
                self.log_temperature.detach().exp().clamp(0.05, 20.0).item()
            )
        stats["patch_pos_rms"] = float(
            self.patch_pos.detach().pow(2).mean().sqrt().item()
        )
        return stats

    def extra_repr(self) -> str:
        return (
            f"channels={self.channels}, temporal_len={self.temporal_len}, "
            f"num_patches={self.num_patches}, patch_size={self.patch_size}, "
            f"token_dim={self.token_dim}, num_layers={self.num_layers}, "
            f"nhead={self.nhead}, out_dim={self.out_dim}, "
            f"aggregation={self.aggregation!r}, version={self.version}, "
            f"pos_init_std={self.pos_init_std}"
        )


def factorized_patch_positions(
    epoch_pos: torch.Tensor,
    patch_pos: torch.Tensor,
) -> torch.Tensor:
    """Build factorized 2D positions for flat patch attention.

    Combines an epoch positional table (broadcast over patches) with an
    intra-epoch patch positional table (broadcast over epochs) so that every
    patch token gets a position that depends separably on both axes::

        pos[l, p] = epoch_pos[l] + patch_pos[p]

    The result is flattened in epoch-major order, matching a
    ``[B, L, P, D].reshape(B, L * P, D)`` token layout (token index ``l * P + p``).

    Args:
        epoch_pos: Epoch positions of shape ``[L, d_model]``.
        patch_pos: Patch positions of shape ``[P, d_model]``.

    Returns:
        Positions of shape ``[L * P, d_model]`` in epoch-major order.

    Raises:
        ValueError: If the feature dimensions of the two tables differ or either
            is not 2D.
    """
    if epoch_pos.dim() != 2 or patch_pos.dim() != 2:
        raise ValueError(
            "epoch_pos and patch_pos must be 2D [L, D] and [P, D], got "
            f"{tuple(epoch_pos.shape)} and {tuple(patch_pos.shape)}"
        )
    if epoch_pos.size(-1) != patch_pos.size(-1):
        raise ValueError(
            f"feature dims must match, got {epoch_pos.size(-1)} vs {patch_pos.size(-1)}"
        )
    combined = epoch_pos.unsqueeze(1) + patch_pos.unsqueeze(0)  # [L, P, D]
    return combined.reshape(-1, combined.size(-1))  # [L * P, D]


if __name__ == "__main__":
    # Smoke test: shapes, gradient flow, bf16, and the factorized-position contract.
    torch.manual_seed(0)
    B, L, Tp, D, P, d_model = 2, 5, 240, 256, 12, 128

    tok = PatchTokenizer(d_cnn=D, d_token=d_model, temporal_len=Tp, num_patches=P)
    feat = torch.randn(B * L, Tp, D, requires_grad=True)
    patches = tok(feat)
    assert patches.shape == (B * L, P, d_model), patches.shape

    patches.sum().backward()
    assert feat.grad is not None, "no gradient reached the tokenizer input"

    epoch_pos = torch.randn(L, d_model)
    patch_pos = torch.randn(P, d_model)
    pos = factorized_patch_positions(epoch_pos, patch_pos)
    assert pos.shape == (L * P, d_model), pos.shape
    # Same patch index, different epoch -> different total.
    assert not torch.allclose(pos[0 * P + 3], pos[1 * P + 3])
    # Same epoch, different patch index -> different total.
    assert not torch.allclose(pos[2 * P + 1], pos[2 * P + 4])

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        _ = tok(torch.randn(B * L, Tp, D))

    print(f"PatchTokenizer params: {sum(p.numel() for p in tok.parameters()):,}")
    print("patch_tokenization smoke test passed")
