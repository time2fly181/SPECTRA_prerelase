"""Attentive-statistics pooling over sub-epoch frames.

Speaker-verification systems (x-vector -> ECAPA-TDNN) converged on one recipe
for collapsing 10-30 s of variable-content signal into a single embedding:
attention weights computed **per channel** and conditioned on a global context
summary, used to take a weighted *mean and standard deviation* over time,
preceded by aggregation of feature maps from several depths. Compared with the
latent-query pool used by the existing epoch encoders this adds second-order
statistics (burstiness of EMG tone, intermittency of spindles), lets every
feature map pick its own diagnostic seconds of the epoch, and optionally keeps
the occupancy (duty-cycle) statistic from :mod:`spectra.models.occupancy_pooling`,
because several AASM rules *are* occupancy rules.

Two repository rules are honoured by construction:

* **Nothing built inside ``forward`` is cached on ``self``.** SupCon calls the
  encoder twice per step before one backward pass and CUDA-graph capture does
  not tolerate a tensor attribute written during replay. Diagnostics are
  exposed as explicit helper methods instead (:meth:`attention_weights`,
  :meth:`pooled_statistics`).
* **Variances are computed in fp32.** Under bf16 autocast the weighted second
  moment would otherwise lose the small ``E[x^2] - E[x]^2`` difference.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from .anti_alias import KaiserAntiAliasDownsample1D
from .common import _make_norm1d

__all__ = ["AttentiveStatisticsPool", "MultiLayerFeatureAggregation"]


def _activation(name: str) -> nn.Module:
    if name == "gelu":
        return nn.GELU(approximate="tanh")
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"activation must be 'gelu' or 'silu', got {name!r}")


class AttentiveStatisticsPool(nn.Module):
    """Pool ``[N, C, T]`` to ``[N, out_dim]`` with attentive mean/std statistics.

    Attention logits are produced by a small bottleneck network over the frame
    features concatenated with the global mean and standard deviation of the
    sequence (ECAPA's channel- and context-dependent statistics pooling); they
    are normalised over time separately for every head and channel. The pooled
    vector is ``[mean_h, std_h for each head] (+ occupancy)`` projected to
    ``out_dim``.

    Args:
        channels: Input channel count ``C``.
        out_dim: Width of the pooled vector. ``None`` keeps ``channels``.
        num_heads: Independent attention heads (each yields a mean and a std).
        bottleneck: Hidden width of the attention network.
        occupancy: Append the learned soft duty-cycle term.
        dropout: Dropout applied to the projected vector.
        occupancy_init_sharpness: Initial per-channel occupancy sigmoid slope.
        eps: Variance floor used by the standard-deviation terms.
    """

    def __init__(
        self,
        channels: int,
        out_dim: int | None = None,
        *,
        num_heads: int = 2,
        bottleneck: int = 128,
        occupancy: bool = True,
        dropout: float = 0.1,
        occupancy_init_sharpness: float = 1.0,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if not isinstance(channels, int) or channels < 1:
            raise ValueError(f"channels must be a positive integer, got {channels!r}")
        resolved_out = channels if out_dim is None else int(out_dim)
        if resolved_out < 1:
            raise ValueError(f"out_dim must be a positive integer, got {out_dim!r}")
        if not isinstance(num_heads, int) or num_heads < 1:
            raise ValueError(f"num_heads must be a positive integer, got {num_heads!r}")
        if not isinstance(bottleneck, int) or bottleneck < 1:
            raise ValueError(
                f"bottleneck must be a positive integer, got {bottleneck!r}"
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout!r}")
        if occupancy_init_sharpness == 0.0:
            raise ValueError(
                "occupancy_init_sharpness must be non-zero; sharpness.abs() has a "
                "zero subgradient at the origin"
            )
        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps!r}")

        self.channels = channels
        self.out_dim = resolved_out
        self.num_heads = num_heads
        self.bottleneck = bottleneck
        self.use_occupancy = bool(occupancy)
        self.eps = float(eps)

        # Channel- and context-dependent attention: frames + global mean/std.
        self.attention = nn.Sequential(
            nn.Conv1d(3 * channels, bottleneck, kernel_size=1),
            nn.BatchNorm1d(bottleneck),
            nn.Tanh(),
            nn.Conv1d(bottleneck, num_heads * channels, kernel_size=1),
        )
        if self.use_occupancy:
            self.tau = nn.Parameter(torch.zeros(1, channels, 1))
            self.sharpness = nn.Parameter(
                torch.full((1, channels, 1), float(occupancy_init_sharpness))
            )
        stats_width = 2 * num_heads * channels + (channels if self.use_occupancy else 0)
        self.project = nn.Linear(stats_width, self.out_dim)
        self.norm = nn.LayerNorm(self.out_dim)
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Re-initialise the attention network and output projection."""
        for module in self.attention:
            if isinstance(module, nn.Conv1d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.project.weight)
        if self.project.bias is not None:
            nn.init.zeros_(self.project.bias)

    # ----------------------------------------------------------- internals

    def _check(self, x: torch.Tensor) -> None:
        if x.ndim != 3 or x.size(1) != self.channels:
            raise ValueError(f"expected [N, {self.channels}, T], got {tuple(x.shape)}")

    def _logits(self, x: torch.Tensor) -> torch.Tensor:
        """Return attention logits ``[N, H, C, T]`` for a checked input."""
        n, c, t = x.shape
        x32 = x.float()
        mean = x32.mean(dim=-1, keepdim=True)
        std = x32.var(dim=-1, unbiased=False, keepdim=True).clamp_min(self.eps).sqrt()
        context = torch.cat(
            (x, mean.to(x.dtype).expand(-1, -1, t), std.to(x.dtype).expand(-1, -1, t)),
            dim=1,
        )
        logits = self.attention(context)  # [N, H*C, T]
        return logits.view(n, self.num_heads, c, t)

    def attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax attention weights ``[N, H, C, T]`` (no caching)."""
        self._check(x)
        logits = self._logits(x).float()
        return torch.softmax(logits.clamp(min=-50.0, max=50.0), dim=-1)

    def pooled_statistics(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return the fp32 pooled statistics before projection.

        Keys: ``mean`` and ``std`` are ``[N, H, C]``; ``occupancy`` (present only
        when enabled) is ``[N, C]``.
        """
        self._check(x)
        weights = self.attention_weights(x)  # [N, H, C, T] fp32
        x32 = x.float().unsqueeze(1)  # [N, 1, C, T]
        mean = (weights * x32).sum(dim=-1)  # [N, H, C]
        # Centred two-pass form: E_w[(x - mu)^2] does not suffer the
        # catastrophic cancellation of E_w[x^2] - mu^2 for near-constant input.
        centred = x32 - mean.unsqueeze(-1)
        std = (weights * centred.square()).sum(dim=-1).clamp_min(self.eps).sqrt()
        stats = {"mean": mean, "std": std}
        if self.use_occupancy:
            occ = torch.sigmoid((x32.squeeze(1) - self.tau) * self.sharpness.abs())
            stats["occupancy"] = occ.mean(dim=-1)
        return stats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stats = self.pooled_statistics(x)
        n = x.size(0)
        pieces = [stats["mean"].reshape(n, -1), stats["std"].reshape(n, -1)]
        if self.use_occupancy:
            pieces.append(stats["occupancy"])
        pooled = torch.cat(pieces, dim=1).to(self.project.weight.dtype)
        return self.drop(self.norm(self.project(pooled)))

    def get_attention_stats(self) -> None:
        """Return ``None``; attention weights are deliberately not retained."""
        return None

    def extra_repr(self) -> str:
        return (
            f"channels={self.channels}, out_dim={self.out_dim}, "
            f"num_heads={self.num_heads}, bottleneck={self.bottleneck}, "
            f"occupancy={self.use_occupancy}"
        )


class MultiLayerFeatureAggregation(nn.Module):
    """Concatenate feature maps from several depths and mix them with a 1x1 conv.

    Maps produced at a higher temporal rate than the last one are Kaiser
    anti-alias decimated to the last map's rate first. The decimation strides
    must be known at construction so the FIR kernels are fixed,
    non-persistent buffers like every other anti-alias filter in the repo.

    Args:
        in_channels: Channel count of each incoming map, shallow to deep.
        out_channels: Width of the aggregated map.
        strides: Integer decimation factor per map (``1`` for the deepest).
        activation: Activation after the 1x1 mix.
        norm: Normalisation family for the mix (``"bn"`` or ``"gn"``).
        fir_taps: Kaiser FIR length used for any decimation.
        fir_cutoff_ratio: Cutoff relative to the post-decimation Nyquist.
        fir_beta: Kaiser window beta.
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int,
        *,
        strides: Sequence[int],
        activation: str = "gelu",
        norm: str = "bn",
        fir_taps: int = 23,
        fir_cutoff_ratio: float = 0.85,
        fir_beta: float = 6.0,
    ) -> None:
        super().__init__()
        in_channels = tuple(int(c) for c in in_channels)
        strides = tuple(int(s) for s in strides)
        if not in_channels or any(c < 1 for c in in_channels):
            raise ValueError(
                f"in_channels must be positive integers, got {in_channels!r}"
            )
        if len(strides) != len(in_channels):
            raise ValueError(
                "strides must have one entry per input map, got "
                f"{strides!r} for {len(in_channels)} maps"
            )
        if any(s < 1 for s in strides) or strides[-1] != 1:
            raise ValueError(
                f"strides must be >= 1 and the deepest map must use 1, got {strides!r}"
            )
        if not isinstance(out_channels, int) or out_channels < 1:
            raise ValueError(
                f"out_channels must be a positive integer, got {out_channels!r}"
            )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.strides = strides
        self.reducers = nn.ModuleList(
            (
                KaiserAntiAliasDownsample1D(
                    channels=c,
                    cutoff_ratio=fir_cutoff_ratio,
                    num_taps=fir_taps,
                    beta=fir_beta,
                    stride=s,
                )
                if s > 1
                else nn.Identity()
            )
            for c, s in zip(in_channels, strides, strict=True)
        )
        self.mix = nn.Sequential(
            nn.Conv1d(sum(in_channels), out_channels, kernel_size=1, bias=False),
            _make_norm1d(norm, out_channels),
            _activation(activation),
        )

    def forward(self, maps: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(maps) != len(self.in_channels):
            raise ValueError(
                f"expected {len(self.in_channels)} feature maps, got {len(maps)}"
            )
        target_len = maps[-1].size(-1)
        aligned: list[torch.Tensor] = []
        for x, reducer, c in zip(maps, self.reducers, self.in_channels, strict=True):
            if x.ndim != 3 or x.size(1) != c:
                raise ValueError(f"expected [N, {c}, T], got {tuple(x.shape)}")
            y = reducer(x)
            if y.size(-1) != target_len:
                raise ValueError(
                    "feature map lengths disagree after decimation: "
                    f"{y.size(-1)} vs {target_len}"
                )
            aligned.append(y)
        return self.mix(torch.cat(aligned, dim=1))
