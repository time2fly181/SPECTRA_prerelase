"""Multirate EEG/EOG/EMG epoch encoder and recording-level envelope normalization.

The stem combines EEG short convolutions and rectified sinc-band envelopes,
lower-rate EOG convolutions, and rectified EMG convolution envelopes. Fused
features pass through a dilated trunk and configurable temporal pooling.
Rectification here uses magnitude, not squared power. Constructor arguments and
``get_config()`` define the checkpoint's filter, stride, width, and pooling
settings.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from .anti_alias import KaiserAntiAliasDownsample1D, KaiserAntiAliasUpsample1D
from .attentive_stats_pooling import (
    AttentiveStatisticsPool,
    MultiLayerFeatureAggregation,
)
from .blocks import FlexiblePhysiologicalStem
from .common import _make_norm1d
from .dilated_blocks import MultiDilatedBlock
from .learnable_pooling import LatentQueryAttentionPool
from .sinc_filterbank import ConstrainedSincFilterBank

__all__ = [
    "EEGTwoScaleBranch",
    "EMGEnvelopeBranch",
    "EOGLowRateBranch",
    "MultiRateAsymmetricEpochCNN",
    "MultiRateModalityStem",
]

_DEFAULT_WIDTHS: tuple[int, int, int, int, int] = (64, 128, 192, 256, 256)
_POOLING_MODES = ("attentive_stats", "learned")
EEG_BAND_CONSTRAINTS = ("legacy", "global", "allocated")
EEG_ALLOCATION_BANDS_HZ: tuple[tuple[float, float], ...] = (
    (0.5, 1.5),
    (1.5, 4.0),
    (4.0, 8.0),
    (8.0, 12.0),
    (12.0, 16.0),
    (16.0, 30.0),
    (30.0, 45.0),
)


def _linear_overlapping_passbands(
    count: int, low_hz: float, high_hz: float
) -> list[tuple[float, float]]:
    """Return ``count`` overlapping passbands confined to one named band."""
    if count < 0:
        raise ValueError(f"band allocation counts must be nonnegative, got {count}")
    if count == 0:
        return []
    anchors = torch.linspace(low_hz, high_hz, count + 2).tolist()
    return [(anchors[index], anchors[index + 2]) for index in range(count)]


def _allocated_eeg_passbands(
    allocation: Sequence[int],
    *,
    high_hz: float = 45.0,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Build initialized passbands and hard bounds for a seven-band allocation."""
    counts = tuple(int(value) for value in allocation)
    if len(counts) != len(EEG_ALLOCATION_BANDS_HZ):
        raise ValueError(
            "eeg_band_allocation must contain seven counts for "
            "SO,delta,theta,alpha,sigma,beta,30-highHz"
        )
    allocation_bands_hz = (*EEG_ALLOCATION_BANDS_HZ[:-1], (30.0, float(high_hz)))
    bands: list[tuple[float, float]] = []
    limits: list[tuple[float, float]] = []
    for count, (limit_low_hz, limit_high_hz) in zip(
        counts, allocation_bands_hz, strict=True
    ):
        bands.extend(_linear_overlapping_passbands(count, limit_low_hz, limit_high_hz))
        limits.extend([(limit_low_hz, limit_high_hz)] * count)
    return bands, limits


def _activation(name: str) -> nn.Module:
    if name == "gelu":
        return nn.GELU(approximate="tanh")
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"activation must be 'gelu' or 'silu', got {name!r}")


def _odd_int(value: Any, name: str) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 1
        or value % 2 == 0
    ):
        raise ValueError(f"{name} must be a positive odd integer, got {value!r}")
    return value


def _lowpass_ratio(cutoff_hz: float, fs: float, name: str) -> float:
    ratio = 2.0 * float(cutoff_hz) / float(fs)
    if not 0.0 < ratio <= 1.0:
        raise ValueError(
            f"{name}={cutoff_hz!r} Hz must lie in (0, fs/2={fs / 2:.1f}] Hz"
        )
    return ratio


def _autocast_dtype(x: torch.Tensor) -> torch.dtype:
    device_type = x.device.type
    if torch.is_autocast_enabled(device_type):
        return torch.get_autocast_dtype(device_type)
    return x.dtype


def _fit_length(y: torch.Tensor, length: int) -> torch.Tensor:
    current = y.size(-1)
    if current == length:
        return y
    if current > length:
        return y[..., :length]
    return F.pad(y, (0, length - current), mode="replicate")


BAND_NORM_STATISTICS: tuple[str, ...] = ("ema", "robust")
"""Statistic modes for :class:`PerRecordingBandNorm`.

``"ema"`` uses mean level and root-mean within-epoch variance. ``"robust"``
uses a low quantile of epoch levels and root-median within-epoch variance.
``ROBUST_LOCATION_QUANTILE`` defines the runtime's robust location quantile.
"""

ROBUST_LOCATION_QUANTILE = 0.10


def _as_reduction_input(x: torch.Tensor) -> torch.Tensor:
    """Move ``[N, C]`` per-epoch statistics to float64 on CPU.

    Reductions share one dtype and device. Inputs are detached, so recording
    statistics do not carry gradients.
    """
    if x.ndim != 2:
        raise ValueError(f"expected [N, C] per-epoch statistics, got {tuple(x.shape)}")
    return x.detach().to(device="cpu", dtype=torch.float64)


def ema_recording_statistics(
    per_epoch_mean: torch.Tensor, per_epoch_var: torch.Tensor, *, eps: float = 1e-5
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce per-epoch ``[N, C]`` statistics the way the EMA converges.

    Returns:
        ``(loc, scale)`` with ``loc = mean_N(level)`` and
        ``scale = sqrt(mean_N(within-epoch variance))``, both ``[C]`` float64.
    """
    mean = _as_reduction_input(per_epoch_mean)
    var = _as_reduction_input(per_epoch_var)
    return mean.mean(dim=0), var.mean(dim=0).clamp_min(eps).sqrt()


def robust_recording_statistics(
    per_epoch_mean: torch.Tensor,
    per_epoch_var: torch.Tensor,
    *,
    location_quantile: float = ROBUST_LOCATION_QUANTILE,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce per-epoch statistics using quantiles rather than arithmetic means.

    Args:
        per_epoch_mean: Envelope levels ``[N, C]`` for valid epochs.
        per_epoch_var: Within-epoch variances ``[N, C]`` for those same epochs.
        location_quantile: Quantile of epoch levels, using linear interpolation.
        eps: Variance floor before taking the square root.

    Returns:
        ``(loc, scale)`` with ``loc = quantile_N(level, q)`` and
        ``scale = sqrt(median_N(within-epoch variance))``, both ``[C]`` float64
        on CPU without gradients. Median uses linear quantile interpolation.
    """
    mean = _as_reduction_input(per_epoch_mean)
    var = _as_reduction_input(per_epoch_var)
    loc = torch.quantile(mean, float(location_quantile), dim=0, interpolation="linear")
    scale = (
        torch.quantile(var, 0.5, dim=0, interpolation="linear").clamp_min(eps).sqrt()
    )
    return loc, scale


def reduce_recording_statistics(
    statistic: str,
    per_epoch_mean: torch.Tensor,
    per_epoch_var: torch.Tensor,
    *,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dispatch to the reducer named by ``statistic`` (see BAND_NORM_STATISTICS)."""
    if statistic == "ema":
        return ema_recording_statistics(per_epoch_mean, per_epoch_var, eps=eps)
    if statistic == "robust":
        return robust_recording_statistics(per_epoch_mean, per_epoch_var, eps=eps)
    raise ValueError(
        f"unknown band-norm statistic {statistic!r}; expected one of {BAND_NORM_STATISTICS}"
    )


BAND_NORM_MODALITIES: tuple[str, ...] = ("eeg", "emg")


def parse_band_norm_modalities(spec: str) -> tuple[str, ...]:
    """Parse ``"eeg"`` / ``"eeg,emg"`` into a canonical, ordered tuple.

    Raises:
        ValueError: On an empty spec or a modality without an envelope path
            (EOG is a linear branch and has nothing to normalise).
    """
    parts = tuple(p.strip().lower() for p in str(spec).split(",") if p.strip())
    if not parts:
        raise ValueError("band_norm_modalities must name at least one modality")
    unknown = sorted(set(parts) - set(BAND_NORM_MODALITIES))
    if unknown:
        raise ValueError(
            f"unknown band-norm modalities {unknown}; expected a subset of "
            f"{BAND_NORM_MODALITIES}"
        )
    return tuple(m for m in BAND_NORM_MODALITIES if m in parts)


class PerRecordingBandNorm(nn.Module):
    """Normalize modality envelopes using saved or pinned recording statistics.

    Evaluation with pinned statistics applies ``(x - mean) / std`` followed by
    saved reference scale and location. The runtime computes those statistics
    from valid epochs of the current recording using ``reduce_recording_statistics``.
    Its ``ema`` reducer uses mean level and mean within-epoch variance; ``robust``
    uses a low quantile of epoch levels and median within-epoch variance.

    Without pinned statistics, recording indices select saved table rows. Cold
    EMA rows blend from identity according to ``seen_count``; robust rows are
    used only when populated. Missing indices, out-of-range indices, and disabled
    normalization pass through unchanged. Forward calls do not update the tables.

    Args:
        num_channels: Envelope feature channels ``C``.
        max_recordings: Rows in the statistics buffers.
        momentum: Saved EMA configuration retained for checkpoint compatibility.
        warmup_updates: Updates before a recording's statistics are trusted
            fully; below it the output blends from identity toward normalised.
        enabled: When False the module is an exact identity and never touches
            its buffers, so existing checkpoints reconstruct bit-for-bit.
        eps: Floor for variance and scale calculations.
        statistic: ``"ema"`` (streaming, default for checkpoint compatibility)
            or ``"robust"`` (saved robust statistics).
    """

    running_mean: torch.Tensor
    running_var: torch.Tensor
    seen_count: torch.Tensor
    ref_mean: torch.Tensor
    ref_std: torch.Tensor

    def __init__(
        self,
        num_channels: int,
        *,
        max_recordings: int = 20000,
        momentum: float = 0.1,
        warmup_updates: int = 20,
        enabled: bool = True,
        eps: float = 1e-5,
        statistic: str = "ema",
    ) -> None:
        super().__init__()
        if num_channels < 1:
            raise ValueError(f"num_channels must be >= 1, got {num_channels!r}")
        if statistic not in BAND_NORM_STATISTICS:
            raise ValueError(
                f"statistic must be one of {BAND_NORM_STATISTICS}, got {statistic!r}"
            )
        if max_recordings < 1:
            raise ValueError(f"max_recordings must be >= 1, got {max_recordings!r}")
        if not 0.0 < momentum <= 1.0:
            raise ValueError(f"momentum must be in (0, 1], got {momentum!r}")
        if warmup_updates < 1:
            raise ValueError(f"warmup_updates must be >= 1, got {warmup_updates!r}")
        self.num_channels = int(num_channels)
        self.max_recordings = int(max_recordings)
        self.momentum = float(momentum)
        self.warmup_updates = int(warmup_updates)
        self.enabled = bool(enabled)
        self.eps = float(eps)
        self.statistic = str(statistic)
        self._warned_unfilled = False
        self._table_meta: dict[str, Any] = {}
        self.register_buffer("running_mean", torch.zeros(max_recordings, num_channels))
        self.register_buffer("running_var", torch.ones(max_recordings, num_channels))
        self.register_buffer(
            "seen_count", torch.zeros(max_recordings, dtype=torch.long)
        )
        # Reference distribution the recordings are mapped onto. Defaults to the
        # standard normal; callers overwrite it with the downstream BatchNorm's
        # running statistics so the trunk keeps seeing its training scale.
        self.register_buffer("ref_mean", torch.zeros(num_channels))
        self.register_buffer("ref_std", torch.ones(num_channels))
        # Non-persistent: pinned per-recording statistics for eval. Never saved,
        # so a checkpoint never carries one recording's statistics into another.
        self._inference_mean: torch.Tensor | None = None
        self._inference_std: torch.Tensor | None = None
        # recording_id -> buffer row. Persisted via get/set_extra_state so a
        # later stage never trusts vocabulary ordering to line up.
        self._recording_ids: dict[str, int] = {}

    @torch.no_grad()
    def set_reference(self, *, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Set the target distribution recordings are mapped onto."""
        self.ref_mean.copy_(mean.detach().reshape(-1).to(self.ref_mean.dtype))
        self.ref_std.copy_(
            std.detach().reshape(-1).clamp_min(self.eps).to(self.ref_std.dtype)
        )

    @staticmethod
    def expand_recording_index(
        recording_index: torch.Tensor, *, batch: int, context_len: int
    ) -> torch.Tensor:
        """Expand ``[B]`` recording ids to ``[B * L]`` for flattened windows.

        Context inputs are ``[B, L, C, T]`` and reach the encoder as
        ``[B * L, C, T]``. Every window of one recording must carry that
        recording's id

        Raises:
            ValueError: If ``recording_index`` does not have ``batch`` entries.
        """
        flat = recording_index.reshape(-1)
        if flat.numel() != batch:
            raise ValueError(
                f"recording_index must have batch={batch} entries, got {flat.numel()}"
            )
        return flat.repeat_interleave(int(context_len))

    @staticmethod
    def per_sample_statistics(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return each epoch's ``[N, C]`` envelope mean and within-epoch variance.

        This is the statistic the module normalises with: the mean level of a
        recording and the *within-epoch* dispersion of its envelope over time
        (``x.var(dim=2)``), averaged over the recording's epochs. The
        between-epoch spread of the level -- a night's N2-vs-N3 contrast -- is
        deliberately not part of it. Training (:meth:`_update`) and inference
        (:func:`spectra.inference.runtime._pin_recording_band_statistics`)
        must both reduce through this so the trunk sees one distribution.
        """
        flat = x.detach()
        return flat.mean(dim=2), flat.var(dim=2, unbiased=False)

    @classmethod
    def recording_statistics(cls, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Reduce ``[N, C, T]`` envelopes of one recording to ``[C]`` mean/var.

        Equals what :meth:`_update` stores for a recording seen once in a
        single batch; whole-recording callers use it so pinned statistics are
        what the EMA would have converged to.
        """
        mean, var = cls.per_sample_statistics(x)
        return mean.mean(dim=0), var.mean(dim=0)

    def set_table_meta(self, meta: Mapping[str, Any]) -> None:
        """Record provenance of the loaded table (persisted in ``state_dict``)."""
        self._table_meta = dict(meta)

    @property
    def table_meta(self) -> dict[str, Any]:
        """Provenance of the loaded table, empty when none was loaded."""
        return dict(getattr(self, "_table_meta", {}))

    def register_recording_ids(self, mapping: Mapping[str, int]) -> None:
        """Record which ``recording_id`` owns which buffer row.

        The recording vocabulary is an in-process dict assigned in first-seen
        order and persisted nowhere, so a row index is only meaningful together
        with the run that produced it. Saving the mapping lets a later stage
        (temporal pretraining, a resume after the split changed) look statistics
        up by recording id.
        """
        self._recording_ids = {str(k): int(v) for k, v in mapping.items()}

    def get_extra_state(self) -> dict[str, Any]:
        """Persist the id -> row mapping, statistic mode and table provenance."""
        return {
            "recording_ids": dict(getattr(self, "_recording_ids", {})),
            "statistic": self.statistic,
            "table_meta": dict(getattr(self, "_table_meta", {})),
        }

    def set_extra_state(self, state: Any) -> None:
        """Restore the mapping and provenance; tolerate older payloads.

        A checkpoint whose ``statistic`` differs from this module's warns,
        because its buffers then mean something else (EMA mean/var vs a
        precomputed table) and must be replaced before use.
        """
        if not isinstance(state, Mapping):
            return
        ids = state.get("recording_ids")
        if isinstance(ids, Mapping):
            self._recording_ids = {str(k): int(v) for k, v in ids.items()}
        meta = state.get("table_meta")
        self._table_meta = dict(meta) if isinstance(meta, Mapping) else {}
        saved = state.get("statistic")
        if saved is not None and str(saved) != self.statistic:
            warnings.warn(
                f"checkpoint band-norm statistic {saved!r} differs from this module's "
                f"{self.statistic!r}; its per-recording buffers must be rebuilt before use",
                UserWarning,
                stacklevel=2,
            )

    @torch.no_grad()
    def set_inference_statistics(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Pin whole-recording statistics for eval, mirroring training.

        A model trained with this module expects normalised envelopes, so at
        inference the *same* correction must be applied or it sees a
        distribution it never learned on. Training used an EMA keyed by
        recording; inference has the whole recording at once, so the caller
        computes its statistics directly and pins them here for the duration of
        that recording. Recordings unseen in training (every OOD store) are
        handled naturally because nothing is looked up.

        Args:
            mean: ``[C]`` per-band mean of this recording's log band envelope.
            std: ``[C]`` per-band standard deviation.
        """
        self._inference_mean = mean.detach().reshape(-1).clone()
        self._inference_std = std.detach().reshape(-1).clamp_min(self.eps).clone()

    def clear_inference_statistics(self) -> None:
        """Drop pinned statistics so the module reverts to buffer lookup."""
        self._inference_mean = None
        self._inference_std = None

    def forward(
        self, x: torch.Tensor, recording_index: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Normalise ``[N, C, T]`` band envelopes by their recording statistics.

        ``recording_index`` of ``None`` (probes, diagnostics, any caller without
        recording provenance) is an exact identity, unless whole-recording
        statistics have been pinned by :meth:`set_inference_statistics`.
        """
        if not self.enabled:
            return x
        pinned_mean = getattr(self, "_inference_mean", None)
        pinned_std = getattr(self, "_inference_std", None)
        if not self.training and pinned_mean is not None and pinned_std is not None:
            m = pinned_mean.to(x.device, x.dtype).view(1, -1, 1)
            sd = pinned_std.to(x.device, x.dtype).view(1, -1, 1)
            ref_mean = self.ref_mean.to(x.device, x.dtype).view(1, -1, 1)
            ref_std = self.ref_std.to(x.device, x.dtype).view(1, -1, 1)
            return (x - m) / sd * ref_std + ref_mean
        if recording_index is None:
            return x
        index = recording_index.reshape(-1).to(x.device, torch.long)
        if index.numel() != x.shape[0]:
            raise ValueError(
                f"recording_index must have {x.shape[0]} entries to match the "
                f"batch, got {index.numel()}"
            )
        valid = (index >= 0) & (index < self.max_recordings)
        if not bool(valid.any()):
            return x

        safe = torch.where(valid, index, torch.zeros_like(index))
        mean = self.running_mean.index_select(0, safe).unsqueeze(-1)
        var = self.running_var.index_select(0, safe).unsqueeze(-1)
        count = self.seen_count.index_select(0, safe)
        ref_mean = self.ref_mean.view(1, -1, 1)
        ref_std = self.ref_std.view(1, -1, 1)

        normed = (x - mean.to(x.dtype)) / (
            var.to(x.dtype).clamp_min(self.eps).sqrt()
        ) * ref_std.to(x.dtype) + ref_mean.to(x.dtype)

        # Blend from identity while a recording's statistics are still cold, and
        # leave out-of-range rows untouched.
        if self.statistic == "robust":
            # Precomputed rows are fully trusted; unfilled rows pass through.
            w = (count > 0).to(x.dtype)
            if not self._warned_unfilled and bool((valid & (count == 0)).any()):
                self._warned_unfilled = True
                warnings.warn(
                    "PerRecordingBandNorm: some recording rows have no precomputed "
                    "statistics and pass through unnormalised -- was the band-norm "
                    "table applied for every store of this run?",
                    UserWarning,
                    stacklevel=2,
                )
        else:
            w = (count.to(x.dtype) / float(self.warmup_updates)).clamp(0.0, 1.0)
        w = torch.where(valid, w, torch.zeros_like(w)).view(-1, 1, 1)
        return torch.lerp(x, normed, w)

    def extra_repr(self) -> str:
        return (
            f"num_channels={self.num_channels}, max_recordings={self.max_recordings}, "
            f"momentum={self.momentum}, warmup_updates={self.warmup_updates}, "
            f"enabled={self.enabled}, statistic={self.statistic!r}"
        )


class EMGEnvelopeBranch(nn.Module):
    """Full-bandwidth short-kernel filters -> rectify -> low-pass -> log envelope.

    The first convolution runs at the input rate. Its absolute magnitude is
    smoothed with a fixed Kaiser low-pass (``lowpass_hz``), compressed with
    ``log1p``, and decimated by two.

    Output: ``[N, out_ch, ceil(T/2)]``.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        num_filters: int,
        kernel_size: int,
        lowpass_hz: float,
        fs: float,
        fir_taps: int,
        fir_cutoff_ratio: float,
        fir_beta: float,
        norm: str,
        activation: str,
        recording_norm: PerRecordingBandNorm | None = None,
    ) -> None:
        super().__init__()
        if num_filters < 1:
            raise ValueError(f"emg_filters must be >= 1, got {num_filters!r}")
        kernel_size = _odd_int(kernel_size, "emg_kernel")
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.num_filters = num_filters
        self.filters = nn.Conv1d(
            in_ch, num_filters, kernel_size, padding=kernel_size // 2, bias=False
        )
        self.smooth = KaiserAntiAliasDownsample1D(
            channels=num_filters,
            cutoff_ratio=_lowpass_ratio(lowpass_hz, fs, "emg_lowpass_hz"),
            num_taps=fir_taps,
            beta=fir_beta,
            stride=1,
        )
        self.env_norm = _make_norm1d(norm, num_filters)
        # Per-recording correction sits BEFORE env_norm, mirroring the EEG band
        # path: chin EMG is high-passed at some sites and not others, which
        # moves the absolute envelope level by an order of magnitude.
        if recording_norm is not None and recording_norm.num_channels != num_filters:
            raise ValueError(
                "recording_norm.num_channels must equal emg_filters "
                f"({recording_norm.num_channels} != {num_filters})"
            )
        self.env_recording_norm: PerRecordingBandNorm | None = recording_norm
        self.env_act = _activation(activation)
        self.project = nn.Sequential(
            nn.Conv1d(num_filters, out_ch, kernel_size=1, bias=False),
            _make_norm1d(norm, out_ch),
            _activation(activation),
        )
        self.downsample = KaiserAntiAliasDownsample1D(
            channels=out_ch,
            cutoff_ratio=fir_cutoff_ratio,
            num_taps=fir_taps,
            beta=fir_beta,
            stride=2,
        )

    def envelope(self, x: torch.Tensor) -> torch.Tensor:
        """Return the log-compressed smoothed magnitude ``[N, F, T]`` (input rate)."""
        rectified = self.filters(x).abs()
        return torch.log1p(self.smooth(rectified).clamp_min(0.0))

    @torch.no_grad()
    def sync_env_norm_reference(self) -> bool:
        """Point the per-recording norm at ``env_norm``'s training statistics."""
        norm = self.env_recording_norm
        mean = getattr(self.env_norm, "running_mean", None)
        var = getattr(self.env_norm, "running_var", None)
        if norm is None or mean is None or var is None:
            return False
        norm.set_reference(mean=mean, std=var.clamp_min(0.0).sqrt())
        return True

    def forward(
        self, x: torch.Tensor, recording_index: torch.Tensor | None = None
    ) -> torch.Tensor:
        envelope = self.envelope(x)
        if self.env_recording_norm is not None:
            envelope = self.env_recording_norm(envelope, recording_index)
        y = self.env_act(self.env_norm(envelope))
        return self.downsample(self.project(y))


class EOGLowRateBranch(nn.Module):
    """Decimate first, then long-receptive-field convolutions at a low rate.

    The branch Kaiser-decimates by ``decimation``, applies parallel dilated
    convolutions at ``fs / decimation``, then upsamples by ``decimation / 2``
    with an anti-imaging filter to match the stem rate. Half of the dilation-one
    filters are initialized odd-symmetric. Kernel lengths, dilation values, and
    decimation are constructor settings.

    Output: ``[N, out_ch, ceil(T/2)]``.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        decimation: int,
        kernel_size: int,
        dilations: Sequence[int],
        fs: float,
        fir_taps: int,
        fir_cutoff_ratio: float,
        fir_beta: float,
        norm: str,
        activation: str,
    ) -> None:
        super().__init__()
        if (
            not isinstance(decimation, int)
            or decimation < 2
            or decimation & (decimation - 1)
        ):
            raise ValueError(
                f"eog_decimation must be a power of two >= 2, got {decimation!r}"
            )
        kernel_size = _odd_int(kernel_size, "eog_kernel")
        dilations = tuple(int(d) for d in dilations)
        if not dilations or any(d < 1 for d in dilations):
            raise ValueError(
                f"eog_dilations must be positive integers, got {dilations!r}"
            )
        del fs  # recorded by the owner; the branch is rate-agnostic by design
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.decimation = decimation
        self.dilations = dilations
        self.kernel_size = kernel_size
        # A long FIR for a deep decimation: 12 taps per unit of stride, odd.
        decimate_taps = max(fir_taps, 12 * decimation - 1)
        self.decimate = KaiserAntiAliasDownsample1D(
            channels=in_ch,
            cutoff_ratio=fir_cutoff_ratio,
            num_taps=decimate_taps,
            beta=fir_beta,
            stride=decimation,
        )
        self.dilated = nn.ModuleList(
            nn.Conv1d(
                in_ch,
                out_ch,
                kernel_size,
                dilation=d,
                padding=d * (kernel_size - 1) // 2,
                bias=False,
            )
            for d in dilations
        )
        hidden = out_ch * len(dilations)
        self.mix = nn.Sequential(
            _make_norm1d(norm, hidden),
            _activation(activation),
            nn.Conv1d(
                hidden, out_ch, kernel_size, padding=kernel_size // 2, bias=False
            ),
            _make_norm1d(norm, out_ch),
            _activation(activation),
        )
        self.upsample = KaiserAntiAliasUpsample1D(
            channels=out_ch,
            cutoff_ratio=fir_cutoff_ratio,
            num_taps=fir_taps,
            beta=fir_beta,
            scale_factor=decimation // 2,
        )
        self._init_odd_symmetric()

    @torch.no_grad()
    def _init_odd_symmetric(self) -> None:
        first = cast(nn.Conv1d, self.dilated[0])  # dilation-1 filters
        weight = first.weight
        half = weight.size(0) // 2
        if half == 0:
            return
        w = weight[:half]
        weight[:half] = (w - w.flip(-1)) / math.sqrt(2.0)

    @property
    def receptive_field_samples(self) -> int:
        """Receptive field at the decimated rate, in samples."""
        k = self.kernel_size
        return max(self.dilations) * (k - 1) + 1 + (k - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_time = (x.size(-1) + 1) // 2
        y = self.decimate(x)
        y = torch.cat([conv(y) for conv in self.dilated], dim=1)
        y = self.mix(y)
        y = self.upsample(y)
        return _fit_length(y, out_time)


class EEGTwoScaleBranch(nn.Module):
    """Short-kernel multi-dilation stem ++ long-kernel sinc band-power path.

    The short path uses :class:`FlexiblePhysiologicalStem`. The band path applies
    ``band_filters`` constrained sinc filters with ``band_kernel`` taps to each
    EEG channel independently, takes absolute magnitude, smooths at
    ``band_lowpass_hz``, applies ``log1p``, and decimates by two. Band limits,
    kernel lengths, and dilations are configurable. Both paths are concatenated
    to ``out_ch``; the rectified envelope is not squared power.

    Output: ``[N, out_ch, ceil(T/2)]``.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        band_fraction: float,
        band_filters: int,
        band_kernel: int,
        band_low_hz: float,
        band_high_hz: float,
        band_constraint: str,
        band_allocation: Sequence[int] | None,
        band_minimum_width_hz: float,
        band_lowpass_hz: float,
        short_kernel: int,
        short_dilations: Sequence[int],
        fs: float,
        fir_taps: int,
        fir_cutoff_ratio: float,
        fir_beta: float,
        norm: str,
        activation: str,
        band_norm_per_recording: bool = False,
        band_norm_momentum: float = 0.1,
        band_norm_warmup_updates: int = 20,
        band_norm_max_recordings: int = 20000,
        band_norm_statistic: str = "ema",
    ) -> None:
        super().__init__()
        if not 0.0 <= band_fraction < 1.0:
            raise ValueError(
                f"eeg_band_fraction must be in [0, 1), got {band_fraction!r}"
            )
        if band_filters < 1:
            raise ValueError(f"eeg_band_filters must be >= 1, got {band_filters!r}")
        band_kernel = _odd_int(band_kernel, "eeg_band_kernel")
        short_kernel = _odd_int(short_kernel, "stem_kernel_size")
        if not 0.0 < band_low_hz < band_high_hz <= fs / 2.0:
            raise ValueError(
                "eeg_band_low_hz/eeg_band_high_hz must satisfy "
                f"0 < low < high <= fs/2, got ({band_low_hz!r}, {band_high_hz!r})"
            )
        if band_constraint not in EEG_BAND_CONSTRAINTS:
            raise ValueError(
                f"eeg_band_constraint must be one of {EEG_BAND_CONSTRAINTS}, "
                f"got {band_constraint!r}"
            )
        initial_bands = None
        band_limits = None
        if band_constraint == "allocated":
            if band_allocation is None:
                raise ValueError(
                    "eeg_band_allocation is required when eeg_band_constraint='allocated'"
                )
            if not math.isclose(band_low_hz, 0.5, rel_tol=0.0, abs_tol=1e-6):
                raise ValueError("allocated EEG bands require eeg_band_low_hz=0.5")
            if not 30.0 + band_minimum_width_hz < band_high_hz <= fs / 2.0:
                raise ValueError(
                    "allocated eeg_band_high_hz must be greater than "
                    f"{30.0 + band_minimum_width_hz:g} and no greater than fs/2"
                )
            initial_bands, band_limits = _allocated_eeg_passbands(
                band_allocation,
                high_hz=band_high_hz,
            )
            if len(initial_bands) != band_filters:
                raise ValueError(
                    f"eeg_band_allocation sums to {len(initial_bands)}, "
                    f"but eeg_band_filters={band_filters}"
                )
        elif band_constraint == "global":
            band_limits = [(band_low_hz, band_high_hz)] * band_filters
        band_out = int(round(band_fraction * out_ch))
        short_out = out_ch - band_out
        if short_out < 1:
            raise ValueError(
                f"eeg_band_fraction={band_fraction!r} leaves no width for the short path"
            )
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.short_out_ch = short_out
        self.band_out_ch = band_out
        self.band_kernel_size = band_kernel
        self.num_band_filters = band_filters
        self.short = FlexiblePhysiologicalStem(
            in_ch=in_ch,
            out_ch=short_out,
            kernel_size=short_kernel,
            dilations=tuple(int(d) for d in short_dilations),
            fs=int(fs),
            norm=norm,
            activation=activation,
            dropout=0.0,
            se_reduction=4,
            return_weights=False,
            aa_cutoff_ratio=fir_cutoff_ratio,
            aa_num_taps=fir_taps,
            aa_beta=fir_beta,
        )
        if band_out > 0:
            self.band_filters: ConstrainedSincFilterBank | None = (
                ConstrainedSincFilterBank(
                    band_filters,
                    fs=int(fs),
                    kernel_size=band_kernel,
                    low_hz=band_low_hz,
                    high_hz=band_high_hz,
                    band_limits_hz=band_limits,
                    initial_bands_hz=initial_bands,
                    minimum_bandwidth_hz=band_minimum_width_hz,
                )
            )
            band_ch = in_ch * band_filters
            self.band_smooth = KaiserAntiAliasDownsample1D(
                channels=band_ch,
                cutoff_ratio=_lowpass_ratio(band_lowpass_hz, fs, "eeg_band_lowpass_hz"),
                num_taps=fir_taps,
                beta=fir_beta,
                stride=1,
            )
            self.band_norm = _make_norm1d(norm, band_ch)
            # Per-recording correction sits BEFORE band_norm so the trunk keeps
            # seeing band_norm's training-time scale (see PerRecordingBandNorm).
            self.band_recording_norm: PerRecordingBandNorm | None = (
                PerRecordingBandNorm(
                    band_ch,
                    max_recordings=band_norm_max_recordings,
                    momentum=band_norm_momentum,
                    warmup_updates=band_norm_warmup_updates,
                    statistic=band_norm_statistic,
                )
                if band_norm_per_recording
                else None
            )
            self.band_act = _activation(activation)
            self.band_project = nn.Sequential(
                nn.Conv1d(band_ch, band_out, kernel_size=1, bias=False),
                _make_norm1d(norm, band_out),
                _activation(activation),
            )
            self.band_down = KaiserAntiAliasDownsample1D(
                channels=band_out,
                cutoff_ratio=fir_cutoff_ratio,
                num_taps=fir_taps,
                beta=fir_beta,
                stride=2,
            )
        else:
            self.band_filters = None
            self.band_recording_norm = None

    def band_edges_hz(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the current (low, high) passband edges of the sinc bank in Hz."""
        if self.band_filters is None:
            raise ValueError("this branch has no band path (eeg_band_fraction=0)")
        bank = self.band_filters
        low, high = bank.band_edges_hz()
        return low.detach(), high.detach()

    def band_envelope(self, x: torch.Tensor) -> torch.Tensor:
        """Return the log band-power map ``[N, C*K, T]`` at the input rate."""
        if self.band_filters is None:
            raise ValueError("this branch has no band path (eeg_band_fraction=0)")
        n, c, t = x.shape
        filtered = self.band_filters(x.reshape(n * c, 1, t)).reshape(n, -1, t)
        smoothed = self.band_smooth(filtered.abs()).clamp_min(0.0)
        return torch.log1p(smoothed)

    @torch.no_grad()
    def sync_band_norm_reference(self) -> bool:
        """Point the per-recording norm at ``band_norm``'s training statistics.
        Returns:
            True when a reference was copied.
        """
        norm = getattr(self, "band_recording_norm", None)
        bn = getattr(self, "band_norm", None)
        if norm is None or bn is None:
            return False
        mean = getattr(bn, "running_mean", None)
        var = getattr(bn, "running_var", None)
        if mean is None or var is None:
            return False
        norm.set_reference(mean=mean, std=var.clamp_min(0.0).sqrt())
        return True

    def forward(
        self, x: torch.Tensor, recording_index: torch.Tensor | None = None
    ) -> torch.Tensor:
        short = cast(torch.Tensor, self.short(x))
        if self.band_filters is None:
            return short
        envelope = self.band_envelope(x)
        if self.band_recording_norm is not None:
            envelope = self.band_recording_norm(envelope, recording_index)
        band = self.band_act(self.band_norm(envelope))
        band = self.band_down(self.band_project(band))
        return torch.cat((short, band), dim=1)


class MultiRateModalityStem(nn.Module):
    """Route channels to modality branches and fuse at half the input rate.

    Absent channels are zeroed. A branch runs only on rows with that modality
    present, so zero placeholders never update its BatchNorm statistics.
    A row with no modality at all raises.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        eeg_indices: Sequence[int],
        eog_indices: Sequence[int],
        emg_indices: Sequence[int],
        modality_split: Sequence[float],
        dropout: float,
        norm: str,
        activation: str,
        fs: float,
        fir_taps: int,
        fir_cutoff_ratio: float,
        fir_beta: float,
        stem_kernel_size: int,
        eeg_short_dilations: Sequence[int],
        eeg_band_filters: int,
        eeg_band_kernel: int,
        eeg_band_fraction: float,
        eeg_band_low_hz: float,
        eeg_band_high_hz: float,
        eeg_band_constraint: str,
        eeg_band_allocation: Sequence[int] | None,
        eeg_band_minimum_width_hz: float,
        eeg_band_lowpass_hz: float,
        band_norm_per_recording: bool = False,
        band_norm_momentum: float = 0.1,
        band_norm_warmup_updates: int = 20,
        band_norm_max_recordings: int = 20000,
        band_norm_statistic: str = "ema",
        band_norm_modalities: str = "eeg",
        eog_decimation: int,
        eog_kernel: int,
        eog_dilations: Sequence[int],
        emg_filters: int,
        emg_kernel: int,
        emg_lowpass_hz: float,
    ) -> None:
        super().__init__()
        if not isinstance(in_ch, int) or in_ch < 1:
            raise ValueError(f"in_ch must be a positive integer, got {in_ch!r}")
        if not isinstance(out_ch, int) or out_ch < 1:
            raise ValueError(f"out_ch must be a positive integer, got {out_ch!r}")
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.eeg_indices = [int(i) for i in eeg_indices]
        self.eog_indices = [int(i) for i in eog_indices]
        self.emg_indices = [int(i) for i in emg_indices]
        all_indices = self.eeg_indices + self.eog_indices + self.emg_indices
        if any(i < 0 or i >= in_ch for i in all_indices):
            raise ValueError(
                f"all modality channel indices must lie in [0, {in_ch}), got {all_indices!r}"
            )
        if len(set(all_indices)) != len(all_indices):
            raise ValueError("modality channel indices must not overlap")
        present = {
            "eeg": bool(self.eeg_indices),
            "eog": bool(self.eog_indices),
            "emg": bool(self.emg_indices),
        }
        if not any(present.values()):
            raise ValueError("At least one modality must have channels")
        split = tuple(float(v) for v in modality_split)
        if len(split) != 3 or any(v < 0.0 for v in split):
            raise ValueError(
                f"modality_split must be three nonnegative ratios, got {modality_split!r}"
            )
        weights = {
            name: (value if present[name] else 0.0)
            for name, value in zip(("eeg", "eog", "emg"), split, strict=True)
        }
        total = sum(weights.values())
        if total <= 0.0:
            raise ValueError("configured modalities must have a positive total ratio")
        names = [name for name in ("eeg", "eog", "emg") if present[name]]
        quotas = {name: weights[name] / total * out_ch for name in names}
        allocation = {name: int(quotas[name]) for name in names}
        leftover = out_ch - sum(allocation.values())
        order = sorted(names, key=lambda name: (-(quotas[name] % 1), names.index(name)))
        for name in order[:leftover]:
            allocation[name] += 1
        self.eeg_out_ch = allocation.get("eeg", 0)
        self.eog_out_ch = allocation.get("eog", 0)
        self.emg_out_ch = allocation.get("emg", 0)

        norm_modalities = parse_band_norm_modalities(band_norm_modalities)
        common: dict[str, Any] = dict(
            fs=fs,
            fir_taps=fir_taps,
            fir_cutoff_ratio=fir_cutoff_ratio,
            fir_beta=fir_beta,
            norm=norm,
            activation=activation,
        )
        self.eeg_branch: EEGTwoScaleBranch | None = (
            EEGTwoScaleBranch(
                len(self.eeg_indices),
                self.eeg_out_ch,
                band_fraction=eeg_band_fraction,
                band_filters=eeg_band_filters,
                band_kernel=eeg_band_kernel,
                band_low_hz=eeg_band_low_hz,
                band_high_hz=eeg_band_high_hz,
                band_constraint=eeg_band_constraint,
                band_allocation=eeg_band_allocation,
                band_minimum_width_hz=eeg_band_minimum_width_hz,
                band_lowpass_hz=eeg_band_lowpass_hz,
                band_norm_per_recording=(
                    band_norm_per_recording and "eeg" in norm_modalities
                ),
                band_norm_momentum=band_norm_momentum,
                band_norm_warmup_updates=band_norm_warmup_updates,
                band_norm_max_recordings=band_norm_max_recordings,
                band_norm_statistic=band_norm_statistic,
                short_kernel=stem_kernel_size,
                short_dilations=eeg_short_dilations,
                **common,
            )
            if self.eeg_out_ch > 0
            else None
        )
        self.eog_branch: EOGLowRateBranch | None = (
            EOGLowRateBranch(
                len(self.eog_indices),
                self.eog_out_ch,
                decimation=eog_decimation,
                kernel_size=eog_kernel,
                dilations=eog_dilations,
                **common,
            )
            if self.eog_out_ch > 0
            else None
        )
        self.emg_branch: EMGEnvelopeBranch | None = (
            EMGEnvelopeBranch(
                len(self.emg_indices),
                self.emg_out_ch,
                num_filters=emg_filters,
                kernel_size=emg_kernel,
                lowpass_hz=emg_lowpass_hz,
                recording_norm=(
                    PerRecordingBandNorm(
                        emg_filters,
                        max_recordings=band_norm_max_recordings,
                        momentum=band_norm_momentum,
                        warmup_updates=band_norm_warmup_updates,
                        statistic=band_norm_statistic,
                    )
                    if band_norm_per_recording and "emg" in norm_modalities
                    else None
                ),
                **common,
            )
            if self.emg_out_ch > 0
            else None
        )
        self.fusion = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, kernel_size=1),
            _make_norm1d(norm, out_ch),
            _activation(activation),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def _derive_modality_mask(
        self,
        batch_size: int,
        channel_mask: torch.Tensor | None,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        configured = torch.tensor(
            [
                self.eeg_branch is not None,
                self.eog_branch is not None,
                self.emg_branch is not None,
            ],
            dtype=torch.bool,
            device=device,
        )
        if channel_mask is None:
            return configured.unsqueeze(0).expand(batch_size, -1)
        mask = channel_mask.to(device=device)
        modality_mask = torch.zeros(batch_size, 3, dtype=torch.bool, device=device)
        if self.eeg_branch is not None:
            modality_mask[:, 0] = mask[:, self.eeg_indices].gt(0).any(dim=1)
        if self.eog_branch is not None:
            modality_mask[:, 1] = mask[:, self.eog_indices].gt(0).any(dim=1)
        if self.emg_branch is not None:
            modality_mask[:, 2] = mask[:, self.emg_indices].gt(0).any(dim=1)
        return modality_mask

    @torch.compiler.disable()
    def _run_branch(
        self,
        branch: nn.Module,
        x_modality: torch.Tensor,
        *,
        out_ch: int,
        sample_mask: torch.Tensor | None,
        recording_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run ``branch`` only on the rows whose modality is present.

        ``recording_index`` is subset in lockstep with ``sample_mask`` so a
        branch never receives statistics belonging to a different recording.
        """

        def _call(inp: torch.Tensor, idx: torch.Tensor | None) -> torch.Tensor:
            return branch(inp) if idx is None else branch(inp, idx)

        batch_size, _, time_len = x_modality.shape
        out_time = (time_len + 1) // 2
        if sample_mask is None:
            return _call(x_modality, recording_index)
        present = sample_mask.to(device=x_modality.device, dtype=torch.bool)
        if bool(present.all()):
            return _call(x_modality, recording_index)
        if bool(present.any()):
            present_out = _call(
                x_modality[present],
                None if recording_index is None else recording_index[present],
            )
            out = present_out.new_zeros(batch_size, out_ch, out_time)
            out[present] = present_out
            return out
        return x_modality.new_zeros(
            batch_size, out_ch, out_time, dtype=_autocast_dtype(x_modality)
        )

    def forward(
        self,
        x: torch.Tensor,
        channel_mask: torch.Tensor | None = None,
        recording_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 3 or x.size(1) != self.in_ch:
            raise ValueError(f"expected [B, {self.in_ch}, T], got {tuple(x.shape)}")
        if channel_mask is not None:
            if channel_mask.ndim == 1:
                channel_mask = channel_mask.unsqueeze(0).expand(x.size(0), -1)
            if channel_mask.shape != x.shape[:2]:
                raise ValueError(
                    f"channel_mask must have shape {tuple(x.shape[:2])}, "
                    f"got {tuple(channel_mask.shape)}"
                )
            channel_mask = channel_mask.to(device=x.device, dtype=x.dtype)
            x = x * channel_mask.unsqueeze(-1)
        modality_mask = self._derive_modality_mask(
            x.size(0), channel_mask, device=x.device
        )
        no_present = ~modality_mask.any(dim=1)
        if bool(no_present.any()):
            indices = no_present.nonzero(as_tuple=False).flatten().tolist()
            raise ValueError(
                "Every sample must contain at least one available modality. "
                f"All channels were missing for batch indices {indices}."
            )
        # Each ``_run_branch`` call is ``torch.compiler.disable``d, so Dynamo
        # breaks the graph here. Bind the results to plain locals rather than
        # appending to a list: a list built across those breaks stays live on
        # the interpreter stack, and every call invalidates the resume frame's
        # cache line ("L['___stack0'] got deallocated"), exhausting
        # ``recompile_limit`` after eight calls and dropping the rest of
        # ``forward`` to eager. Tensors in named locals guard cleanly.
        eeg_out: torch.Tensor | None = None
        eog_out: torch.Tensor | None = None
        emg_out: torch.Tensor | None = None
        if self.eeg_branch is not None:
            eeg_out = self._run_branch(
                self.eeg_branch,
                x[:, self.eeg_indices, :],
                out_ch=self.eeg_out_ch,
                sample_mask=modality_mask[:, 0],
                recording_index=recording_index,
            )
        if self.eog_branch is not None:
            eog_out = self._run_branch(
                self.eog_branch,
                x[:, self.eog_indices, :],
                out_ch=self.eog_out_ch,
                sample_mask=modality_mask[:, 1],
            )
        if self.emg_branch is not None:
            emg_out = self._run_branch(
                self.emg_branch,
                x[:, self.emg_indices, :],
                out_ch=self.emg_out_ch,
                sample_mask=modality_mask[:, 2],
                recording_index=recording_index,
            )
        outputs = [t for t in (eeg_out, eog_out, emg_out) if t is not None]
        return self.fusion(torch.cat(outputs, dim=1))

    def get_architecture_info(self) -> dict[str, Any]:
        """Return stem geometry for logging."""
        return {
            "stem_type": "MultiRateModalityStem",
            "eeg_channels": len(self.eeg_indices),
            "eog_channels": len(self.eog_indices),
            "emg_channels": len(self.emg_indices),
            "eeg_out_ch": self.eeg_out_ch,
            "eog_out_ch": self.eog_out_ch,
            "emg_out_ch": self.emg_out_ch,
            "eeg_short_out_ch": (
                self.eeg_branch.short_out_ch if self.eeg_branch is not None else 0
            ),
            "eeg_band_out_ch": (
                self.eeg_branch.band_out_ch if self.eeg_branch is not None else 0
            ),
        }


class MultiRateAsymmetricEpochCNN(nn.Module):
    """Encode normalized epochs with modality branches, a trunk, and pooling.

    Inputs are ``[N, in_ch, time_len]`` floating tensors; output is
    ``[N, out_dim]`` on the input device. Model and input must share a device.
    Saved constructor settings determine the architecture, including the
    normalization modalities and filter constraints. Use ``get_config()`` for
    the complete constructor configuration.

    Args:
        in_ch: Number of input channels.
        time_len: Samples per epoch (3840 at 128 Hz).
        widths: Five widths ``(stem, stage1, stage2, stage3, stage4)``; ``None``
            selects ``_DEFAULT_WIDTHS``. Slot 4
            is the pre-pool width and, unless ``pool_out_dim`` is set, the
            encoder output width.
        dropout: Dropout in the stem fusion and trunk blocks.
        norm: Normalization family. Only ``"bn"`` is supported.
        fs: Sampling rate in Hz (used numerically by the filter designs).
        activation: Activation name used throughout.
        eeg_indices / eog_indices / emg_indices: Channel slots per modality.
        modality_split: Stem width ratios ``(eeg, eog, emg)``; renormalised over
            the modalities that have channels.
        stem_kernel_size: Kernel of the EEG short path.
        eeg_short_dilations: Dilations of the EEG short path.
        eeg_band_filters / eeg_band_kernel / eeg_band_fraction / eeg_band_low_hz /
            eeg_band_high_hz / eeg_band_lowpass_hz: EEG band-power path.
        eog_decimation / eog_kernel / eog_dilations: EOG low-rate path.
        emg_filters / emg_kernel / emg_lowpass_hz: EMG envelope path.
        fir_taps / fir_cutoff_ratio / fir_beta: Kaiser design shared by the
            stem filters (cutoff relative to the post-decimation Nyquist).
        trunk_kernel_size / trunk_strides / trunk_dilations / trunk_fir_taps:
            The flexible trunk schedule. Total downsampling is ``2 * prod(strides)``.
        pooling_mode: ``"attentive_stats"`` or ``"learned"`` latent-query pooling.
        pool_out_dim: Pooled width; ``None`` keeps ``widths[4]``.
        pool_heads / pool_bottleneck / pool_occupancy: Attentive-stats options.
        pool_mfa: Aggregate stage-3 and stage-4 maps before pooling.
        pool_dropout: Dropout on the pooled vector; ``None`` uses ``dropout``.
    """

    def __init__(
        self,
        in_ch: int = 5,
        time_len: int = 3840,
        widths: Sequence[int] | None = None,
        dropout: float = 0.1,
        norm: str = "bn",
        fs: float = 128.0,
        activation: str = "gelu",
        eeg_indices: Sequence[int] = (0, 1),
        eog_indices: Sequence[int] = (2, 3),
        emg_indices: Sequence[int] = (4,),
        modality_split: Sequence[float] = (0.60, 0.25, 0.15),
        stem_kernel_size: int = 9,
        eeg_short_dilations: Sequence[int] = (1, 2, 4, 8),
        eeg_band_filters: int = 16,
        eeg_band_kernel: int = 129,
        eeg_band_fraction: float = 0.4,
        eeg_band_low_hz: float = 0.5,
        eeg_band_high_hz: float = 30.0,
        eeg_band_constraint: str = "legacy",
        eeg_band_allocation: Sequence[int] | None = None,
        eeg_band_minimum_width_hz: float = 0.1,
        eeg_band_lowpass_hz: float = 4.0,
        band_norm_per_recording: bool = False,
        band_norm_momentum: float = 0.1,
        band_norm_warmup_updates: int = 20,
        band_norm_max_recordings: int = 20000,
        band_norm_statistic: str = "ema",
        band_norm_modalities: str = "eeg",
        eog_decimation: int = 8,
        eog_kernel: int = 15,
        eog_dilations: Sequence[int] = (1, 2, 4),
        emg_filters: int = 24,
        emg_kernel: int = 11,
        emg_lowpass_hz: float = 10.0,
        fir_taps: int = 47,
        fir_cutoff_ratio: float = 0.85,
        fir_beta: float = 6.0,
        trunk_kernel_size: int = 7,
        trunk_strides: Sequence[int] = (2, 2, 2, 1),
        trunk_dilations: Sequence[Sequence[int]] = (
            (1, 2, 4),
            (1, 4, 8),
            (1, 4, 8),
            (1, 4, 8),
        ),
        trunk_fir_taps: Sequence[int] = (31, 23, 23, 23),
        pooling_mode: str = "attentive_stats",
        pool_out_dim: int | None = None,
        pool_heads: int = 2,
        pool_bottleneck: int = 128,
        pool_occupancy: bool = True,
        pool_mfa: bool = True,
        pool_dropout: float | None = None,
    ) -> None:
        super().__init__()
        if norm != "bn":
            raise ValueError(
                f"MultiRateAsymmetricEpochCNN requires norm='bn', got {norm!r}"
            )
        if pooling_mode not in _POOLING_MODES:
            raise ValueError(
                f"pooling_mode must be one of {_POOLING_MODES}, got {pooling_mode!r}"
            )
        if not isinstance(in_ch, int) or in_ch < 1:
            raise ValueError(f"in_ch must be a positive integer, got {in_ch!r}")
        if not isinstance(time_len, int) or time_len < 16:
            raise ValueError(f"time_len must be an integer >= 16, got {time_len!r}")
        resolved_widths = tuple(
            int(w) for w in (_DEFAULT_WIDTHS if widths is None else widths)
        )
        if len(resolved_widths) != 5 or any(w < 1 for w in resolved_widths):
            raise ValueError(
                f"widths must be five positive integers, got {resolved_widths!r}"
            )
        strides = tuple(int(s) for s in trunk_strides)
        if len(strides) != 4 or any(s not in (1, 2) for s in strides):
            raise ValueError(
                f"trunk_strides must contain four values in (1, 2), got {strides!r}"
            )
        dilations = tuple(tuple(int(d) for d in group) for group in trunk_dilations)
        if len(dilations) != 4 or any(
            not g or any(d < 1 for d in g) for g in dilations
        ):
            raise ValueError(
                f"trunk_dilations must be four tuples of positive integers, got {dilations!r}"
            )
        taps = tuple(int(t) for t in trunk_fir_taps)
        if len(taps) != 4:
            raise ValueError(f"trunk_fir_taps must contain four values, got {taps!r}")
        for value, name in (
            (fir_taps, "fir_taps"),
            (trunk_kernel_size, "trunk_kernel_size"),
        ):
            _odd_int(value, name)
        for t in taps:
            _odd_int(t, "trunk_fir_taps")
        if not 0.0 < fir_cutoff_ratio <= 1.0:
            raise ValueError(
                f"fir_cutoff_ratio must be in (0, 1], got {fir_cutoff_ratio!r}"
            )
        if fir_beta < 0.0:
            raise ValueError(f"fir_beta must be nonnegative, got {fir_beta!r}")
        if pool_out_dim is not None and (
            not isinstance(pool_out_dim, int) or pool_out_dim < 1
        ):
            raise ValueError(
                f"pool_out_dim must be a positive integer, got {pool_out_dim!r}"
            )
        if pool_dropout is not None and not 0.0 <= pool_dropout < 1.0:
            raise ValueError(f"pool_dropout must be in [0, 1), got {pool_dropout!r}")

        self.in_ch = in_ch
        self.time_len = time_len
        self.widths = resolved_widths
        self.dropout = float(dropout)
        self.norm_type = norm
        if fs != 128:
            raise ValueError("SPECTRA requires a fixed sample rate of 128 Hz")
        self.fs = 128.0
        self.activation = str(activation)
        self.eeg_indices = tuple(int(i) for i in eeg_indices)
        self.eog_indices = tuple(int(i) for i in eog_indices)
        self.emg_indices = tuple(int(i) for i in emg_indices)
        self.modality_split = tuple(float(v) for v in modality_split)
        self.stem_kernel_size = int(stem_kernel_size)
        self.eeg_short_dilations = tuple(int(d) for d in eeg_short_dilations)
        self.eeg_band_filters = int(eeg_band_filters)
        self.eeg_band_kernel = int(eeg_band_kernel)
        self.eeg_band_fraction = float(eeg_band_fraction)
        self.eeg_band_low_hz = float(eeg_band_low_hz)
        self.eeg_band_high_hz = float(eeg_band_high_hz)
        self.eeg_band_constraint = str(eeg_band_constraint)
        self.eeg_band_allocation = (
            None
            if eeg_band_allocation is None
            else tuple(int(value) for value in eeg_band_allocation)
        )
        self.eeg_band_minimum_width_hz = float(eeg_band_minimum_width_hz)
        self.eeg_band_lowpass_hz = float(eeg_band_lowpass_hz)
        self.band_norm_per_recording = bool(band_norm_per_recording)
        self.band_norm_momentum = float(band_norm_momentum)
        self.band_norm_warmup_updates = int(band_norm_warmup_updates)
        self.band_norm_max_recordings = int(band_norm_max_recordings)
        if band_norm_statistic not in BAND_NORM_STATISTICS:
            raise ValueError(
                f"band_norm_statistic must be one of {BAND_NORM_STATISTICS}, "
                f"got {band_norm_statistic!r}"
            )
        self.band_norm_statistic = str(band_norm_statistic)
        self.band_norm_modalities = ",".join(
            parse_band_norm_modalities(band_norm_modalities)
        )
        self.eog_decimation = int(eog_decimation)
        self.eog_kernel = int(eog_kernel)
        self.eog_dilations = tuple(int(d) for d in eog_dilations)
        self.emg_filters = int(emg_filters)
        self.emg_kernel = int(emg_kernel)
        self.emg_lowpass_hz = float(emg_lowpass_hz)
        self.fir_taps = int(fir_taps)
        self.fir_cutoff_ratio = float(fir_cutoff_ratio)
        self.fir_beta = float(fir_beta)
        self.trunk_kernel_size = int(trunk_kernel_size)
        self.trunk_strides = strides
        self.trunk_dilations = dilations
        self.trunk_fir_taps = taps
        self.pooling_mode = pooling_mode
        self.pool_out_dim = None if pool_out_dim is None else int(pool_out_dim)
        self.pool_heads = int(pool_heads)
        self.pool_bottleneck = int(pool_bottleneck)
        self.pool_occupancy = bool(pool_occupancy)
        self.pool_mfa = bool(pool_mfa)
        self.pool_dropout = None if pool_dropout is None else float(pool_dropout)

        self.multirate_stem = MultiRateModalityStem(
            in_ch,
            resolved_widths[0],
            eeg_indices=self.eeg_indices,
            eog_indices=self.eog_indices,
            emg_indices=self.emg_indices,
            modality_split=self.modality_split,
            dropout=self.dropout,
            norm=norm,
            activation=self.activation,
            fs=self.fs,
            fir_taps=self.fir_taps,
            fir_cutoff_ratio=self.fir_cutoff_ratio,
            fir_beta=self.fir_beta,
            stem_kernel_size=self.stem_kernel_size,
            eeg_short_dilations=self.eeg_short_dilations,
            eeg_band_filters=self.eeg_band_filters,
            eeg_band_kernel=self.eeg_band_kernel,
            eeg_band_fraction=self.eeg_band_fraction,
            eeg_band_low_hz=self.eeg_band_low_hz,
            eeg_band_high_hz=self.eeg_band_high_hz,
            eeg_band_constraint=self.eeg_band_constraint,
            eeg_band_allocation=self.eeg_band_allocation,
            eeg_band_minimum_width_hz=self.eeg_band_minimum_width_hz,
            eeg_band_lowpass_hz=self.eeg_band_lowpass_hz,
            band_norm_per_recording=self.band_norm_per_recording,
            band_norm_momentum=self.band_norm_momentum,
            band_norm_warmup_updates=self.band_norm_warmup_updates,
            band_norm_max_recordings=self.band_norm_max_recordings,
            band_norm_statistic=self.band_norm_statistic,
            band_norm_modalities=self.band_norm_modalities,
            eog_decimation=self.eog_decimation,
            eog_kernel=self.eog_kernel,
            eog_dilations=self.eog_dilations,
            emg_filters=self.emg_filters,
            emg_kernel=self.emg_kernel,
            emg_lowpass_hz=self.emg_lowpass_hz,
        )

        def _block(cin: int, cout: int, stage: int, stride: int) -> MultiDilatedBlock:
            return MultiDilatedBlock(
                in_ch=cin,
                out_ch=cout,
                kernel_size=self.trunk_kernel_size,
                dilations=self.trunk_dilations[stage],
                stride=stride,
                dropout=self.dropout,
                norm=norm,
                activation=self.activation,
                aa_cutoff_ratio=self.fir_cutoff_ratio,
                aa_num_taps=self.trunk_fir_taps[stage],
                aa_beta=self.fir_beta,
                anti_alias_dilated_branches=False,
                aa_legacy_cutoff=False,
            )

        self.trunk = nn.ModuleList(
            nn.Sequential(
                _block(resolved_widths[i], resolved_widths[i + 1], i, strides[i]),
                _block(resolved_widths[i + 1], resolved_widths[i + 1], i, 1),
            )
            for i in range(4)
        )

        length = -(-time_len // 2)
        for stride in strides:
            if stride > 1:
                length = -(-length // stride)
        self.final_temporal_len = length
        self.total_downsample_factor = 2 * math.prod(strides)

        pre_pool = resolved_widths[4]
        self.aggregate: MultiLayerFeatureAggregation | None = (
            MultiLayerFeatureAggregation(
                (resolved_widths[3], resolved_widths[4]),
                pre_pool,
                strides=(strides[3], 1),
                activation=self.activation,
                norm=norm,
                fir_taps=self.trunk_fir_taps[3],
                fir_cutoff_ratio=self.fir_cutoff_ratio,
                fir_beta=self.fir_beta,
            )
            if self.pool_mfa
            else None
        )
        out_dim = pre_pool if self.pool_out_dim is None else self.pool_out_dim
        pool_p = self.dropout if self.pool_dropout is None else self.pool_dropout
        if pooling_mode == "attentive_stats":
            self.pool: nn.Module = AttentiveStatisticsPool(
                pre_pool,
                out_dim=out_dim,
                num_heads=self.pool_heads,
                bottleneck=self.pool_bottleneck,
                occupancy=self.pool_occupancy,
                dropout=pool_p,
            )
        else:
            self.pool = LatentQueryAttentionPool(
                channels=pre_pool, out_dim=out_dim, num_heads=4, dropout=pool_p
            )
        self.out_dim = out_dim
        self._temporal_dim = pre_pool

    @property
    def temporal_dim(self) -> int:
        """Channel width of the pre-pooling (stage-4) feature map."""
        return self._temporal_dim

    def _prepare_input(
        self, x: torch.Tensor, channel_mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if x.ndim != 3 or x.size(1) != self.in_ch:
            raise ValueError(f"expected [N, {self.in_ch}, T], got {tuple(x.shape)}")
        if channel_mask is not None:
            if channel_mask.ndim == 1:
                channel_mask = channel_mask.unsqueeze(0).expand(x.size(0), -1)
            if channel_mask.shape != x.shape[:2]:
                raise ValueError(
                    f"channel_mask must have shape {tuple(x.shape[:2])}, "
                    f"got {tuple(channel_mask.shape)}"
                )
            channel_mask = channel_mask.to(device=x.device, dtype=x.dtype)
            x = x * channel_mask.unsqueeze(-1)
        return x, channel_mask

    def _run(self, module: nn.Module, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return module(x, **kwargs)

    def _forward_backbone(
        self,
        x: torch.Tensor,
        channel_mask: torch.Tensor | None,
        recording_index: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the stage-3 and stage-4 maps ``[N, C, T']``."""
        x, channel_mask = self._prepare_input(x, channel_mask)
        x = self._run(
            self.multirate_stem,
            x,
            channel_mask=channel_mask,
            recording_index=recording_index,
        )
        x = self._run(self.trunk[0], x)
        x = self._run(self.trunk[1], x)
        stage3 = self._run(self.trunk[2], x)
        stage4 = self._run(self.trunk[3], stage3)
        return stage3, stage4

    def forward_temporal(
        self,
        x: torch.Tensor,
        channel_mask: torch.Tensor | None = None,
        recording_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the stage-4 map as ``[N, T', D]`` (channels last)."""
        _, stage4 = self._forward_backbone(x, channel_mask, recording_index)
        return stage4.transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        channel_mask: torch.Tensor | None = None,
        recording_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode a flattened epoch batch to ``[N, out_dim]``.

        ``recording_index`` is ``[N]`` -- one entry per *flattened* epoch. For
        context inputs use :meth:`PerRecordingBandNorm.expand_recording_index`
        to repeat each recording id ``L`` times first.
        """
        stage3, stage4 = self._forward_backbone(x, channel_mask, recording_index)
        pooled_input = (
            self.aggregate([stage3, stage4]) if self.aggregate is not None else stage4
        )
        return self.pool(pooled_input)

    @torch.no_grad()
    def sync_band_norm_reference(self) -> bool:
        """Sync every per-recording norm to its trunk BatchNorm statistics.

        Call after loading pretrained encoder weights so the correction starts
        from the scale the trunk was trained on. No-op when the feature is off.
        """
        synced = False
        eeg = getattr(self.multirate_stem, "eeg_branch", None)
        if eeg is not None:
            synced |= bool(eeg.sync_band_norm_reference())
        emg = getattr(self.multirate_stem, "emg_branch", None)
        if emg is not None:
            synced |= bool(emg.sync_env_norm_reference())
        return synced

    def recording_norm_modules(self) -> dict[str, PerRecordingBandNorm]:
        """Return the enabled per-recording norms keyed by modality.

        Empty when the feature is off. Keys follow
        :data:`BAND_NORM_MODALITIES` order so every caller iterates identically.
        """
        out: dict[str, PerRecordingBandNorm] = {}
        eeg = getattr(self.multirate_stem, "eeg_branch", None)
        norm = getattr(eeg, "band_recording_norm", None)
        if norm is not None and norm.enabled:
            out["eeg"] = norm
        emg = getattr(self.multirate_stem, "emg_branch", None)
        norm = getattr(emg, "env_recording_norm", None)
        if norm is not None and norm.enabled:
            out["emg"] = norm
        return out

    def modality_channel_indices(self, modality: str) -> tuple[int, ...]:
        """Return the canonical channel indices feeding ``modality``'s branch."""
        if modality == "eeg":
            return tuple(self.eeg_indices)
        if modality == "emg":
            return tuple(self.emg_indices)
        raise ValueError(
            f"unknown band-norm modality {modality!r}; expected one of "
            f"{BAND_NORM_MODALITIES}"
        )

    def modality_envelope(self, modality: str, x: torch.Tensor) -> torch.Tensor:
        """Return the raw log envelope ``[N, C, T]`` that ``modality``'s norm sees.

        ``x`` is ``[N, n_modality_channels, T]`` -- already sliced with
        :meth:`modality_channel_indices`. This is the function every statistics
        pass (training table, inference pin) must call so both see the same
        envelope.
        """
        if modality == "eeg":
            branch = self.multirate_stem.eeg_branch
            if branch is None:
                raise ValueError("this encoder has no EEG branch")
            return branch.band_envelope(x)
        if modality == "emg":
            emg = self.multirate_stem.emg_branch
            if emg is None:
                raise ValueError("this encoder has no EMG branch")
            return emg.envelope(x)
        raise ValueError(
            f"unknown band-norm modality {modality!r}; expected one of "
            f"{BAND_NORM_MODALITIES}"
        )

    def get_attention_stats(self) -> None:
        """Return ``None``; no pooling state is retained between forwards."""
        return None

    def get_config(self) -> dict[str, Any]:
        """Return constructor kwargs that rebuild this encoder exactly."""
        return {
            "in_ch": self.in_ch,
            "time_len": self.time_len,
            "widths": self.widths,
            "dropout": self.dropout,
            "norm": self.norm_type,
            "fs": self.fs,
            "activation": self.activation,
            "eeg_indices": self.eeg_indices,
            "eog_indices": self.eog_indices,
            "emg_indices": self.emg_indices,
            "modality_split": self.modality_split,
            "stem_kernel_size": self.stem_kernel_size,
            "eeg_short_dilations": self.eeg_short_dilations,
            "eeg_band_filters": self.eeg_band_filters,
            "eeg_band_kernel": self.eeg_band_kernel,
            "eeg_band_fraction": self.eeg_band_fraction,
            "eeg_band_low_hz": self.eeg_band_low_hz,
            "eeg_band_high_hz": self.eeg_band_high_hz,
            "eeg_band_constraint": self.eeg_band_constraint,
            "eeg_band_allocation": self.eeg_band_allocation,
            "eeg_band_minimum_width_hz": self.eeg_band_minimum_width_hz,
            "eeg_band_lowpass_hz": self.eeg_band_lowpass_hz,
            "band_norm_per_recording": self.band_norm_per_recording,
            "band_norm_momentum": self.band_norm_momentum,
            "band_norm_warmup_updates": self.band_norm_warmup_updates,
            "band_norm_max_recordings": self.band_norm_max_recordings,
            "band_norm_statistic": self.band_norm_statistic,
            "band_norm_modalities": self.band_norm_modalities,
            "eog_decimation": self.eog_decimation,
            "eog_kernel": self.eog_kernel,
            "eog_dilations": self.eog_dilations,
            "emg_filters": self.emg_filters,
            "emg_kernel": self.emg_kernel,
            "emg_lowpass_hz": self.emg_lowpass_hz,
            "fir_taps": self.fir_taps,
            "fir_cutoff_ratio": self.fir_cutoff_ratio,
            "fir_beta": self.fir_beta,
            "trunk_kernel_size": self.trunk_kernel_size,
            "trunk_strides": self.trunk_strides,
            "trunk_dilations": self.trunk_dilations,
            "trunk_fir_taps": self.trunk_fir_taps,
            "pooling_mode": self.pooling_mode,
            "pool_out_dim": self.pool_out_dim,
            "pool_heads": self.pool_heads,
            "pool_bottleneck": self.pool_bottleneck,
            "pool_occupancy": self.pool_occupancy,
            "pool_mfa": self.pool_mfa,
            "pool_dropout": self.pool_dropout,
        }

    def get_architecture_info(self) -> dict[str, Any]:
        """Return diagnostic metadata (not a constructor round trip)."""
        rate = self.fs / 2.0
        resolution = {"post_stem": rate}
        for idx, stride in enumerate(self.trunk_strides, start=1):
            rate /= stride
            resolution[f"post_stage{idx}"] = rate
        info: dict[str, Any] = {
            "encoder_type": "MultiRateAsymmetricEpochCNN",
            "widths": self.widths,
            "trunk_strides": self.trunk_strides,
            "final_temporal_len": self.final_temporal_len,
            "total_downsample_factor": self.total_downsample_factor,
            "temporal_dim": self._temporal_dim,
            "out_dim": self.out_dim,
            "pooling_mode": self.pooling_mode,
            "pool_mfa": self.pool_mfa,
            "temporal_resolution_hz": resolution,
            "eog_rate_hz": self.fs / self.eog_decimation,
            "param_count": sum(p.numel() for p in self.parameters()),
        }
        info.update(self.multirate_stem.get_architecture_info())
        return info
