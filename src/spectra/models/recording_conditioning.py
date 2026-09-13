"""Frozen early-feature observations and trainable recording-conditioned FiLM."""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import nn

ObservationProvider = Callable[
    [torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
]


def select_recording_epochs(valid: torch.Tensor, samples: int) -> torch.Tensor:
    """Select evenly spaced valid epochs, including both recording endpoints."""
    if samples < 1 or samples > 64:
        raise ValueError("recording_conditioning_samples must be in [1, 64]")
    indices = valid.to(dtype=torch.bool).reshape(-1).nonzero().flatten()
    if indices.numel() <= samples:
        return indices
    positions = (
        torch.linspace(0, indices.numel() - 1, samples, device=indices.device)
        .round()
        .long()
    )
    return indices.index_select(0, positions)


class FrozenRecordingObserver(nn.Module):
    """Checkpointed snapshot of a parent's stem and first stage, always in eval."""

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.multirate = hasattr(encoder, "multirate_stem")
        self.legacy_unmasked_stem = bool(
            getattr(encoder, "legacy_unmasked_stem", False)
        )
        self.stem = copy.deepcopy(
            getattr(encoder, "multirate_stem" if self.multirate else "stem")
        )
        self.stage1: nn.Module = copy.deepcopy(
            cast(Any, encoder).trunk[0] if self.multirate else cast(Any, encoder).stage1
        )
        # Observer norms always use explicitly pinned support statistics. Avoid
        # carrying a second, unused recording vocabulary/table onto the device.
        if self.multirate:
            from .multirate_asymmetric_epoch_cnn import PerRecordingBandNorm

            for module in self.stem.modules():
                if isinstance(module, PerRecordingBandNorm):
                    module.running_mean = torch.zeros_like(module.running_mean[:1])
                    module.running_var = torch.ones_like(module.running_var[:1])
                    module.seen_count = torch.zeros_like(module.seen_count[:1])
                    module.max_recordings = 1
                    module.register_recording_ids({})
                    module.set_table_meta({})
                    module.clear_inference_statistics()
        self.freeze()

    def freeze(self) -> None:
        """Restore the observer freeze after any parent-wide training operation."""
        self.requires_grad_(False)
        super().train(False)

    def train(self, mode: bool = True) -> FrozenRecordingObserver:
        """Keep copied BatchNorm statistics and dropout fixed in every mode."""
        del mode
        self.freeze()
        return self

    def _pin_band_statistics(self, wave: torch.Tensor, mask: torch.Tensor) -> None:
        from .multirate_asymmetric_epoch_cnn import reduce_recording_statistics

        for modality, norm_name, envelope_name in (
            ("eeg", "band_recording_norm", "band_envelope"),
            ("emg", "env_recording_norm", "envelope"),
        ):
            branch = getattr(self.stem, f"{modality}_branch", None)
            norm = getattr(branch, norm_name, None)
            if norm is None or not norm.enabled:
                continue
            norm.clear_inference_statistics()
            indices = getattr(self.stem, f"{modality}_indices")
            present = mask[:, indices].bool()
            if not bool(present.any()):
                # This modality is excluded by the stem's branch mask.
                continue
            envelope = getattr(branch, envelope_name)(wave[:, indices, :]).float()
            mean, var = norm.per_sample_statistics(envelope)
            # Each input channel owns a contiguous filter block. Do not let an
            # absent channel's zero waveform contaminate its band's statistics.
            bands_present = present.repeat_interleave(
                mean.shape[1] // len(indices), dim=1
            )
            locations, scales = [], []
            for band in range(mean.shape[1]):
                keep = bands_present[:, band]
                if bool(keep.any()):
                    loc, scale = reduce_recording_statistics(
                        norm.statistic,
                        mean[keep, band : band + 1],
                        var[keep, band : band + 1],
                        eps=norm.eps,
                    )
                    locations.append(loc.squeeze(0).to(norm.ref_mean.device))
                    scales.append(scale.squeeze(0).to(norm.ref_std.device))
                else:
                    locations.append(norm.ref_mean[band])
                    scales.append(norm.ref_std[band])
            norm.set_inference_statistics(torch.stack(locations), torch.stack(scales))

    @torch.no_grad()
    def forward(self, wave: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Observe masked support with frozen fp32 kernels and normalization."""
        self.freeze()
        with torch.autocast(device_type=wave.device.type, enabled=False):
            wave = wave.float().masked_fill(~mask.bool().unsqueeze(-1), 0.0)
            if self.multirate:
                self._pin_band_statistics(wave, mask)
            try:
                stem_mask = None if self.legacy_unmasked_stem else mask
                result = self.stage1(self.stem(wave, channel_mask=stem_mask))
            finally:
                if self.multirate:
                    for module in self.stem.modules():
                        clear = getattr(module, "clear_inference_statistics", None)
                        if callable(clear):
                            clear()
            return result.float()


class RecordingConditioner(nn.Module):
    """Summarize detached night observations and modulate early CNN features.

    Args:
        encoder: Main encoder whose initialized early layers define the observer.
        stem_width: Channels at the first FiLM site.
        stage1_width: Channels at the second FiLM site and in observer features.
        samples: Maximum label-free support epochs, at most 64.
        dim: Trainable observation and recording embedding width.
    """

    def __init__(
        self,
        encoder: nn.Module,
        *,
        stem_width: int,
        stage1_width: int,
        samples: int = 64,
        dim: int = 64,
    ) -> None:
        super().__init__()
        if not 1 <= samples <= 64:
            raise ValueError("recording_conditioning_samples must be in [1, 64]")
        if dim < 1:
            raise ValueError("recording_conditioning_dim must be positive")
        self.samples = int(samples)
        self.dim = int(dim)
        self.in_ch = int(cast(Any, encoder).in_ch)
        self.observation_dim = stage1_width * 8 + self.in_ch
        self.observer = FrozenRecordingObserver(encoder)
        self.observation_encoder = nn.Sequential(
            nn.LayerNorm(self.observation_dim),
            nn.Linear(self.observation_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.attention = nn.Linear(dim, 1)
        self.summary_norm = nn.LayerNorm(dim)
        self.film_heads = nn.ModuleList(
            nn.Linear(dim, 2 * width) for width in (stem_width, stage1_width)
        )
        for module in self.film_heads:
            head = cast(nn.Linear, module)
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        self.provider: ObservationProvider | None = None
        self._observations: tuple[torch.Tensor, torch.Tensor] | None = None

    def freeze_observer(self) -> None:
        """Keep the observer frozen after main-encoder freeze/unfreeze changes."""
        self.observer.freeze()

    def reset_observer_from_encoder(self, encoder: nn.Module) -> None:
        """Replace the snapshot after a parent-only checkpoint warm start."""
        self.observer = FrozenRecordingObserver(encoder)
        self.clear_observations()

    @torch.no_grad()
    def compute_observations(
        self,
        wave: torch.Tensor,
        presence_mask: torch.Tensor,
        epoch_valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create detached fp32 temporal-bin statistics from valid support epochs."""
        if wave.ndim != 3 or wave.shape[1] != self.in_ch:
            raise ValueError(f"wave must have shape [E, {self.in_ch}, T]")
        mask = presence_mask.to(device=wave.device, dtype=torch.bool)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0).expand(wave.shape[0], -1)
        if mask.shape != wave.shape[:2]:
            raise ValueError("presence_mask must have shape [C] or [E, C]")
        valid = mask.any(dim=1)
        if epoch_valid is not None:
            if epoch_valid.shape != (wave.shape[0],):
                raise ValueError("epoch_valid must have shape [E]")
            valid = valid & epoch_valid.to(device=wave.device, dtype=torch.bool)
        selected = select_recording_epochs(valid, self.samples)
        if selected.numel() == 0:
            raise ValueError(
                "recording conditioning requires at least one valid observation"
            )
        device = next(self.observer.parameters()).device
        selected_wave = wave.index_select(0, selected).to(
            device=device, dtype=torch.float32
        )
        selected_mask = mask.index_select(0, selected).to(device=device)
        selected_mask = selected_mask & torch.isfinite(selected_wave).all(dim=-1)
        usable = selected_mask.any(dim=1)
        if not bool(usable.any()):
            raise ValueError(
                "recording conditioning requires at least one finite valid observation"
            )
        selected_wave, selected_mask = selected_wave[usable], selected_mask[usable]
        features = self.observer(selected_wave, selected_mask)
        means = F.adaptive_avg_pool1d(features, 4)
        # Direct per-bin variance avoids cancellation for nearly constant signals.
        stds = torch.stack(
            [
                features[
                    ...,
                    (i * features.shape[-1])
                    // 4 : ((i + 1) * features.shape[-1] + 3)
                    // 4,
                ]
                .var(dim=-1, unbiased=False)
                .clamp_min(0.0)
                .sqrt()
                for i in range(4)
            ],
            dim=-1,
        )
        tokens = torch.cat(
            (means.flatten(1), stds.flatten(1), selected_mask.float()), dim=1
        ).detach()
        return tokens, torch.ones(tokens.shape[0], dtype=torch.bool, device=device)

    @staticmethod
    def _validate_observations(
        tokens: torch.Tensor, valid: torch.Tensor, width: int
    ) -> None:
        if tokens.ndim != 3 or tokens.shape[-1] != width:
            raise ValueError(f"observations must have shape [U, K, {width}]")
        if valid.shape != tokens.shape[:2] or not bool(valid.bool().any(dim=1).all()):
            raise ValueError("each recording must have at least one valid observation")
        if not bool(torch.isfinite(tokens[valid.bool()]).all()):
            raise ValueError("valid recording observations must be finite")

    def set_observations(self, tokens: torch.Tensor, valid: torch.Tensor) -> None:
        """Pin one complete recording's detached observations for inference."""
        self._validate_observations(
            tokens.unsqueeze(0), valid.unsqueeze(0), self.observation_dim
        )
        self._observations = (tokens.detach().clone(), valid.detach().bool().clone())

    @property
    def pinned_observations(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Return current pinned support so nested inference can restore it."""
        return self._observations

    def clear_observations(self) -> None:
        """Remove pinned context without changing a training provider."""
        self._observations = None

    @torch.compiler.disable()
    def _resolve_observations(
        self, recording_index: torch.Tensor | None, batch: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.freeze_observer()
        if self._observations is not None:
            tokens, valid = self._observations
            tokens, valid = tokens.unsqueeze(0), valid.unsqueeze(0)
            inverse = torch.zeros(batch, dtype=torch.long, device=device)
        elif self.provider is not None and recording_index is not None:
            if recording_index.numel() != batch:
                raise ValueError(
                    "recording_index must contain one index per encoder input"
                )
            tokens, valid, inverse = self.provider(recording_index)
        else:
            raise ValueError(
                "recording conditioning needs pinned observations or a provider and recording_index"
            )
        self._validate_observations(tokens, valid, self.observation_dim)
        if inverse.shape != (batch,) or bool(
            ((inverse < 0) | (inverse >= tokens.shape[0])).any()
        ):
            raise ValueError(
                "provider inverse must map each input to its recording observations"
            )
        return (
            tokens.detach().to(device=device, dtype=torch.float32),
            valid.to(device=device, dtype=torch.bool),
            inverse.to(device=device, dtype=torch.long),
        )

    def context(
        self, recording_index: torch.Tensor | None, batch: int, device: torch.device
    ) -> torch.Tensor:
        """Compute trainable summaries once per unique recording in this batch."""
        tokens, valid, inverse = self._resolve_observations(
            recording_index, batch, device
        )
        with torch.autocast(device_type=device.type, enabled=False):
            tokens = tokens.masked_fill(~valid.unsqueeze(-1), 0.0)
            encoded = self.observation_encoder(tokens)
            scores = self.attention(encoded).squeeze(-1).masked_fill(~valid, -torch.inf)
            summary = (encoded * scores.softmax(dim=1).unsqueeze(-1)).sum(dim=1)
            summary = self.summary_norm(summary)
        return summary.index_select(0, inverse)

    def modulate(
        self, x: torch.Tensor, context: torch.Tensor, site: int
    ) -> torch.Tensor:
        """Apply identity-initialized gain and bias, each bounded to +/- 0.1."""
        with torch.autocast(device_type=x.device.type, enabled=False):
            gain, bias = self.film_heads[site](context.float()).tanh().chunk(2, dim=-1)
        return x * (1.0 + 0.1 * gain.to(x.dtype).unsqueeze(-1)) + 0.1 * bias.to(
            x.dtype
        ).unsqueeze(-1)
