"""Embedded preprocessing normalization module.

This module implements normalization as a learnable PyTorch module that can be
embedded as the first layer of a model. The normalization logic exactly mirrors
the logic in spectra.data.channel.normalization for reproducibility.

Key features:
- Calibration: Compute per-channel quartiles/median statistics once
- Forward: Apply IQR or sigma-based normalization with consistent clipping
- Checkpoint persistence: Statistics saved as buffers in model state_dict
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import ProcCfg

__all__ = ["EmbeddedPreproc"]


class EmbeddedPreproc(nn.Module):
    """Embedded preprocessing normalization layer.

    Implements the same normalization strategy as ``PSGNormalizer`` with learnable
    buffers that can be saved inside a checkpoint. Supports multiple normalization modes:

    - **iqr** (default): Median removal, divide by IQR, clip at ±20.
      Matches batch_edf_to_zarr_fp32.py offline preprocessing.
    - **percentile_1_99**: 1st-99th percentile clip-and-scale to [-1, 1]
    - **sigma**: Legacy sigma-based normalization with configurable clipping
    - **keep**: Legacy percentile clipping mode

    The default ``iqr`` mode matches batch_edf_to_zarr_fp32.py preprocessing.
    """

    def __init__(
        self,
        cfg: ProcCfg,
        channel_types: list[str],
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.channel_types = channel_types
        self.n_channels = len(channel_types)

        # Handle device input: convert string to torch.device if needed
        # CRITICAL: torch.compile detects CPU buffers and skips CUDA graphs
        if device is None:
            device = torch.device("cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        if dtype is None:
            dtype = torch.float32

        # Register buffers for normalization statistics (shape: 1 x C x 1)
        self.register_buffer(
            "q1", torch.zeros(1, self.n_channels, 1, device=device, dtype=dtype)
        )
        self.register_buffer(
            "q3", torch.zeros(1, self.n_channels, 1, device=device, dtype=dtype)
        )
        self.register_buffer(
            "median", torch.zeros(1, self.n_channels, 1, device=device, dtype=dtype)
        )
        self.register_buffer(
            "iqr", torch.ones(1, self.n_channels, 1, device=device, dtype=dtype)
        )
        self.register_buffer(
            "clip_threshold",
            torch.full(
                (1, self.n_channels, 1),
                float(self.cfg.normalization.clip_threshold),
                device=device,
                dtype=dtype,
            ),
        )
        self.register_buffer("calibrated", torch.tensor(False, device=device))

        # Type hints for buffers (for type checker)
        self.q1: torch.Tensor
        self.q3: torch.Tensor
        self.median: torch.Tensor
        self.calibrated: torch.Tensor
        self.iqr: torch.Tensor
        self.clip_threshold: torch.Tensor
        self.k_thresholds: torch.Tensor

        # Pre-compute channel-specific k thresholds
        # Must match PSGNormalizer: EMG uses k=12.0, EEG/EOG/ECG use k=10.0
        k_values = []
        for ch_type in channel_types:
            ch_lower = ch_type.lower()
            # Check if channel is EMG type
            if "emg" in ch_lower or "chin" in ch_lower or "mentalis" in ch_lower:
                k_values.append(cfg.normalization.clamp["k_emg"])  # Default: 12.0
            else:
                k_values.append(cfg.normalization.clamp["k_eeg"])  # Default: 10.0

        self.register_buffer(
            "k_thresholds",
            torch.tensor(k_values, device=device, dtype=dtype).view(1, -1, 1),
        )

    @torch.no_grad()
    def calibrate(
        self,
        sample_signals: torch.Tensor | list[torch.Tensor],
        channel_mask: torch.Tensor | None = None,
        *,
        device: torch.device | None = None,
    ) -> None:
        """Calibrate normalization statistics from raw waveforms.

        Args:
            sample_signals: Tensor or list of tensors shaped (C, T) or (B, C, T).
                Use the full-night recording when possible for stable quartiles.
            channel_mask: Optional binary mask shaped ``[C]`` or ``[B,C]``.
            device: Optional device override. Defaults to the module's buffers.
        """
        if device is None:
            device = self.median.device

        def _prepare_tensor(t: torch.Tensor) -> torch.Tensor:
            t = t.to(device).float()
            if t.ndim == 2:
                t = t.unsqueeze(0)
            elif t.ndim != 3:
                raise ValueError(
                    f"Expected tensor with 2 or 3 dims, got shape {t.shape}"
                )
            # Collapse batch dimension for concatenation along time
            return t.reshape(1, t.size(1), -1)

        if isinstance(sample_signals, list):
            if not sample_signals:
                raise ValueError("sample_signals list must not be empty")
            signals = torch.cat([_prepare_tensor(t) for t in sample_signals], dim=-1)
        else:
            signals = sample_signals.to(device).float()
            if signals.ndim == 2:
                signals = signals.unsqueeze(0)

        if signals.ndim != 3:
            raise ValueError(f"Expected (B, C, T) tensor, got shape {signals.shape}")
        if signals.size(1) != self.n_channels:
            raise ValueError(
                f"Channel mismatch: expected {self.n_channels}, got {signals.size(1)}"
            )

        # Flatten batch/time for percentile computation: (B, C, T) -> (C, B*T)
        x_flat = signals.transpose(0, 1).reshape(self.n_channels, -1)

        if channel_mask is None:
            resolved_mask = torch.ones(self.n_channels, dtype=torch.bool, device=device)
        else:
            resolved_mask = channel_mask.to(device)
            if resolved_mask.dtype != torch.bool:
                resolved_mask = resolved_mask.bool()
            if resolved_mask.ndim == 1 and resolved_mask.shape != (self.n_channels,):
                raise ValueError(
                    f"Expected channel_mask shape {(self.n_channels,)}, got "
                    f"{tuple(resolved_mask.shape)}"
                )
            if resolved_mask.ndim == 2 and resolved_mask.shape != (
                signals.size(0),
                self.n_channels,
            ):
                raise ValueError(
                    "Expected epoch-specific channel_mask shape "
                    f"{(signals.size(0), self.n_channels)}, got "
                    f"{tuple(resolved_mask.shape)}"
                )
            if resolved_mask.ndim not in (1, 2):
                raise ValueError(
                    "channel_mask must be [C] or [B,C], got "
                    f"{tuple(resolved_mask.shape)}"
                )

        mode = self.cfg.normalization.mode
        if mode == "iqr":
            lo_frac = self.cfg.normalization.q1_percentile / 100.0
            hi_frac = self.cfg.normalization.q3_percentile / 100.0
        elif mode == "percentile_1_99":
            # Match batch_edf_to_zarr_fp32.py normalization
            lo_frac = self.cfg.normalization.p_low_percentile / 100.0
            hi_frac = self.cfg.normalization.p_high_percentile / 100.0
        elif mode == "sigma":
            # Legacy sigma mode still uses P0.5/P99.5 to estimate sigma
            lo_frac, hi_frac = 0.005, 0.995
        else:  # keep mode
            lo_frac, hi_frac = 0.01, 0.99

        q = torch.tensor([lo_frac, hi_frac], dtype=signals.dtype, device=device)

        q1_vals = torch.zeros(self.n_channels, dtype=signals.dtype, device=device)
        q3_vals = torch.ones(self.n_channels, dtype=signals.dtype, device=device)
        med_vals = torch.zeros(self.n_channels, dtype=signals.dtype, device=device)

        for c in range(self.n_channels):
            if resolved_mask.ndim == 1:
                channel_present = bool(resolved_mask[c].item())
                ch_data = x_flat[c]
            else:
                valid_rows = resolved_mask[:, c]
                channel_present = bool(valid_rows.any().item())
                ch_data = signals[valid_rows, c, :].reshape(-1)
            if not channel_present:
                q1_vals[c] = -1.0
                q3_vals[c] = 1.0
                med_vals[c] = 0.0
                continue

            if not torch.isfinite(ch_data).all():
                ch_data = ch_data[torch.isfinite(ch_data)]
            if ch_data.numel() == 0:
                q1_vals[c] = -1.0
                q3_vals[c] = 1.0
                med_vals[c] = 0.0
                continue

            qvals = torch.quantile(ch_data, q)
            q1_vals[c], q3_vals[c] = qvals[0], qvals[1]
            med_vals[c] = torch.median(ch_data)

        # Store in buffers (reshape to 1 x C x 1 for broadcasting)
        self.q1.copy_(q1_vals.view(1, self.n_channels, 1))
        self.q3.copy_(q3_vals.view(1, self.n_channels, 1))
        self.median.copy_(med_vals.view(1, self.n_channels, 1))
        iqr = torch.clamp(self.q3 - self.q1, min=1e-6)
        self.iqr.copy_(iqr)
        self.clip_threshold.fill_(float(self.cfg.normalization.clip_threshold))
        self.calibrated.fill_(True)

    def forward(
        self,
        x: torch.Tensor,
        channel_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply normalization using the calibrated statistics."""
        torch._assert(
            self.calibrated,
            "EmbeddedPreproc must be calibrated before use. Call calibrate() first.",
        )

        x_fp32 = x.float()
        eps = 1e-6
        mode = self.cfg.normalization.mode

        if mode == "iqr":
            iqr = torch.clamp(self.iqr, min=eps)
            z = (x_fp32 - self.median) / iqr
            return torch.clamp(z, -self.clip_threshold, self.clip_threshold)

        if mode == "percentile_1_99":
            # Match batch_edf_to_zarr_fp32.py normalization:
            # 1. Clip to [p_low, p_high] percentiles
            # 2. Scale to [0, 1]: (x - p_low) / (p_high - p_low)
            # 3. Scale to [-1, 1]: 2 * x - 1
            p_range = torch.clamp(self.q3 - self.q1, min=eps)
            x_clipped = torch.clamp(x_fp32, self.q1, self.q3)
            x_normalized = (x_clipped - self.q1) / p_range  # Scale to [0, 1]
            return 2.0 * x_normalized - 1.0  # Scale to [-1, 1]

        if mode == "sigma":
            zspan = self.cfg.normalization.zspan
            sigma = torch.clamp((self.q3 - self.q1) / zspan, min=eps)
            z = (x_fp32 - self.median) / sigma
            if self.cfg.normalization.clip_method == "hard":
                return torch.clamp(z, -self.k_thresholds, self.k_thresholds)
            tanh_c_val = self.cfg.normalization.clamp.get("tanh_c", 10.0)
            tanh_c = float(tanh_c_val) if tanh_c_val is not None else 10.0
            return torch.tanh(z / tanh_c)

        # Legacy keep mode
        x_clipped = torch.clamp(x_fp32, self.q1, self.q3)
        span = torch.clamp(self.q3 - self.q1, min=eps)
        return (x_clipped - self.median) / span

    @property
    def p_lo(self) -> torch.Tensor:
        """Backward-compatible alias for lower percentile buffer."""
        return self.q1

    @property
    def p_hi(self) -> torch.Tensor:
        """Backward-compatible alias for upper percentile buffer."""
        return self.q3

    def extra_repr(self) -> str:
        """Return extra representation string for module."""
        return (
            f"n_channels={self.n_channels}, "
            f"mode={self.cfg.normalization.mode}, "
            f"calibrated={self.calibrated.item()}"
        )
