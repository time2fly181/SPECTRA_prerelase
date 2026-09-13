"""
Sleep-specific feature extraction for PSG signals.
Extracts interpretable features that supplement CNN-learned representations.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

# Torch.compile compatibility: Prevent STFT compilation
# torch.stft has known issues with torch.compile that can produce NaN
# See: PyTorch Issue #88293, TorchInductor complex number limitations
try:
    # PyTorch 2.0+ with torch.compile support
    _torch_compile_disable = torch.compiler.disable

    def torch_compile_disable(func: Callable[..., Any]) -> Callable[..., Any]:
        return cast(Callable[..., Any], _torch_compile_disable(func))

    _COMPILE_DISABLE_AVAILABLE = True
except (ImportError, AttributeError):
    # PyTorch < 2.0 or torch.compile not available
    # Create a no-op decorator
    def torch_compile_disable(func: Callable[..., Any]) -> Callable[..., Any]:
        return func

    _COMPILE_DISABLE_AVAILABLE = False


@torch_compile_disable
def _compile_safe_stft(x_flat, n_fft, hop_length, window, **kwargs):
    """
    Wrapper around torch.stft that prevents compilation issues.

    **Why this is needed:**
    TorchInductor (torch.compile backend) has known bugs with complex FFT operations:
    - Complex number arithmetic is not fully optimized
    - Some edge cases produce incorrect results (including NaN)
    - CUDA graphs may be skipped, leading to inconsistent behavior
    - FP16/BF16 FFT operations can underflow/overflow without proper handling
    - Backward pass compilation fails with KeyError: 'complex64' in Triton codegen

    By using @torch.compiler.disable decorator, this function is explicitly
    excluded from compilation and runs in eager mode even when the parent
    model is compiled with torch.compile(). This applies to BOTH forward
    and backward passes, preventing complex64 dtype issues in gradients.

    **Performance impact:** Negligible (~1-2% overhead), as FFT is compute-bound
    and skipping compilation doesn't significantly affect runtime.

    References:
    - PyTorch Issue #88293: torch.compile + FFT produces NaN
    - TorchInductor complex number limitations
    - Triton KeyError: 'complex64' in backward pass
    - https://pytorch.org/docs/stable/generated/torch.compiler.disable.html
    """
    return torch.stft(
        x_flat, n_fft=n_fft, hop_length=hop_length, window=window, **kwargs
    )


@torch_compile_disable
def _compile_safe_istft(stft_matrix, n_fft, hop_length, window, **kwargs):
    """
    Wrapper around torch.istft that prevents compilation issues.

    Same rationale as _compile_safe_stft: complex number operations in both
    forward and backward passes cause Triton codegen errors with torch.compile.
    """
    return torch.istft(
        stft_matrix, n_fft=n_fft, hop_length=hop_length, window=window, **kwargs
    )


class SleepFeatureExtractor(nn.Module):
    """
    Extract sleep-specific features from raw PSG signals.

    Features:
    - Spectral power bands (delta, theta, alpha, sigma, beta)
    - Delta continuity summary (temporal variance statistics for N2/N3 discrimination)
    - Temporal summary ratios (spindle max ratio, K-complex amplitude ratio)
    - REM/EMG power summaries
    - Channel presence indicators
    - YASA-inspired Hjorth/ratio features

    These features complement CNN-learned representations with domain knowledge.

    Args:
        fs: Sampling frequency in Hz (default: 100)
        epoch_sec: Epoch duration in seconds (default: 30)
        learnable_bands: Whether to make band weights learnable (default: True)
        num_channels: Number of channels to expect (default: 3 for EEG, EOG, EMG)
        normalize: Normalization strategy ('none', 'standardize', 'minmax') (default: 'none')
            - 'none': No normalization applied
            - 'standardize': Z-score normalization (mean=0, std=1)
            - 'minmax': Min-max scaling to [0, 1] range

    Output:
        [B, F] feature vector where F = num_eeg_channels * 6 (bands) + presence + YASA features

    Example:
        >>> extractor = SleepFeatureExtractor(fs=128, epoch_sec=30, num_channels=3)
        >>> x = torch.randn(8, 3, 3840)  # [batch, channels, time]
        >>> features = extractor(x)  # [8, F] where F = num_eeg * 6 bands + C presence + YASA
    """

    def __init__(
        self,
        fs: int = 128,
        epoch_sec: int = 30,
        learnable_bands: bool = True,
        num_channels: int = 3,
        normalize: str = "none",
        channel_names: list[str] | None = None,
        # YASA feature parameters
        use_precomputed: bool = True,  # DEPRECATED: all features computed on-the-fly
        include_statistical: bool = False,  # DEPRECATED: statistical features removed
        include_hjorth: bool = True,
        include_slow_band: bool = False,  # DEPRECATED: slow band now in main band_powers
        include_extra_ratios: bool = True,
        # Multi-resolution STFT for better temporal/frequency trade-off
        use_multi_resolution: bool = False,
    ):
        super().__init__()
        self.fs = fs
        self.epoch_sec = epoch_sec
        self.n_samples = int(fs * epoch_sec)
        self.num_channels = num_channels

        # Deprecation warning for use_precomputed
        if use_precomputed:
            warnings.warn(
                "use_precomputed is deprecated and will be removed in a future version. "
                "All features are now computed on-the-fly.",
                DeprecationWarning,
                stacklevel=2,
            )

        # Validate normalization strategy
        if normalize not in ("none", "standardize", "minmax"):
            raise ValueError(
                f"normalize must be 'none', 'standardize', or 'minmax', got '{normalize}'"
            )
        self.normalize = normalize

        # Identify EEG channels for band power features
        # Import here to avoid circular dependency
        from spectra.data.channel import infer_channel_type

        self.channel_names = channel_names or [f"ch_{i}" for i in range(num_channels)]
        self.channel_types = [infer_channel_type(name) for name in self.channel_names]

        # Build EEG channel indices (only EEG channels used for band power features)
        self.eeg_channel_indices = [
            i for i, ch_type in enumerate(self.channel_types) if ch_type == "eeg"
        ]

        # Build EOG channel indices (for REM feature extraction)
        # REM eye movements should be extracted ONLY from EOG channels to avoid
        # delta band crosstalk from EEG channels
        self.eog_channel_indices = [
            i for i, ch_type in enumerate(self.channel_types) if ch_type == "eog"
        ]

        # Build EMG channel indices (for muscle tone features)
        self.emg_channel_indices = [
            i for i, ch_type in enumerate(self.channel_types) if ch_type == "emg"
        ]

        # If no channel types were detected (all "unknown", e.g. generic ch_0..ch_4
        # names when no canon_json is provided), apply the standard 5-channel PSG
        # layout: first 2 = EEG, next 2 = EOG, last = EMG.
        if (
            not self.eeg_channel_indices
            and not self.eog_channel_indices
            and not self.emg_channel_indices
            and num_channels >= 5
        ):
            self.eeg_channel_indices = [0, 1]
            self.eog_channel_indices = [2, 3]
            self.emg_channel_indices = [4]
        elif not self.eeg_channel_indices:
            # Fewer than 5 channels with no type info: treat all as EEG
            self.eeg_channel_indices = list(range(num_channels))

        self.num_eeg_channels = len(self.eeg_channel_indices)

        # Debug logging for channel detection
        if not torch.compiler.is_compiling():
            import logging

            logger = logging.getLogger(__name__)
            logger.info(
                f"SleepFeatureExtractor initialized: "
                f"num_channels={num_channels}, "
                f"num_eeg_channels={self.num_eeg_channels}, "
                f"eeg_indices={self.eeg_channel_indices}, "
                f"channel_names={self.channel_names[: min(10, len(self.channel_names))]}"
            )
            if self.num_eeg_channels == 0:
                logger.warning(
                    "No EEG channels detected! Band power features will be empty. "
                    "This may indicate incorrect channel naming or detection logic."
                )

            # EMG feature warning for low sampling rates
            if fs < 200:
                logger.warning(
                    f"Sampling rate {fs}Hz is below 200Hz. EMG features (30-50Hz band) "
                    f"may be unreliable due to Nyquist limit ({fs / 2:.1f}Hz). "
                    f"Consider using time-domain EMG features (RMS, variance) instead, "
                    f"or increase sampling rate to >= 200Hz for accurate EMG spectral analysis."
                )

        # Register EEG indices as a buffer for torch.compile compatibility
        # This prevents dynamic tensor creation in forward pass which breaks CUDA graphs
        self.eeg_indices_tensor: torch.Tensor
        eeg_indices_tensor = torch.tensor(self.eeg_channel_indices, dtype=torch.long)
        self.register_buffer("eeg_indices_tensor", eeg_indices_tensor, persistent=False)

        # STFT parameters (better than FFT for spectral features)
        # Target 5s window for ~0.2 Hz resolution - critical for low-frequency bands
        # Previous 2.56s window gave only ~2 bins for slow wave (0.4-1 Hz)
        # With 5s: slow wave (0.4-1 Hz) gets ~3 bins, fdelta (1-4 Hz) gets ~15 bins
        # For 100Hz: 500 samples (5s) -> resolution = 100/500 = 0.2 Hz
        # For 128Hz: 640 samples (5s) -> resolution = 128/640 = 0.2 Hz
        self.n_fft = int(5.0 * self.fs)
        # Ensure even number for stability
        if self.n_fft % 2 != 0:
            self.n_fft += 1

        self.hop_length = self.n_fft // 2  # 50% overlap for smooth spectra

        # Pre-compute and register window as buffer
        self.window: torch.Tensor
        window = torch.hann_window(self.n_fft)
        self.register_buffer("window", window, persistent=False)

        # Multi-resolution STFT configuration
        # Problem: 5s window gives 0.2 Hz resolution but only ~2 STFT frames per 30s epoch
        # This limits temporal dynamics capture (K-complex timing, spindle bursts)
        # Solution: Use longer windows for slow bands (need frequency resolution) and
        # shorter windows for fast bands (need temporal resolution)
        self.use_multi_resolution = use_multi_resolution
        if use_multi_resolution:
            # Slow resolution: 5s window for slow oscillations (slow, fdelta)
            # - Good frequency resolution (0.2 Hz) for precise low-frequency measurement
            # - ~6 STFT frames per 30s epoch with 50% overlap
            slow_n_fft = int(5.0 * self.fs)
            if slow_n_fft % 2 != 0:
                slow_n_fft += 1
            self.slow_n_fft = slow_n_fft
            self.slow_hop_length = slow_n_fft // 2
            self.slow_window: torch.Tensor
            slow_window = torch.hann_window(slow_n_fft)
            self.register_buffer("slow_window", slow_window, persistent=False)
            # Pre-compute frequency bins for slow resolution
            self.slow_freqs: torch.Tensor
            slow_freqs = torch.fft.rfftfreq(slow_n_fft, d=1 / self.fs)
            self.register_buffer("slow_freqs", slow_freqs, persistent=False)

            # Fast resolution: 2s window for spindles, alpha, beta
            # - Moderate frequency resolution (0.5 Hz) - still sufficient for 4+ Hz bands
            # - ~15 STFT frames per 30s epoch with 50% overlap
            # - Better captures temporal dynamics (spindle bursts, alpha blocking)
            fast_n_fft = int(2.0 * self.fs)
            if fast_n_fft % 2 != 0:
                fast_n_fft += 1
            self.fast_n_fft = fast_n_fft
            self.fast_hop_length = fast_n_fft // 2
            self.fast_window: torch.Tensor
            fast_window = torch.hann_window(fast_n_fft)
            self.register_buffer("fast_window", fast_window, persistent=False)
            # Pre-compute frequency bins for fast resolution
            self.fast_freqs: torch.Tensor
            fast_freqs = torch.fft.rfftfreq(fast_n_fft, d=1 / self.fs)
            self.register_buffer("fast_freqs", fast_freqs, persistent=False)

            # Map bands to resolutions
            # - Slow resolution (5s): slow, fdelta (need precise low-frequency resolution)
            # - Fast resolution (2s): theta, alpha, sigma, beta (need temporal dynamics)
            self.slow_bands = {"slow", "fdelta"}
            self.fast_bands = {"theta", "alpha", "sigma", "beta"}

        # Define frequency bands (Hz) - matches YASA spectral bands
        self.bands = {
            "slow": (0.4, 1),  # Slow oscillations (N3 specific)
            "fdelta": (1, 4),  # Fast delta (deep sleep marker)
            "theta": (4, 7),  # Light sleep, drowsiness
            "alpha": (8, 13),  # Awake with eyes closed
            "sigma": (11, 16),  # Sleep spindles (stage N2)
            "beta": (13, 30),  # Awake, alert
        }

        # Pre-compute frequency bins for STFT
        self.freqs: torch.Tensor
        freqs = torch.fft.rfftfreq(self.n_fft, d=1 / self.fs)
        self.register_buffer("freqs", freqs, persistent=False)

        # Tensor buffers created in _register_band_indices
        self._indices_slow: torch.Tensor
        self._indices_fdelta: torch.Tensor
        self._indices_theta: torch.Tensor
        self._indices_alpha: torch.Tensor
        self._indices_sigma: torch.Tensor
        self._indices_beta: torch.Tensor
        self._indices_spindle: torch.Tensor
        self._indices_delta: torch.Tensor
        self._indices_sem: torch.Tensor
        self._indices_muscle: torch.Tensor
        self._indices_vertex: torch.Tensor
        self._indices_kcomplex: torch.Tensor
        self._indices_rem_eye: torch.Tensor

        # Pre-compute band indices to avoid dynamic indexing in forward pass
        # This eliminates symbolic shape warnings from torch.compile
        self._register_band_indices()

        # Learnable band weights (allows model to reweight features)
        if learnable_bands:
            self.band_weights = nn.Parameter(torch.ones(len(self.bands)))
        else:
            self.register_buffer("band_weights", torch.ones(len(self.bands)))

        # Dynamic per-sample band weighting from EEG context.
        # The final layer is zero-initialized so initial behavior matches the
        # global prior until the model learns useful dynamic adjustments.
        num_bands = len(self.bands)
        hidden_dim = max(8, num_bands * 2)
        self.band_dynamic_mlp = nn.Sequential(
            nn.Linear(num_bands, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_bands),
        )
        final_linear = cast(nn.Linear, self.band_dynamic_mlp[-1])
        nn.init.zeros_(final_linear.weight)
        nn.init.zeros_(final_linear.bias)
        if not learnable_bands:
            for param in self.band_dynamic_mlp.parameters():
                param.requires_grad = False

        # Cached diagnostics from the most recent forward pass.
        self._last_band_weights: torch.Tensor | None = None

        # Output dimension: 6 bands * num_eeg_channels + presence + YASA features
        # Spectral (6 per EEG channel, matches YASA):
        #   - Slow power (0.4-1 Hz)  -> N3 slow oscillations
        #   - Fdelta power (1-4 Hz)  -> Deep sleep marker
        #   - Theta power (4-7 Hz)   -> N1, N2 marker
        #   - Alpha power (8-13 Hz)  -> Wake, N1 marker
        #   - Sigma power (11-16 Hz) -> N2 marker (spindles)
        #   - Beta power (13-30 Hz)  -> Wake, REM marker
        # NOTE: Band power features ONLY use EEG channels, not all channels

        # YASA feature configuration (deprecated - all computed on-the-fly now)
        self.use_precomputed = use_precomputed
        self.include_statistical = include_statistical
        self.include_hjorth = include_hjorth
        # NOTE: include_slow_band is deprecated - slow band is now included in main band_powers
        # which uses precomputed YASA spectral features directly
        self.include_slow_band = False  # Always False to avoid duplication
        self.include_extra_ratios = include_extra_ratios

        # Calculate YASA feature dimension (now computed on-the-fly)
        # Note: Band powers (slow, fdelta, theta, alpha, sigma, beta) are separate from YASA features
        # YASA features add: hjorth and ratios (statistical features deprecated)
        yasa_dim = 0
        if include_statistical:
            # DEPRECATED: statistical features are no longer used
            yasa_dim += (
                5 * self.num_eeg_channels
            )  # std, iqr, skew, kurt, nzc per EEG channel
        if include_hjorth:
            yasa_dim += (
                2 * self.num_eeg_channels
            )  # mobility, complexity per EEG channel
        # Slow band removed - now included in main band_powers
        if include_extra_ratios:
            yasa_dim += 4  # delta/theta, delta/sigma, delta/beta, theta/alpha

        self.yasa_feature_dim = yasa_dim

        # Lazy-load YASA torch extractor for real-time fallback (deprecated)
        self._yasa_torch_extractor = None

        # Output dimension:
        # 6 bands * num_eeg_channels (slow, fdelta, theta, alpha, sigma, beta)
        # + num_channels (Presence indicators)
        # + YASA features (hjorth, ratios - computed on-the-fly)
        self.out_dim = (
            len(self.bands) * self.num_eeg_channels
            + self.num_channels
            + self.yasa_feature_dim
        )

        # Engineered feature split metadata used by downstream fusion modules.
        self.core_feature_dim = self.out_dim - self.yasa_feature_dim
        self.yasa_start_idx = self.core_feature_dim

        # Register buffers for global feature statistics (subject-level normalization)
        # These are computed across the entire training set for stable normalization
        # across batches, rather than per-batch statistics which vary and cause instability
        # CRITICAL: persistent=True ensures these are saved in checkpoints for inference
        self.global_mean: torch.Tensor
        self.global_std: torch.Tensor
        self.stats_initialized: torch.Tensor
        self.register_buffer("global_mean", torch.zeros(self.out_dim), persistent=True)
        self.register_buffer("global_std", torch.ones(self.out_dim), persistent=True)
        self.register_buffer("stats_initialized", torch.tensor(False), persistent=True)

        # Separate statistics for N1 features (5 core features)
        # Features: theta/alpha ratio, alpha CV, theta emergence, vertex score, SEM score
        # CRITICAL: persistent=True ensures these are saved in checkpoints for inference
        self._n1_feature_dim = 5
        self.n1_global_mean: torch.Tensor
        self.n1_global_std: torch.Tensor
        self.n1_stats_initialized: torch.Tensor
        self.register_buffer(
            "n1_global_mean", torch.zeros(self._n1_feature_dim), persistent=True
        )
        self.register_buffer(
            "n1_global_std", torch.ones(self._n1_feature_dim), persistent=True
        )
        self.register_buffer(
            "n1_stats_initialized", torch.tensor(False), persistent=True
        )

        # Alpha power baseline statistics for tracking wake-state alpha power
        # Used for normalizing alpha-related features (e.g., alpha attenuation detection)
        # CRITICAL: persistent=True ensures these are saved in checkpoints for inference
        self.alpha_global_baseline: torch.Tensor
        self.alpha_global_count: torch.Tensor
        self.register_buffer("alpha_global_baseline", torch.zeros(1), persistent=True)
        self.register_buffer(
            "alpha_global_count", torch.zeros(1, dtype=torch.long), persistent=True
        )

        # === Fixed Detection Thresholds ===
        # Thresholds are fixed (non-learnable) to enforce deterministic behavior.
        # N1-related thresholds (fixed)
        self.register_buffer("alpha_dropout_thresh", torch.tensor(0.5), persistent=True)
        self.register_buffer(
            "vertex_amplitude_thresh", torch.tensor(2.0), persistent=True
        )
        self.register_buffer(
            "theta_emergence_thresh", torch.tensor(1.3), persistent=True
        )
        self.register_buffer(
            "alpha_sustained_thresh", torch.tensor(0.3), persistent=True
        )

        # Pre-compute SEM detection window sizes for extract_n1_features
        self._sem_window_samples = int(2.0 * self.fs)  # 2s envelope smoothing
        self._sem_peak_window_samples = int(4.0 * self.fs)  # 4s variance window

        # Clip threshold for robust scaling (matches preprocessing in batch_edf_to_zarrfp32.py)
        self._robust_scale_clip = 20.0

        # Flag to suppress worker process logging (set by preinitialize_statistics)
        self._suppress_init_logging = False

        # Controls whether running engineered/N1 statistics are updated during training.
        # When True, normalization keeps using the stored buffers but stops EMA updates.
        self._freeze_feature_stats_updates: bool = False

        # Temperature for differentiable soft-thresholding (sigmoid approximation).
        # Higher = sharper sigmoid (closer to hard threshold), but still differentiable.
        # 10.0 gives ~90% of hard threshold sharpness while allowing gradient flow.
        self.hard_threshold_inference: bool = True
        self._threshold_temp_start: float = 5.0
        self._threshold_temp_end: float = 40.0
        self._threshold_temperature: float = 10.0

    def _soft_threshold(
        self,
        signal: torch.Tensor,
        threshold: torch.Tensor,
        reference: torch.Tensor | None = None,
        *,
        greater: bool = True,
    ) -> torch.Tensor:
        """Differentiable soft thresholding using sigmoid approximation.

        During training, replaces ``(signal > threshold).float()`` with a
        smooth sigmoid so that gradients can flow through the learnable
        threshold parameters.  During eval, uses hard boolean for crisp
        feature values.

        Args:
            signal: Values to threshold.
            threshold: Already-computed effective threshold (same shape or broadcastable).
            reference: Optional normalization reference for the sigmoid scale.
                       If None, uses ``threshold.abs().clamp(min=1e-6)``.
            greater: If True, sigmoid approximates ``signal > threshold``.
                     If False, sigmoid approximates ``signal < threshold``.
        """
        if self.hard_threshold_inference and not self.training:
            if greater:
                return (signal > threshold).to(signal.dtype)
            return (signal < threshold).to(signal.dtype)

        ref = (reference if reference is not None else threshold).abs().clamp(min=1e-6)
        if greater:
            logits = self._threshold_temperature * (signal - threshold) / ref
        else:
            logits = self._threshold_temperature * (threshold - signal) / ref
        return torch.sigmoid(logits)

    # -- Threshold mask API --------------------------------------------------

    def _threshold_mask(
        self,
        signal: torch.Tensor,
        raw_threshold: torch.Tensor,
        reference: torch.Tensor,
        *,
        greater: bool = True,
    ) -> torch.Tensor:
        """Compatibility wrapper for the legacy threshold-mask helper API.

        Multiplies *raw_threshold* by *reference* to get the effective threshold,
        then delegates to ``_soft_threshold``.
        """
        effective_thresh = raw_threshold * reference
        return self._soft_threshold(
            signal, effective_thresh, reference=reference, greater=greater
        )

    def set_threshold_temperature(self, temp: float) -> None:
        """Set sigmoid temperature used for differentiable threshold masks."""
        self._threshold_temperature = max(float(temp), 1e-3)

    def set_threshold_anneal_progress(self, progress: float) -> None:
        """Linearly anneal threshold temperature from start to end."""
        p = min(max(float(progress), 0.0), 1.0)
        temp = self._threshold_temp_start + p * (
            self._threshold_temp_end - self._threshold_temp_start
        )
        self.set_threshold_temperature(temp)

    def set_feature_stats_frozen(self, frozen: bool) -> None:
        """Enable or disable updates to running engineered/N1 statistics."""
        self._freeze_feature_stats_updates = bool(frozen)

    def feature_stats_frozen(self) -> bool:
        """Return whether running engineered/N1 statistics are frozen."""
        return bool(self._freeze_feature_stats_updates)

    def get_learned_thresholds(self) -> dict[str, float]:
        """Return current fixed threshold values.

        Method name is kept for backward compatibility.

        Returns:
            Dict mapping threshold names to their current values.
        """
        alpha_dropout_thresh = cast(torch.Tensor, self.alpha_dropout_thresh)
        vertex_amplitude_thresh = cast(torch.Tensor, self.vertex_amplitude_thresh)
        theta_emergence_thresh = cast(torch.Tensor, self.theta_emergence_thresh)
        alpha_sustained_thresh = cast(torch.Tensor, self.alpha_sustained_thresh)
        return {
            "alpha_dropout": float(alpha_dropout_thresh),
            "vertex_amplitude": float(vertex_amplitude_thresh),
            "theta_emergence": float(theta_emergence_thresh),
            "alpha_sustained": float(alpha_sustained_thresh),
        }

    def get_feature_names(self) -> list[str]:
        """Return list of feature names in output order.

        Useful for debugging to identify which features are dead/problematic.
        """
        names = []
        band_names = list(self.bands.keys())  # slow, fdelta, theta, alpha, sigma, beta

        # 1. Band powers: [num_eeg * 6]
        for eeg_idx in range(self.num_eeg_channels):
            for band in band_names:
                names.append(f"eeg{eeg_idx}_{band}_power")

        # 2. Presence indicators: [num_channels]
        if self.channel_names:
            for ch in self.channel_names:
                names.append(f"presence_{ch}")
        else:
            for i in range(self.num_channels):
                names.append(f"presence_ch{i}")

        # 6. Hjorth parameters (if enabled): [num_eeg * 2]
        if self.include_hjorth:
            for eeg_idx in range(self.num_eeg_channels):
                names.append(f"eeg{eeg_idx}_hjorth_mobility")
                names.append(f"eeg{eeg_idx}_hjorth_complexity")

        # 4. Band ratios (if enabled): [4]
        if self.include_extra_ratios:
            ratio_names = [
                "ratio_delta_theta",
                "ratio_delta_sigma",
                "ratio_delta_beta",
                "ratio_theta_alpha",
            ]
            names.extend(ratio_names)

        # N3 criterion features removed — caused N3 over-recall.

        return names

    @torch.no_grad()
    def preinitialize_statistics(
        self,
        sample_batches: list[torch.Tensor] | None = None,
        dataloader: torch.utils.data.DataLoader | None = None,
        n_batches: int = 10,
    ) -> None:
        """Pre-initialize feature statistics before training/inference.

        This should be called in the main process BEFORE creating DataLoader workers
        to ensure all workers start with initialized statistics. This prevents each
        worker from independently re-initializing statistics and logging.

        Args:
            sample_batches: List of sample tensors [B, C, T] to compute stats from.
            dataloader: Optional DataLoader to sample batches from (if sample_batches not provided).
            n_batches: Number of batches to use from dataloader (default: 10).

        Example:
            >>> extractor = SleepFeatureExtractor(fs=128, num_channels=3, normalize='standardize')
            >>> # Option 1: With sample data
            >>> sample = torch.randn(32, 3, 3840)  # [B, C, T]
            >>> extractor.preinitialize_statistics(sample_batches=[sample])
            >>>
            >>> # Option 2: With DataLoader
            >>> extractor.preinitialize_statistics(dataloader=train_loader, n_batches=10)
            >>>
            >>> # Now safe to use in DataLoader workers - all workers start initialized
        """
        import logging

        logger = logging.getLogger(__name__)

        if self.stats_initialized.item():
            logger.info("[PRE-INIT] Feature statistics already initialized, skipping.")
            return

        # Collect sample data
        all_features = []
        was_training = self.training
        self.train()  # Ensure we're in training mode for stats computation

        if sample_batches is not None:
            for batch in sample_batches:
                # Run forward pass to extract features
                features = self._extract_features_no_normalize(batch)
                all_features.append(features)
        elif dataloader is not None:
            batch_count = 0
            for batch in dataloader:
                x: torch.Tensor | None = None
                # Handle different batch formats
                if isinstance(batch, dict):
                    if "signals" in batch:
                        x = batch["signals"]
                    elif "epochs" in batch:
                        x = batch["epochs"]
                    elif "x" in batch:
                        x = batch["x"]
                    else:
                        # Try first tensor value
                        for v in batch.values():
                            if isinstance(v, torch.Tensor) and v.dim() >= 2:
                                x = v
                                break
                elif isinstance(batch, (tuple, list)):
                    x = batch[0]
                    if isinstance(x, dict):
                        x = x.get("wave", x.get("signals", x.get("x")))
                else:
                    x = batch

                if x is None:
                    continue

                # Handle sequence dimension [B, T, C, L] -> [B*T, C, L]
                if x.dim() == 4:
                    B, T, C, L = x.shape
                    x = x.view(B * T, C, L)

                # Move to same device as model
                x = x.to(next(self.parameters()).device)

                features = self._extract_features_no_normalize(x)
                all_features.append(features)

                batch_count += 1
                if batch_count >= n_batches:
                    break
        else:
            logger.warning(
                "[PRE-INIT] No sample data provided for pre-initialization. "
                "Stats will be initialized from first training batch."
            )
            if was_training:
                self.train()
            else:
                self.eval()
            return

        if not all_features:
            logger.warning(
                "[PRE-INIT] No valid features extracted. "
                "Stats will be initialized from first training batch."
            )
            if was_training:
                self.train()
            else:
                self.eval()
            return

        # Concatenate all features and compute statistics
        all_features = torch.cat(all_features, dim=0)
        batch_mean = all_features.mean(dim=0)
        batch_std = all_features.std(dim=0, unbiased=False).clamp(min=1e-3)

        # Set the global statistics
        self.global_mean.copy_(batch_mean.to(dtype=self.global_mean.dtype))
        self.global_std.copy_(batch_std.to(dtype=self.global_std.dtype))
        self.stats_initialized.fill_(True)

        # Suppress future initialization logging in workers
        self._suppress_init_logging = True

        logger.info(
            f"[PRE-INIT] Initialized feature statistics from {len(all_features)} samples: "
            f"mean range=[{self.global_mean.min().item():.3f}, {self.global_mean.max().item():.3f}], "
            f"std range=[{self.global_std.min().item():.3f}, {self.global_std.max().item():.3f}]"
        )

        # Restore original training mode
        if was_training:
            self.train()
        else:
            self.eval()

    def _extract_features_no_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Extract raw features without normalization (for statistics computation).

        Args:
            x: Input tensor [B, C, T]

        Returns:
            Raw features [B, F] before normalization
        """
        # This is a simplified version that extracts features without applying normalization
        # We'll need to call the core feature extraction logic
        # For now, we use a temporary flag to skip normalization
        original_normalize = self.normalize
        self.normalize = "none"

        try:
            # Forward pass with normalization disabled
            features = self.forward(x)
        finally:
            self.normalize = original_normalize

        return features

    def _robust_scale_and_clip(
        self, x: torch.Tensor, dim: int = 0, clip_threshold: float | None = None
    ) -> torch.Tensor:
        """Apply robust scaling (median/IQR normalization) with clipping.

        This matches the normalization used in batch_edf_to_zarrfp32.py for
        precomputed features. Applied per-feature across the batch dimension.

        Args:
            x: Input tensor
            dim: Dimension to compute statistics over (default: 0 = batch)
            clip_threshold: Clip normalized values to ±threshold (default: self._robust_scale_clip = 20.0)

        Returns:
            Normalized and clipped tensor
        """
        if clip_threshold is None:
            clip_threshold = self._robust_scale_clip

        # Compute median and IQR across the specified dimension
        median = torch.median(x, dim=dim, keepdim=True).values
        q75 = torch.quantile(x, 0.75, dim=dim, keepdim=True)
        q25 = torch.quantile(x, 0.25, dim=dim, keepdim=True)

        # CRITICAL FIX: Use much more conservative IQR minimum (0.1 instead of 1e-6)
        # With constant features (all identical), q75 == q25, so iqr ≈ min value
        # Using 1e-6 caused division to produce values of order 1e6 → overflow
        iqr = (q75 - q25).clamp(min=0.1)

        # Normalize: (x - median) / IQR
        normalized = (x - median) / iqr

        # Clip to ±threshold
        clipped = torch.clamp(normalized, -clip_threshold, clip_threshold)

        # Sanitize any remaining NaN/Inf
        return torch.nan_to_num(
            clipped, nan=0.0, posinf=clip_threshold, neginf=-clip_threshold
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """
        Handle backward compatibility for checkpoints with different feature dimensions.

        Handles BOTH directions:
        1. Older checkpoints (fewer features) loading into this model -> pad with defaults
        2. Newer checkpoints (more features, e.g., 10 temporal) loading into this model -> truncate

        Also handles missing/extra buffers/parameters:
        - alpha_global_baseline, alpha_global_count (for alpha power tracking)
        - n1_theta_alpha_center, n1_ratio_bandwidth (for N1 detection)
        - Newer checkpoints may be missing these learnable params (they were removed)
        """
        # Handle resizing of existing buffers (global_mean, global_std)
        for key in ["global_mean", "global_std"]:
            full_key = prefix + key
            if full_key in state_dict:
                ckpt_tensor = state_dict[full_key]
                current_tensor = getattr(self, key)

                if ckpt_tensor.shape != current_tensor.shape:
                    # Check if it's just a size mismatch in the first dimension
                    if ckpt_tensor.ndim == 1 and current_tensor.ndim == 1:
                        if ckpt_tensor.shape[0] < current_tensor.shape[0]:
                            # Checkpoint has fewer features -> Pad with defaults
                            # For mean, pad with 0. For std, pad with 1.
                            pad_size = current_tensor.shape[0] - ckpt_tensor.shape[0]

                            if key == "global_mean":
                                padding = torch.zeros(
                                    pad_size,
                                    device=ckpt_tensor.device,
                                    dtype=ckpt_tensor.dtype,
                                )
                            else:  # global_std
                                padding = torch.ones(
                                    pad_size,
                                    device=ckpt_tensor.device,
                                    dtype=ckpt_tensor.dtype,
                                )

                            new_tensor = torch.cat([ckpt_tensor, padding])
                            state_dict[full_key] = new_tensor

                            if not torch.compiler.is_compiling():
                                import logging

                                logger = logging.getLogger(__name__)
                                logger.warning(
                                    f"Resized checkpoint buffer '{full_key}' from {ckpt_tensor.shape} "
                                    f"to {current_tensor.shape} (padded with defaults) for backward compatibility."
                                )
                        else:
                            # Checkpoint has MORE features (e.g., from newer 10-feature version)
                            # Truncate to match current model's expected size
                            state_dict[full_key] = ckpt_tensor[
                                : current_tensor.shape[0]
                            ]
                            if not torch.compiler.is_compiling():
                                import logging

                                logger = logging.getLogger(__name__)
                                logger.warning(
                                    f"Truncated checkpoint buffer '{full_key}' from {ckpt_tensor.shape} "
                                    f"to {current_tensor.shape} (newer checkpoint had more features)."
                                )

        # Handle resizing of N1 feature buffers
        for key in ["n1_global_mean", "n1_global_std"]:
            full_key = prefix + key
            if full_key in state_dict:
                ckpt_tensor = state_dict[full_key]
                current_tensor = getattr(self, key, None)
                if (
                    current_tensor is not None
                    and ckpt_tensor.shape != current_tensor.shape
                ):
                    if ckpt_tensor.ndim == 1 and current_tensor.ndim == 1:
                        if ckpt_tensor.shape[0] < current_tensor.shape[0]:
                            pad_size = current_tensor.shape[0] - ckpt_tensor.shape[0]
                            if key == "n1_global_mean":
                                padding = torch.zeros(
                                    pad_size,
                                    device=ckpt_tensor.device,
                                    dtype=ckpt_tensor.dtype,
                                )
                            else:
                                padding = torch.ones(
                                    pad_size,
                                    device=ckpt_tensor.device,
                                    dtype=ckpt_tensor.dtype,
                                )
                            state_dict[full_key] = torch.cat([ckpt_tensor, padding])
                        else:
                            state_dict[full_key] = ckpt_tensor[
                                : current_tensor.shape[0]
                            ]
                        if not torch.compiler.is_compiling():
                            import logging

                            logger = logging.getLogger(__name__)
                            logger.info(
                                f"[BACKWARD COMPAT] Resized N1 buffer '{key}' from {ckpt_tensor.shape} "
                                f"to {current_tensor.shape}"
                            )

        # Handle missing buffers/parameters for older checkpoints OR newer checkpoints
        # that removed these learnable parameters.
        default_values = {
            "alpha_global_baseline": torch.zeros(1),
            "alpha_global_count": torch.zeros(1, dtype=torch.long),
        }

        # Backward compatibility for checkpoints without dynamic band weighting MLP.
        for key, value in self.band_dynamic_mlp.state_dict().items():
            full_key = prefix + f"band_dynamic_mlp.{key}"
            if full_key not in state_dict:
                state_dict[full_key] = value.detach().clone()

        for key, default_value in default_values.items():
            full_key = prefix + key
            if full_key not in state_dict:
                # Add missing parameter/buffer with default value
                state_dict[full_key] = default_value
                if not torch.compiler.is_compiling():
                    import logging

                    logger = logging.getLogger(__name__)
                    logger.info(
                        f"[BACKWARD COMPAT] Added missing '{key}' with default value "
                        f"(checkpoint missing this parameter - either older or newer version)"
                    )

        # Thresholds are fixed constants in the current extractor implementation.
        # Force checkpoint values to current defaults so old learned/raw values
        # do not leak into inference/training behavior.
        fixed_threshold_values = {
            "alpha_dropout_thresh": torch.tensor(0.5),
            "vertex_amplitude_thresh": torch.tensor(2.0),
            "theta_emergence_thresh": torch.tensor(1.3),
            "alpha_sustained_thresh": torch.tensor(0.3),
        }
        for key, default_value in fixed_threshold_values.items():
            full_key = prefix + key
            state_dict[full_key] = default_value

        # Pop removed buffers so old checkpoints load cleanly under strict=True.
        _removed_keys = [
            "slow_wave_thresh",
            "delta_high_thresh",
            "delta_burst_thresh",
            "spindle_power_thresh",
            "kcomplex_amplitude_thresh",
            "rem_eog_variance_thresh",
            "emg_atonia_thresh",
            "emg_high_thresh",
        ]
        for key in _removed_keys:
            state_dict.pop(prefix + key, None)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply normalization to features based on self.normalize strategy.

        Uses subject-level (global) statistics computed across the entire training set
        for stable normalization, rather than per-batch statistics which vary and cause
        instability across batches.

        Args:
            features: [B, F] feature tensor

        Returns:
            [B, F] normalized features
        """
        if self.normalize == "none":
            return features

        elif self.normalize == "standardize":
            # SUBJECT-LEVEL NORMALIZATION: Use global statistics for stable normalization
            # Initialize global stats on first batch (training mode only)
            # Also check for default values and reinitialize if needed
            should_initialize = False
            reason = ""

            # CRITICAL: Use .item() to convert tensor to Python bool for proper comparison
            # This prevents issues with torch.compile tensor tracing and ensures correct
            # boolean logic when checking initialization status
            stats_init_flag = bool(self.stats_initialized.item())

            if self.training and not stats_init_flag:
                should_initialize = True
                reason = "first initialization"
            elif self.training and stats_init_flag:
                # Check for default values (mean=0, std=1) that indicate patched checkpoint
                mean_is_default = self.global_mean.abs().max().item() < 1e-6
                std_is_default = (self.global_std - 1.0).abs().max().item() < 1e-6

                if mean_is_default or std_is_default:
                    should_initialize = True
                    reason = "detected default values from patched checkpoint"
            elif not self.training and not stats_init_flag:
                # CRITICAL FIX: Initialize stats from eval batch instead of using terrible defaults
                # This ensures feature normalization is consistent with the actual data distribution
                features_clean = torch.nan_to_num(
                    features, nan=0.0, posinf=0.0, neginf=0.0
                )
                stats_source = features_clean.detach().to(dtype=torch.float32)
                batch_mean = stats_source.mean(dim=0)
                batch_std = stats_source.std(dim=0, unbiased=False)
                batch_std = torch.nan_to_num(batch_std, nan=1.0, posinf=1.0, neginf=1.0)
                batch_std = torch.clamp(batch_std, min=1e-3)
                with torch.no_grad():
                    self.global_mean.copy_(batch_mean.to(dtype=self.global_mean.dtype))
                    self.global_std.copy_(batch_std.to(dtype=self.global_std.dtype))
                    self.stats_initialized.fill_(True)
                if not torch.compiler.is_compiling():
                    import logging

                    logger = logging.getLogger(__name__)
                    logger.info(
                        f"[EVAL] Initialized feature statistics from eval batch: "
                        f"mean range=[{self.global_mean.min().item():.3f}, {self.global_mean.max().item():.3f}], "
                        f"std range=[{self.global_std.min().item():.3f}, {self.global_std.max().item():.3f}]"
                    )

            should_update_running_stats = self.training and (
                (not self._freeze_feature_stats_updates) or (not stats_init_flag)
            )

            if should_update_running_stats:
                # Use running statistics (exponential moving average) like BatchNorm
                # This prevents feature drift and handles distribution shifts during training
                # CRITICAL: Ensure features are finite before computing statistics
                features_clean = torch.nan_to_num(
                    features, nan=0.0, posinf=0.0, neginf=0.0
                )
                # Always compute statistics in float32 to avoid AMP underflow issues
                stats_source = features_clean.detach().to(dtype=torch.float32)
                # unbiased=False prevents NaN when batch has a single valid sample
                batch_mean = stats_source.mean(dim=0)
                batch_std = stats_source.std(dim=0, unbiased=False)

                # Distributed synchronization: Average stats across all processes
                if dist.is_available() and dist.is_initialized():
                    world_size = dist.get_world_size()
                    if world_size > 1:
                        # Stack for single all_reduce call
                        stats_tensor = torch.cat([batch_mean, batch_std])
                        dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)
                        stats_tensor /= world_size

                        # Unpack
                        batch_mean = stats_tensor[: self.out_dim]
                        batch_std = stats_tensor[self.out_dim :]

                # Safety check: ensure statistics are finite
                batch_mean = torch.nan_to_num(
                    batch_mean, nan=0.0, posinf=0.0, neginf=0.0
                )
                batch_std = torch.nan_to_num(batch_std, nan=1.0, posinf=1.0, neginf=1.0)
                batch_std = torch.clamp(batch_std, min=1e-3)  # Ensure non-zero

                # Update running statistics with exponential moving average
                # momentum=0.1 means 90% old stats + 10% new batch stats
                momentum = 0.1
                with torch.no_grad():
                    if should_initialize:
                        # First batch: initialize with batch statistics
                        self.global_mean.copy_(
                            batch_mean.to(dtype=self.global_mean.dtype)
                        )
                        self.global_std.copy_(batch_std.to(dtype=self.global_std.dtype))
                        self.stats_initialized.fill_(True)
                        # Log initialization for debugging (suppress in worker processes)
                        if not torch.compiler.is_compiling() and not getattr(
                            self, "_suppress_init_logging", False
                        ):
                            import logging

                            logger = logging.getLogger(__name__)
                            logger.info(
                                f"[TRAINING] Initialized engineered feature statistics from first batch ({reason}): "
                                f"mean range=[{self.global_mean.min().item():.3f}, {self.global_mean.max().item():.3f}], "
                                f"std range=[{self.global_std.min().item():.3f}, {self.global_std.max().item():.3f}]"
                            )
                    else:
                        # Update running statistics (exponential moving average)
                        self.global_mean.mul_(1 - momentum).add_(
                            batch_mean.to(dtype=self.global_mean.dtype) * momentum
                        )
                        self.global_std.mul_(1 - momentum).add_(
                            batch_std.to(dtype=self.global_std.dtype) * momentum
                        )
            elif not self.training and stats_init_flag:
                # INFERENCE MODE: Using pre-computed statistics from training
                # Log once per inference run for verification (suppress in workers)
                if not hasattr(self, "_inference_stats_logged"):
                    self._inference_stats_logged = True
                    if not torch.compiler.is_compiling() and not getattr(
                        self, "_suppress_init_logging", False
                    ):
                        import logging

                        logger = logging.getLogger(__name__)
                        logger.info(
                            f"[INFERENCE] Using pre-computed engineered feature statistics from training: "
                            f"mean range=[{self.global_mean.min().item():.3f}, {self.global_mean.max().item():.3f}], "
                            f"std range=[{self.global_std.min().item():.3f}, {self.global_std.max().item():.3f}]"
                        )

            # Check for NaN/Inf in input features before normalization
            if torch.isnan(features).any() or torch.isinf(features).any():
                # Replace NaN/Inf with zeros to prevent propagation
                features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

            # Use global stats (consistent across batches)
            # This prevents feature drift and provides stable training dynamics
            # CRITICAL: Ensure global_std is never zero to prevent NaN from division
            safe_std = torch.clamp(self.global_std, min=1e-3)
            normalized = (features - self.global_mean) / safe_std

            # Clamp to prevent extreme outliers while preserving gradient flow
            # [-5, 5] range balances outlier protection with gradient preservation
            # With running statistics, this should now be more stable
            normalized = torch.clamp(normalized, min=-5.0, max=5.0)

            # Track clamp rate for monitoring feature dominance
            # Features that frequently clip at ±5 carry disproportionate weight
            if should_update_running_stats and not torch.compiler.is_compiling():
                with torch.no_grad():
                    clamp_rate = (normalized.abs() >= 4.99).float().mean().item()
                    per_feat_clamp = (normalized.abs() >= 4.99).float().mean(dim=0)
                    if not hasattr(self, "_clamp_rate_ema"):
                        self._clamp_rate_ema = clamp_rate
                        self._per_feat_clamp_rate = per_feat_clamp
                    else:
                        self._clamp_rate_ema = (
                            0.9 * self._clamp_rate_ema + 0.1 * clamp_rate
                        )
                        self._per_feat_clamp_rate = (
                            0.9 * self._per_feat_clamp_rate + 0.1 * per_feat_clamp
                        )

            # Final NaN check after normalization (should rarely trigger with proper preprocessing)
            normalized = torch.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)

            return normalized

        elif self.normalize == "minmax":
            # Min-max scaling to [0, 1]
            # Compute per-feature min/max across batch
            # MEMORY FIX: Detach statistics to prevent gradient graph retention
            min_vals = features.min(dim=0, keepdim=True)[0].detach()
            max_vals = features.max(dim=0, keepdim=True)[0].detach()
            # Avoid division by zero (constant features)
            range_vals = max_vals - min_vals
            range_vals = torch.clamp(range_vals, min=1e-3)  # More aggressive clamping
            normalized = (features - min_vals) / range_vals
            # Clamp to ensure it stays in [0, 1]
            normalized = torch.clamp(normalized, min=0.0, max=1.0)
            # NaN check
            normalized = torch.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
            return normalized

        return features

    def _normalize_n1_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply normalization specifically to N1 features (5-dimensional).

        Uses separate statistics buffers (n1_global_mean, n1_global_std) sized for
        N1 features to avoid dimension mismatch with full feature statistics.

        Args:
            features: [B, 5] N1 feature tensor

        Returns:
            [B, 5] normalized N1 features
        """
        if self.normalize == "none":
            return features

        elif self.normalize == "standardize":
            # Initialize N1-specific stats on first batch (training mode only)
            # Also check for default values and reinitialize if needed
            should_initialize = False
            reason = ""

            # CRITICAL: Use .item() to convert tensor to Python bool for proper comparison
            n1_stats_init_flag = bool(self.n1_stats_initialized.item())

            if self.training and not n1_stats_init_flag:
                should_initialize = True
                reason = "first initialization"
            elif self.training and n1_stats_init_flag:
                # Check for default values (mean=0, std=1) that indicate patched checkpoint
                mean_is_default = self.n1_global_mean.abs().max().item() < 1e-6
                std_is_default = (self.n1_global_std - 1.0).abs().max().item() < 1e-6

                if mean_is_default or std_is_default:
                    should_initialize = True
                    reason = "detected default values from patched checkpoint"

            should_update_running_stats = self.training and (
                (not self._freeze_feature_stats_updates) or (not n1_stats_init_flag)
            )

            if should_update_running_stats:
                # Use running statistics (exponential moving average) like BatchNorm
                # CRITICAL: Ensure features are finite before computing statistics
                features_clean = torch.nan_to_num(
                    features, nan=0.0, posinf=0.0, neginf=0.0
                )
                # Always compute statistics in float32 to avoid AMP underflow issues
                stats_source = features_clean.detach().to(dtype=torch.float32)
                batch_mean = stats_source.mean(dim=0)
                # FIX: Use unbiased=False to prevent NaN when batch_size=1 (divides by n-1)
                batch_std = stats_source.std(dim=0, unbiased=False)

                # Distributed synchronization: Average stats across all processes
                if dist.is_available() and dist.is_initialized():
                    world_size = dist.get_world_size()
                    if world_size > 1:
                        stats_tensor = torch.cat([batch_mean, batch_std])
                        dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)
                        stats_tensor /= world_size
                        batch_mean = stats_tensor[: self._n1_feature_dim]
                        batch_std = stats_tensor[self._n1_feature_dim :]

                # Safety check: ensure statistics are finite
                batch_mean = torch.nan_to_num(
                    batch_mean, nan=0.0, posinf=0.0, neginf=0.0
                )
                batch_std = torch.nan_to_num(batch_std, nan=1.0, posinf=1.0, neginf=1.0)
                batch_std = torch.clamp(batch_std, min=1e-3)  # Ensure non-zero

                # Update running statistics with exponential moving average
                # momentum=0.1 means 90% old stats + 10% new batch stats
                momentum = 0.1
                with torch.no_grad():
                    if should_initialize:
                        # First batch: initialize with batch statistics
                        self.n1_global_mean.copy_(
                            batch_mean.to(dtype=self.n1_global_mean.dtype)
                        )
                        self.n1_global_std.copy_(
                            batch_std.to(dtype=self.n1_global_std.dtype)
                        )
                        self.n1_stats_initialized.fill_(True)
                        # Log initialization (suppress in worker processes)
                        if not torch.compiler.is_compiling() and not getattr(
                            self, "_suppress_init_logging", False
                        ):
                            import logging

                            logger = logging.getLogger(__name__)
                            logger.info(
                                f"[TRAINING] Initialized N1 feature statistics from first batch ({reason}): "
                                f"mean range=[{self.n1_global_mean.min().item():.3f}, {self.n1_global_mean.max().item():.3f}], "
                                f"std range=[{self.n1_global_std.min().item():.3f}, {self.n1_global_std.max().item():.3f}]"
                            )
                    else:
                        # Update running statistics (exponential moving average)
                        self.n1_global_mean.mul_(1 - momentum).add_(
                            batch_mean.to(dtype=self.n1_global_mean.dtype) * momentum
                        )
                        self.n1_global_std.mul_(1 - momentum).add_(
                            batch_std.to(dtype=self.n1_global_std.dtype) * momentum
                        )
            elif not self.training and n1_stats_init_flag:
                # Log once for verification (suppress in workers)
                if not hasattr(self, "_n1_inference_stats_logged"):
                    self._n1_inference_stats_logged = True
                    if not torch.compiler.is_compiling() and not getattr(
                        self, "_suppress_init_logging", False
                    ):
                        import logging

                        logger = logging.getLogger(__name__)
                        logger.info(
                            f"[INFERENCE] Using pre-computed N1 feature statistics from training: "
                            f"mean range=[{self.n1_global_mean.min().item():.3f}, {self.n1_global_mean.max().item():.3f}], "
                            f"std range=[{self.n1_global_std.min().item():.3f}, {self.n1_global_std.max().item():.3f}]"
                        )

            # Check for NaN/Inf
            if torch.isnan(features).any() or torch.isinf(features).any():
                features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

            # Normalize using N1-specific stats
            # CRITICAL: Ensure n1_global_std is never zero to prevent NaN from division
            safe_std = torch.clamp(self.n1_global_std, min=1e-3)
            normalized = (features - self.n1_global_mean) / safe_std
            normalized = torch.clamp(normalized, min=-10.0, max=10.0)
            normalized = torch.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)

            return normalized

        elif self.normalize == "minmax":
            # Min-max scaling for N1 features
            min_vals = features.min(dim=0, keepdim=True)[0].detach()
            max_vals = features.max(dim=0, keepdim=True)[0].detach()
            range_vals = torch.clamp(max_vals - min_vals, min=1e-3)
            normalized = (features - min_vals) / range_vals
            normalized = torch.clamp(normalized, min=0.0, max=1.0)
            normalized = torch.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
            return normalized

        return features

    def _register_band_indices(self):
        """Pre-compute frequency bin indices for each band with Nyquist validation."""
        import warnings

        freqs = cast(torch.Tensor, self.freqs)
        nyquist = self.fs / 2

        for band_name, (low, high) in self.bands.items():
            # Validate band bounds against Nyquist frequency
            if high > nyquist:
                warnings.warn(
                    f"Band '{band_name}' upper bound ({high} Hz) exceeds Nyquist "
                    f"({nyquist} Hz) at fs={self.fs} Hz. Clamping to {nyquist} Hz.",
                    UserWarning,
                    stacklevel=2,
                )
                high = nyquist
            if low >= nyquist:
                warnings.warn(
                    f"Band '{band_name}' lower bound ({low} Hz) exceeds Nyquist "
                    f"({nyquist} Hz). This band will have no valid frequency bins.",
                    UserWarning,
                    stacklevel=2,
                )

            # Find frequency bins in range
            mask = (freqs >= low) & (freqs < high)
            indices = torch.where(mask)[0]
            # Store as buffer for device transfer
            self.register_buffer(f"_indices_{band_name}", indices, persistent=False)

        # Pre-compute indices for special features
        # Spindle detection (sigma range)
        spindle_mask = (freqs >= 11) & (freqs < 16)
        spindle_indices = torch.where(spindle_mask)[0]
        self.register_buffer("_indices_spindle", spindle_indices, persistent=False)

        # Delta band (0.5-4 Hz) for N3 discrimination
        delta_mask = (freqs >= 0.5) & (freqs < 4)
        delta_indices = torch.where(delta_mask)[0]
        self.register_buffer("_indices_delta", delta_indices, persistent=False)

        # Slow eye movement band (0.2 - 0.7 Hz)
        sem_mask = (freqs >= 0.2) & (freqs < 0.7)
        sem_indices = torch.where(sem_mask)[0]
        self.register_buffer("_indices_sem", sem_indices, persistent=False)

        # Muscle tone (high frequencies, limited to Nyquist)
        high_freq_limit = min(50, self.fs / 2 - 1)
        muscle_mask = (freqs >= 30) & (freqs < high_freq_limit)
        muscle_indices = torch.where(muscle_mask)[0]
        self.register_buffer("_indices_muscle", muscle_indices, persistent=False)

        # Vertex sharp waves (5-7 Hz, N1 marker)
        # NARROWED from 4-7 Hz to 5-7 Hz to avoid K-complex energy at 2-4 Hz
        # K-complexes have significant low-frequency components that bleed into 4-5 Hz
        # True vertex waves are brief sharp deflections in upper theta range, maximal at Cz
        vertex_mask = (freqs >= 5) & (freqs < 7)
        vertex_indices = torch.where(vertex_mask)[0]
        self.register_buffer("_indices_vertex", vertex_indices, persistent=False)

        # K-complex specific band (0.5-1.5 Hz) for N2 vs N3 discrimination
        # K-complexes have peak energy around 0.8-1.2 Hz, narrower than full delta band
        # Used to distinguish isolated K-complexes (N2) from continuous slow waves (N3)
        kcomplex_mask = (freqs >= 0.5) & (freqs < 1.5)
        kcomplex_indices = torch.where(kcomplex_mask)[0]
        self.register_buffer("_indices_kcomplex", kcomplex_indices, persistent=False)

        # Rapid eye movement band (1.5-4 Hz) - distinct from both SEM (0.2-0.7 Hz) and slow waves (0.5-2 Hz)
        # REM saccades are FASTER than slow waves - previous 0.5-2 Hz overlapped with delta!
        # This caused N3 slow waves to activate REM features, leading to N3→REM confusion
        # True REM eye movements have faster frequencies and sharper onset than slow rolling movements
        rem_eye_mask = (freqs >= 1.5) & (freqs < 4.0)
        rem_eye_indices = torch.where(rem_eye_mask)[0]
        self.register_buffer("_indices_rem_eye", rem_eye_indices, persistent=False)

        # Multi-resolution band indices (for use_multi_resolution mode)
        # Register indices for both slow and fast resolutions
        if self.use_multi_resolution:
            nyquist = self.fs / 2
            # Slow resolution indices (using slow_freqs from 5s window)
            slow_freqs = cast(torch.Tensor, self.slow_freqs)
            for band_name, (low, high) in self.bands.items():
                if band_name in self.slow_bands:
                    effective_high = min(high, nyquist)
                    mask = (slow_freqs >= low) & (slow_freqs < effective_high)
                    indices = torch.where(mask)[0]
                    self.register_buffer(
                        f"_slow_indices_{band_name}", indices, persistent=False
                    )

            # Fast resolution indices (using fast_freqs from 2s window)
            fast_freqs = cast(torch.Tensor, self.fast_freqs)
            for band_name, (low, high) in self.bands.items():
                if band_name in self.fast_bands:
                    effective_high = min(high, nyquist)
                    mask = (fast_freqs >= low) & (fast_freqs < effective_high)
                    indices = torch.where(mask)[0]
                    self.register_buffer(
                        f"_fast_indices_{band_name}", indices, persistent=False
                    )

    def _resolve_subject_ids(
        self,
        metadata: dict | None,
        batch_size: int,
    ) -> list[str] | None:
        """Extract subject identifiers from optional metadata."""
        if metadata is None:
            return None

        candidates = None
        if isinstance(metadata, dict):
            for key in ("subject_ids", "subject_id", "subjects", "subject"):
                if key in metadata:
                    candidates = metadata[key]
                    break
        if candidates is None:
            return None

        if isinstance(candidates, torch.Tensor):
            if candidates.ndim == 0:
                ids = [str(candidates.item())]
            elif candidates.ndim == 1:
                ids = [str(item) for item in candidates.tolist()]
            else:
                ids = [str(candidates.view(-1)[0].item())]
        elif isinstance(candidates, (list, tuple)):
            ids = []
            for item in candidates:
                if isinstance(item, torch.Tensor):
                    ids.append(str(item.item()))
                else:
                    ids.append(str(item))
        elif isinstance(candidates, str):
            ids = [candidates]
        else:
            ids = [str(candidates)]

        if not ids:
            return None
        if len(ids) == batch_size:
            return ids
        # If a single id provided, broadcast to batch
        if len(ids) == 1:
            return ids * batch_size
        # Fallback: truncate or pad with last id
        if len(ids) > batch_size:
            return ids[:batch_size]
        ids_extended = ids + [ids[-1]] * (batch_size - len(ids))
        return ids_extended

    @torch_compile_disable
    def compute_band_power(
        self, x: torch.Tensor, band_indices: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Compute power in frequency band using STFT.

        Args:
            x: [B, C, T] waveform
            band_indices: [n_indices] pre-computed frequency bin indices

        Returns:
            [B, C] power in band

        Note: Decorated with @torch_compile_disable to prevent backward pass
        compilation errors with complex64 gradients from STFT.
        """
        B, C, T = x.shape

        # Reshape for STFT: [B*C, T]
        x_flat = x.reshape(B * C, T)

        # CRITICAL NaN FIX: Sanitize input before STFT to prevent NaN propagation
        # Replace NaN/Inf with zeros to prevent STFT from producing invalid outputs
        x_flat = torch.nan_to_num(x_flat, nan=0.0, posinf=0.0, neginf=0.0)

        # CRITICAL NaN FIX: Clamp extreme values to prevent numerical instability
        # Extremely large values can cause overflow in FFT operations
        x_flat = torch.clamp(x_flat, min=-1e6, max=1e6)

        # STFT: [B*C, n_freqs, n_frames]
        # return_complex=True is more efficient than separate real/imag
        # CRITICAL: Use compile-safe wrapper to prevent torch.compile NaN bugs
        stft = _compile_safe_stft(
            x_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True,
            center=True,
            normalized=False,
            onesided=True,
        )

        # Power spectrum: [B*C, n_freqs, n_frames]
        power = torch.abs(stft) ** 2

        # Sanitize power spectrum: Even with clean input, abs(STFT)^2 can overflow to Inf
        # with very large (but finite) amplitude signals, especially with long FFT windows
        power = torch.nan_to_num(power, nan=0.0, posinf=0.0, neginf=0.0)

        # CRITICAL FIX: Clamp power to reasonable range to prevent downstream overflow
        # Lower bound: 1e-12 prevents log issues, upper bound: 1e10 prevents overflow
        power = torch.clamp(power, min=1e-12, max=1e10)

        # Average over time: [B*C, n_freqs]
        power_avg = power.mean(dim=-1)

        # Extract band using pre-computed indices
        if band_indices is not None and len(band_indices) > 0:
            # Use advanced indexing with pre-computed indices (compile-safe)
            band_power = power_avg[:, band_indices].mean(dim=-1)  # [B*C]
        else:
            # Fallback: no valid frequencies in band
            band_power = torch.zeros(B * C, device=x.device, dtype=x.dtype)

        # Reshape back: [B*C] -> [B, C]
        band_power = band_power.view(B, C)

        # Ensure positive with small epsilon for smooth gradient flow
        # Using min=1e-10 instead of min=0.0 avoids discontinuity at zero
        # log1p(0) = 0, log1p(1e-10) ≈ 1e-10, so gradients are smooth near zero
        band_power = torch.clamp(band_power, min=1e-10)

        # Log transform for stability (log1p is safe with positive input)
        band_power = torch.log1p(band_power)

        return band_power

    @torch_compile_disable
    def _bandpass_filter_stft(
        self,
        x: torch.Tensor,
        band_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Band-limit a waveform using pre-computed STFT indices.

        Args:
            x: [B, C, T] waveform
            band_indices: Tensor of frequency-bin indices to retain

        Returns:
            Band-limited signal with the same shape as input.
        """
        B, C, T = x.shape
        if band_indices is None or len(band_indices) == 0:
            return torch.zeros_like(x)

        x_flat = x.reshape(B * C, T)
        x_flat = torch.nan_to_num(x_flat, nan=0.0, posinf=0.0, neginf=0.0)
        x_flat = torch.clamp(x_flat, min=-1e6, max=1e6)

        stft = _compile_safe_stft(
            x_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True,
            center=True,
            normalized=False,
            onesided=True,
        )

        filtered = torch.zeros_like(stft)
        filtered[:, band_indices, :] = stft[:, band_indices, :]

        x_filtered = _compile_safe_istft(
            filtered,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=True,
            normalized=False,
            onesided=True,
            length=T,
        )

        return torch.nan_to_num(
            x_filtered.view(B, C, T), nan=0.0, posinf=0.0, neginf=0.0
        )

    @torch_compile_disable
    def compute_all_band_powers(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute all band powers efficiently using STFT.

        When use_multi_resolution=True, uses dual STFTs:
        - Slow resolution (5s window): 0.2 Hz resolution for slow/fdelta bands
        - Fast resolution (2s window): 0.5 Hz resolution for theta/alpha/sigma/beta

        This improves temporal dynamics capture for fast bands (~15 STFT frames
        vs ~6 frames per 30s epoch) while maintaining precise frequency resolution
        for slow oscillations.

        Args:
            x: [B, C, T] waveform

        Returns:
            [B, C, num_bands] power in all bands

        Note: Decorated with @torch_compile_disable to prevent backward pass
        compilation errors with complex64 gradients from STFT.
        """
        B, C, T = x.shape

        # Reshape for STFT: [B*C, T]
        x_flat = x.reshape(B * C, T)

        # CRITICAL NaN FIX: Sanitize input before STFT to prevent NaN propagation
        x_flat = torch.nan_to_num(x_flat, nan=0.0, posinf=0.0, neginf=0.0)
        x_flat = torch.clamp(x_flat, min=-1e6, max=1e6)

        num_bands = len(self.bands)
        band_powers_tensor = torch.zeros(
            B * C, num_bands, device=x.device, dtype=x.dtype
        )

        if self.use_multi_resolution:
            # Multi-resolution mode: compute separate STFTs for different frequency ranges
            # This gives better temporal resolution for fast bands while maintaining
            # precise frequency resolution for slow oscillations

            # === Slow resolution STFT (5s window) for slow and fdelta bands ===
            slow_stft = _compile_safe_stft(
                x_flat,
                n_fft=self.slow_n_fft,
                hop_length=self.slow_hop_length,
                window=self.slow_window,
                return_complex=True,
                center=True,
                normalized=False,
                onesided=True,
            )
            slow_power = torch.abs(slow_stft) ** 2
            slow_power = torch.nan_to_num(slow_power, nan=0.0, posinf=0.0, neginf=0.0)
            slow_power = torch.clamp(slow_power, min=1e-12, max=1e10)
            slow_power_avg = slow_power.mean(dim=-1)  # [B*C, n_slow_freqs]
            slow_freqs = cast(torch.Tensor, self.slow_freqs)

            # === Fast resolution STFT (2s window) for theta/alpha/sigma/beta ===
            fast_stft = _compile_safe_stft(
                x_flat,
                n_fft=self.fast_n_fft,
                hop_length=self.fast_hop_length,
                window=self.fast_window,
                return_complex=True,
                center=True,
                normalized=False,
                onesided=True,
            )
            fast_power = torch.abs(fast_stft) ** 2
            fast_power = torch.nan_to_num(fast_power, nan=0.0, posinf=0.0, neginf=0.0)
            fast_power = torch.clamp(fast_power, min=1e-12, max=1e10)
            fast_power_avg = fast_power.mean(dim=-1)  # [B*C, n_fast_freqs]
            fast_freqs = cast(torch.Tensor, self.fast_freqs)

            # Compute band powers using appropriate resolution
            for idx, band_name in enumerate(self.bands.keys()):
                if band_name in self.slow_bands:
                    # Use slow resolution for slow/fdelta
                    indices = getattr(self, f"_slow_indices_{band_name}")
                    if len(indices) > 0:
                        freqs_band = slow_freqs[indices]
                        band_powers_tensor[:, idx] = torch.trapezoid(
                            slow_power_avg[:, indices], freqs_band, dim=-1
                        )
                elif band_name in self.fast_bands:
                    # Use fast resolution for theta/alpha/sigma/beta
                    indices = getattr(self, f"_fast_indices_{band_name}")
                    if len(indices) > 0:
                        freqs_band = fast_freqs[indices]
                        band_powers_tensor[:, idx] = torch.trapezoid(
                            fast_power_avg[:, indices], freqs_band, dim=-1
                        )
                # else: already zero-initialized for unrecognized bands
        else:
            # Single resolution mode (original behavior)
            # STFT: [B*C, n_freqs, n_frames]
            stft = _compile_safe_stft(
                x_flat,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=self.window,
                return_complex=True,
                center=True,
                normalized=False,
                onesided=True,
            )

            # Power spectrum: [B*C, n_freqs, n_frames]
            power = torch.abs(stft) ** 2
            power = torch.nan_to_num(power, nan=0.0, posinf=0.0, neginf=0.0)
            power = torch.clamp(power, min=1e-12, max=1e10)
            freqs = cast(torch.Tensor, self.freqs)

            # Average over time: [B*C, n_freqs]
            power_avg = power.mean(dim=-1)

            # Compute band powers using pre-computed indices
            for idx, band_name in enumerate(self.bands.keys()):
                indices = getattr(self, f"_indices_{band_name}")
                if len(indices) > 0:
                    # Use trapezoidal integration (matches YASA, physically correct for PSD)
                    freqs_band = freqs[indices]
                    band_powers_tensor[:, idx] = torch.trapezoid(
                        power_avg[:, indices], freqs_band, dim=-1
                    )
                # else: already zero-initialized

        # Reshape: [B*C, num_bands] -> [B, C, num_bands]
        band_powers = band_powers_tensor.view(B, C, num_bands)

        # Ensure non-negative (power spectrum should be non-negative by definition)
        band_powers = torch.clamp(band_powers, min=0.0)

        # Log transform for stability (log1p is safe with non-negative input)
        band_powers = torch.log1p(band_powers)

        return band_powers

    def _compute_hjorth_on_the_fly(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Hjorth parameters (mobility, complexity) on-the-fly.

        Hjorth parameters characterize time-domain signal properties:
        - Mobility: Rate of change relative to signal amplitude
        - Complexity: Bandwidth of the signal relative to mobility

        These are useful complements to frequency-domain features.

        Args:
            x: [B, C, T] normalized waveforms

        Returns:
            [B, C, 2] Hjorth parameters: [mobility, complexity]
        """
        # First derivative: dx[t] = x[t+1] - x[t]
        dx = x[:, :, 1:] - x[:, :, :-1]  # [B, C, T-1]

        # Second derivative: ddx[t] = dx[t+1] - dx[t]
        ddx = dx[:, :, 1:] - dx[:, :, :-1]  # [B, C, T-2]

        # CRITICAL FIX: Use more aggressive variance clamping (1e-6 instead of adding epsilon)
        # For flat/constant signals, variance can be extremely small, causing division to explode
        var_x = torch.clamp(x.var(dim=-1), min=1e-6)  # [B, C]
        var_dx = torch.clamp(dx.var(dim=-1), min=1e-6)  # [B, C]
        var_ddx = torch.clamp(ddx.var(dim=-1), min=1e-6)  # [B, C]

        # Mobility = sqrt(var(dx) / var(x))
        mobility = torch.sqrt(var_dx / var_x)
        # CRITICAL: Clamp mobility to physiologically reasonable range
        mobility = torch.clamp(mobility, min=1e-4, max=1e4)

        # Mobility of derivative
        mobility_dx = torch.sqrt(var_ddx / var_dx)
        mobility_dx = torch.clamp(mobility_dx, min=1e-4, max=1e4)

        # Complexity = mobility(dx) / mobility(x)
        # Use conservative minimum for mobility divisor
        complexity = mobility_dx / torch.clamp(mobility, min=1e-3)
        # Clamp to physiologically reasonable range (typical values 1-5)
        complexity = torch.clamp(complexity, min=0.0, max=10.0)

        result = torch.stack([mobility, complexity], dim=-1)  # [B, C, 2]
        result = torch.nan_to_num(result, nan=0.0, posinf=10.0, neginf=0.0)

        return result

    def _compute_ratios_on_the_fly(self, band_powers: torch.Tensor) -> torch.Tensor:
        """Compute band power ratios on-the-fly.

        Ratios capture relative power between bands, which is often more
        informative than absolute power levels.

        Args:
            band_powers: [B, num_eeg, 6] - slow, fdelta, theta, alpha, sigma, beta

        Returns:
            [B, 4] ratios: delta/theta, delta/sigma, delta/beta, theta/alpha
        """
        # Use first EEG channel
        bp = band_powers[:, 0, :]  # [B, 6]

        slow = bp[:, 0]
        fdelta = bp[:, 1]
        theta = bp[:, 2]
        alpha = bp[:, 3]
        sigma = bp[:, 4]
        beta = bp[:, 5]

        # Combined delta (average of slow and fdelta)
        delta = (slow + fdelta) / 2

        # Use log difference instead of ratio to avoid division issues
        # log(a/b) = log(a) - log(b)
        eps = 1e-3  # Conservative epsilon for log-transformed band powers

        # Clamp to ensure positive values for log computation
        delta_safe = torch.clamp(delta, min=eps)
        theta_safe = torch.clamp(theta, min=eps)
        alpha_safe = torch.clamp(alpha, min=eps)
        sigma_safe = torch.clamp(sigma, min=eps)
        beta_safe = torch.clamp(beta, min=eps)

        # Compute log ratios (more numerically stable than direct division)
        delta_theta = torch.log(delta_safe) - torch.log(theta_safe)
        delta_sigma = torch.log(delta_safe) - torch.log(sigma_safe)
        delta_beta = torch.log(delta_safe) - torch.log(beta_safe)
        theta_alpha = torch.log(theta_safe) - torch.log(alpha_safe)

        # Clamp to reasonable range (log ratios typically -5 to +5)
        delta_theta = torch.clamp(delta_theta, min=-10.0, max=10.0)
        delta_sigma = torch.clamp(delta_sigma, min=-10.0, max=10.0)
        delta_beta = torch.clamp(delta_beta, min=-10.0, max=10.0)
        theta_alpha = torch.clamp(theta_alpha, min=-10.0, max=10.0)

        ratios = torch.stack(
            [delta_theta, delta_sigma, delta_beta, theta_alpha],
            dim=-1,
        )
        ratios = torch.nan_to_num(ratios, nan=0.0, posinf=10.0, neginf=-10.0)

        return ratios

    def _design_bandpass_kernel(
        self, low_hz: float, high_hz: float, num_taps: int = 51
    ) -> torch.Tensor:
        """
        Design a simple FIR bandpass kernel using windowed sinc method.

        Args:
            low_hz: Low cutoff frequency
            high_hz: High cutoff frequency
            num_taps: Length of filter kernel (should be odd)

        Returns:
            [1, 1, num_taps] kernel tensor
        """
        # Nyquist frequency
        nyquist = 0.5 * self.fs

        # Normalize frequencies
        low = low_hz / nyquist
        high = high_hz / nyquist

        # Create time vector centered at 0
        window_device = cast(torch.Tensor, self.window).device
        t = (
            torch.arange(num_taps, dtype=torch.float32, device=window_device)
            - (num_taps - 1) / 2
        )

        # Sinc function: sin(pi*x)/(pi*x)
        # Lowpass at high_hz
        # Avoid division by zero at t=0
        t_non_zero = t.clone()
        t_non_zero[t == 0] = 1.0e-9

        # Bandpass = Lowpass(high) - Lowpass(low)
        # lp_high = high * sinc(high * t)
        h_high = (
            high * torch.sin(np.pi * high * t_non_zero) / (np.pi * high * t_non_zero)
        )
        h_high[t == 0] = high

        # lp_low = low * sinc(low * t)
        h_low = low * torch.sin(np.pi * low * t_non_zero) / (np.pi * low * t_non_zero)
        h_low[t == 0] = low

        # Bandpass impulse response
        h = h_high - h_low

        # Apply Hamming window
        window = torch.hamming_window(num_taps, device=window_device)
        h = h * window

        # Normalize to unit gain at center frequency
        denom = h.sum()
        denom = torch.where(denom.abs() < 1e-8, torch.full_like(denom, 1e-8), denom)
        h = h / denom

        return h.view(1, 1, -1)

    def _bandpass_filter_time_domain(
        self, x: torch.Tensor, low_hz: float, high_hz: float, num_taps: int = 101
    ) -> torch.Tensor:
        """
        Apply time-domain bandpass filter using Conv1d.
        Faster and more stable than STFT-based filtering for simple envelopes.

        Args:
            x: [B, C, T] waveform
            low_hz, high_hz: Passband frequencies
            num_taps: Filter length

        Returns:
            Filtered waveform [B, C, T]
        """
        B, C, T = x.shape

        # Design kernel (cached if possible, but fast enough to compute)
        kernel = self._design_bandpass_kernel(low_hz, high_hz, num_taps)
        kernel = kernel.to(x.device).type(x.dtype)

        # Reshape input for depthwise conv: [B*C, 1, T]
        x_reshaped = x.reshape(B * C, 1, T)

        # Pad to maintain length
        padding = num_taps // 2

        filtered = F.conv1d(x_reshaped, kernel, padding=padding)
        return filtered.view(B, C, T)

    def detect_vertex_sharp_waves(self, x: torch.Tensor) -> torch.Tensor:
        """
        Detect vertex sharp waves (VSW) using learned threshold.

        VSW are characteristic of N1/N2 - negative sharp waves, usually < 200ms.
        Uses 2-8 Hz band and second derivative for sharpness detection.

        Args:
            x: [B, C, T] waveform (expects EEG channels)

        Returns:
            [B, C, 3] features:
                - vertex_density: Fraction of samples above threshold
                - peak_ratio: Max sharpness relative to mean
                - event_count: Count of distinct events per second
        """
        B, C, T = x.shape

        # Time-domain filter: 2-8 Hz
        vsw_filtered = self._bandpass_filter_time_domain(x, 2.0, 8.0, num_taps=101)

        # Detect sharp negative transients using second derivative
        # VSW are typically negative at Cz (but can vary by montage)
        # We look for sharp transients of either polarity for robustness
        dx = torch.diff(vsw_filtered, dim=-1)
        d2x = torch.diff(dx, dim=-1)  # Second derivative

        # Sharpness = magnitude of 2nd derivative
        sharpness = d2x.abs()
        sharpness_clean = torch.nan_to_num(sharpness, nan=0.0, posinf=0.0, neginf=0.0)

        # Per-sample threshold (differentiable)
        mean_sharp = sharpness_clean.mean(dim=-1, keepdim=True)
        vertex_mult = cast(torch.Tensor, self.vertex_amplitude_thresh)
        vertex_thresh = vertex_mult * mean_sharp
        above_thresh = self._soft_threshold(
            sharpness_clean, vertex_thresh, reference=mean_sharp
        )

        # Feature 1: vertex_density - fraction above threshold
        vertex_density = above_thresh.mean(dim=-1)

        # Feature 2: peak_ratio - max relative to mean
        max_sharp = sharpness_clean.max(dim=-1)[0]
        peak_ratio = max_sharp / mean_sharp.squeeze(-1).clamp(min=1e-3)

        # Feature 3: event_count - distinct events per second (soft rising edges)
        padded = F.pad(above_thresh, (1, 0), value=0.0)
        transitions = padded[:, :, 1:] - padded[:, :, :-1]
        event_count = torch.relu(transitions).sum(dim=-1)
        event_count = event_count / (T / self.fs)  # Normalize per second

        return torch.stack([vertex_density, peak_ratio, event_count], dim=-1)

    def compute_n3_criterion_features(self, x: torch.Tensor) -> torch.Tensor:
        """Compute multi-threshold epoch-level N3 occupancy features.

        .. deprecated::
            No longer called from ``forward()``. These features caused N3
            over-recall by injecting heavily N3-biased signals. Retained for
            checkpoint compatibility and potential offline analysis.

        This approximates the AASM >=20% slow-wave occupancy rule with a
        differentiable thresholding path over smoothed 0.5-2 Hz envelopes.

        Args:
            x: [B, C, T] normalized EEG waveforms

        Returns:
            [B, C, 2 * len(self._n3_criterion_thresholds)] with interleaved
            occupancy and soft indicator features.
        """
        B, C, T = x.shape
        n3_thresholds = cast(list[float], self._n3_criterion_thresholds)
        n3_steepness = cast(float, self._n3_criterion_steepness)
        n3_occupancy_threshold = cast(float, self._n3_occupancy_threshold)
        n_thresholds = len(n3_thresholds)
        result = torch.zeros(B, C, 2 * n_thresholds, device=x.device, dtype=x.dtype)

        if C == 0 or T == 0:
            return result

        # Slow-wave band envelope (matches slow-wave activity smoothing scale)
        slow_filtered = self._bandpass_filter_time_domain(x, 0.5, 2.0, num_taps=101)

        window_size = max(1, int(1.0 * self.fs))
        kernel = (
            torch.ones(1, 1, window_size, device=x.device, dtype=x.dtype) / window_size
        )
        padding = window_size // 2
        envelope = F.conv1d(
            slow_filtered.abs().view(B * C, 1, T),
            kernel,
            padding=padding,
        )
        envelope = envelope[:, :, :T].view(B, C, T)
        envelope = torch.nan_to_num(envelope, nan=0.0, posinf=0.0, neginf=0.0)

        mean_envelope = envelope.mean(dim=-1, keepdim=True).clamp(min=1e-6)

        for idx, multiplier in enumerate(n3_thresholds):
            threshold = mean_envelope * float(multiplier)
            occupancy = self._soft_threshold(
                envelope, threshold, reference=mean_envelope
            ).mean(dim=-1)
            n3_soft = torch.sigmoid(n3_steepness * (occupancy - n3_occupancy_threshold))

            out_idx = 2 * idx
            result[:, :, out_idx] = occupancy
            result[:, :, out_idx + 1] = n3_soft

        result = torch.nan_to_num(result, nan=0.0, posinf=10.0, neginf=0.0)
        return result

    @torch_compile_disable
    def detect_slow_eye_movements(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Detect slow eye movements (SEM) in EOG channels.

        Returns:
            Tuple of (sem_score [B, C], sem_filtered_signal [B, C, T])
        """
        B, C, T = x.shape
        sem_indices = getattr(self, "_indices_sem", None)
        if sem_indices is None or len(sem_indices) == 0:
            return torch.zeros(B, C, device=x.device, dtype=x.dtype), torch.zeros_like(
                x
            )

        sem_filtered = self._bandpass_filter_stft(x, sem_indices)

        # Envelope (2 s smoothing)
        kernel_env = max(1, min(self._sem_window_samples, T))
        env_kernel = (
            torch.ones(1, 1, kernel_env, device=x.device, dtype=x.dtype) / kernel_env
        )
        envelope = F.conv1d(
            sem_filtered.abs().view(B * C, 1, T),
            env_kernel,
            padding=kernel_env // 2,
        )
        envelope = envelope[:, :, :T].view(B, C, T)

        # FIX: Sanitize envelope before std() to prevent NaN propagation
        envelope_clean = torch.nan_to_num(envelope, nan=0.0, posinf=0.0, neginf=0.0)
        envelope_mean = envelope_clean.mean(dim=-1)
        # CRITICAL FIX: Use min=1e-3 to prevent division issues downstream
        envelope_std = torch.clamp(envelope_clean.std(dim=-1), min=1e-3)

        # Local variance over 4 s window
        kernel_var = max(1, min(self._sem_peak_window_samples, T))
        var_kernel = (
            torch.ones(1, 1, kernel_var, device=x.device, dtype=x.dtype) / kernel_var
        )
        centered = envelope_clean - envelope_mean.unsqueeze(-1)
        variance = F.conv1d(
            (centered**2).view(B * C, 1, T),
            var_kernel,
            padding=kernel_var // 2,
        )
        variance = variance[:, :, :T].view(B, C, T)
        variance_score = variance.mean(dim=-1)

        # Return raw statistics
        # 1. Variance score (local variance)
        # 2. Envelope mean
        # 3. Envelope std

        stats = torch.stack(
            [variance_score, envelope_mean, envelope_std], dim=-1
        )  # [B, C, 3]

        return stats, sem_filtered

    @torch_compile_disable
    def extract_n1_features(
        self,
        x: torch.Tensor,
        channel_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Extract 5 N1-specific features for FocusedN1Attention.

        Features designed to capture N1-specific patterns while avoiding
        redundancy and stage confusion:

        1. Theta/alpha ratio - N1 sweet spot is 1.0-2.0 (rising theta, falling alpha)
        2. Alpha CV (coefficient of variation) - fragmentation marker
        3. Theta emergence - theta power relative to total power
        4. Vertex wave score - brief high-amplitude transients
        5. Slow eye movement (SEM) score - N1 specific, distinguishes from REM

        Args:
            x: [B, C, T] waveform (expects EEG in channel 0, EOG in channel 1+)
            channel_mask: [B, C] optional binary mask

        Returns:
            [B, 5] N1 feature vector

        Note: Decorated with @torch_compile_disable because it calls methods that use
        STFT operations which have complex number operations.
        """
        B, C, T = x.shape
        device = x.device
        dtype = x.dtype

        # Apply channel mask if provided
        if channel_mask is not None:
            if channel_mask.dim() == 1:
                channel_mask = channel_mask.unsqueeze(0).expand(B, -1)
            x = x * channel_mask.unsqueeze(-1).to(dtype)

        # Split channels: EEG (channel 0), EOG (channel 1 if available)
        x_eeg = x[:, :1, :]  # [B, 1, T]
        has_eog = C >= 2
        x_eog = (
            x[:, 1:2, :]
            if has_eog
            else torch.zeros(B, 1, T, device=device, dtype=dtype)
        )

        # Get band indices
        theta_indices = self._indices_theta
        alpha_indices = self._indices_alpha
        delta_indices = self._indices_delta
        sigma_indices = self._indices_spindle  # sigma band

        # 1. Theta/alpha ratio (N1 sweet spot: 1.0-2.0)
        theta_power = self.compute_band_power(
            x_eeg, band_indices=theta_indices
        ).squeeze(
            1
        )  # [B]
        alpha_power = self.compute_band_power(
            x_eeg, band_indices=alpha_indices
        ).squeeze(
            1
        )  # [B]
        theta_alpha_ratio = theta_power / (alpha_power + 1e-6)  # [B]

        # 2. Alpha CV (coefficient of variation) - temporal fragmentation marker
        # Compute alpha envelope and its temporal variability
        alpha_filtered = self._bandpass_filter_stft(x_eeg, alpha_indices)  # [B, 1, T]
        # Compute envelope via abs
        alpha_envelope = alpha_filtered.abs().squeeze(1)  # [B, T]
        # CRITICAL FIX: Use min=1e-3 to prevent division issues with CV
        alpha_mean = alpha_envelope.mean(dim=-1, keepdim=True).clamp(min=1e-3)
        alpha_std = alpha_envelope.std(dim=-1, keepdim=True).clamp(min=1e-3)
        alpha_cv = (alpha_std / alpha_mean).squeeze(-1)  # [B]

        # 3. Theta emergence (relative to total power)
        delta_power = self.compute_band_power(
            x_eeg, band_indices=delta_indices
        ).squeeze(1)
        sigma_power = self.compute_band_power(
            x_eeg, band_indices=sigma_indices
        ).squeeze(1)
        # Total power = sum of main bands (avoiding beta which is noisy)
        total_power = delta_power + theta_power + alpha_power + sigma_power + 1e-6
        theta_emergence = theta_power / total_power  # [B]

        # 4. Vertex wave score (brief high-amplitude transients in theta range)
        vertex_stats = self.detect_vertex_sharp_waves(x_eeg)  # [B, 1, 3]
        # Use max sharpness as the primary vertex indicator
        vertex_score = vertex_stats[:, 0, 2]  # [B] - max sharpness from first channel

        # 5. Slow eye movement (SEM) score - N1 specific
        if has_eog:
            sem_stats, _ = self.detect_slow_eye_movements(x_eog)  # [B, 1, 3], _
            # Use variance score as primary SEM indicator
            sem_score = sem_stats[:, 0, 0]  # [B] - variance score
        else:
            sem_score = torch.zeros(B, device=device, dtype=dtype)

        # Stack all 5 features
        n1_features = torch.stack(
            [
                theta_alpha_ratio,
                alpha_cv,
                theta_emergence,
                vertex_score,
                sem_score,
            ],
            dim=-1,
        )  # [B, 5]

        # Sanitize
        n1_features = torch.nan_to_num(n1_features, nan=0.0, posinf=0.0, neginf=0.0)

        # Apply N1-specific normalization
        n1_features = self._normalize_n1_features(n1_features)

        return n1_features

    @torch_compile_disable
    def _compute_band_powers_stft(self, x: torch.Tensor) -> torch.Tensor:
        """Compute band powers via STFT (fallback when precomputed features unavailable).

        This method is decorated with @torch_compile_disable because STFT and complex
        spectral operations cause CUDA graph issues. When precomputed spectral features
        are available, this method is not called and the forward pass remains compile-safe.

        Args:
            x: [B, C, T] input signals

        Returns:
            [B, num_eeg, 6] band powers for EEG channels
        """
        eeg_indices_tensor = cast(torch.Tensor, self.eeg_indices_tensor)
        x_eeg = x[:, eeg_indices_tensor, :]  # [B, num_eeg, T]

        # [B, num_eeg, num_bands] where num_bands = 6 (slow, fdelta, theta, alpha, sigma, beta)
        band_powers = self.compute_all_band_powers(x_eeg).clone(
            memory_format=torch.contiguous_format
        )  # [B, num_eeg, 6]

        # Apply robust scaling (median/IQR normalization) with clipping to ±20
        # This matches the normalization used in batch_edf_to_zarr_fp32.py for precomputed features
        # Reshape to [B, num_eeg * 6] for per-feature normalization across batch
        B_local, num_eeg, num_bands = band_powers.shape
        band_powers_flat = band_powers.reshape(B_local, -1)  # [B, num_eeg * 6]
        band_powers_flat = self._robust_scale_and_clip(band_powers_flat, dim=0)
        band_powers = band_powers_flat.reshape(
            B_local, num_eeg, num_bands
        )  # [B, num_eeg, 6]

        return band_powers

    def _compute_effective_band_weights(
        self,
        band_powers_eeg: torch.Tensor,
    ) -> torch.Tensor:
        """Compute global+dynamic band weights for the current batch.

        Args:
            band_powers_eeg: [B, num_eeg, num_bands] EEG band powers

        Returns:
            [B, num_bands] effective per-sample band weights
        """
        num_bands = len(self.bands)
        eps = 1e-6

        # Global prior from checkpoint-compatible parameter name `band_weights`.
        global_prior = F.softplus(self.band_weights.float())
        global_prior = (global_prior / global_prior.sum().clamp(min=eps)) * num_bands
        global_prior = torch.clamp(global_prior, min=eps)

        # Per-sample context from average EEG band power.
        band_context = torch.nan_to_num(
            band_powers_eeg.mean(dim=1),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        dynamic_logits = self.band_dynamic_mlp(band_context.float())

        combined_logits = torch.log(global_prior).unsqueeze(0) + dynamic_logits
        effective_weights = torch.softmax(combined_logits, dim=-1) * num_bands
        effective_weights = effective_weights.to(dtype=band_powers_eeg.dtype)
        self._last_band_weights = effective_weights.detach()
        return effective_weights

    @torch_compile_disable
    def forward(
        self,
        x: torch.Tensor,
        channel_mask: torch.Tensor | None = None,
        metadata: dict | None = None,
        precomputed_features: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Extract all features on-the-fly from normalized waveforms.

        Note: Decorated with @torch_compile_disable because this method uses STFT
        operations that cannot be compiled. Running in eager mode prevents CUDA graph
        issues and NaN propagation that occurs with torch.compile + reduce-overhead mode.

        All features are computed on-the-fly from the input waveforms.
        The precomputed_features parameter is kept for backward compatibility
        but is ignored - all features are always computed from x.

        Args:
            x: [B, C, T] normalized waveforms (from signals_stacked)
            channel_mask: [B, C] optional binary mask (1 = present, 0 = absent)
            metadata: Optional dict (unused, kept for API compatibility)
            precomputed_features: DEPRECATED - ignored, all features computed on-the-fly

        Returns:
            [B, F] feature vector where F = self.out_dim
        """
        B, C, T = x.shape
        self._last_band_weights = None

        # Sanitize input to prevent NaN propagation through STFT
        x = torch.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)

        # Apply channel mask if provided
        if channel_mask is not None:
            if channel_mask.dim() == 1:
                channel_mask = channel_mask.unsqueeze(0).expand(B, -1)
            x = x * channel_mask.unsqueeze(-1).to(x.dtype)

        feature_list = []

        # ============================================================
        # BAND POWERS
        # ============================================================
        if self.num_eeg_channels == 0:
            # No EEG channels - create zero features
            band_features = torch.zeros(B, 0, device=x.device, dtype=x.dtype)
            feature_list.append(band_features)
            band_powers_eeg = None
        else:
            band_powers_eeg = self._compute_band_powers_stft(x)  # [B, num_eeg, 6]

            # Apply global+dynamic per-sample band weights.
            effective_band_weights = self._compute_effective_band_weights(
                band_powers_eeg
            )  # [B, 6]
            weighted_powers = band_powers_eeg * effective_band_weights.unsqueeze(1)

            band_features = weighted_powers.reshape(B, -1)
            feature_list.append(band_features)

            # Sanitize band powers
            band_powers_eeg = torch.nan_to_num(
                band_powers_eeg, nan=0.0, posinf=0.0, neginf=0.0
            )

        # ============================================================
        # PRESENCE INDICATORS
        # ============================================================
        if channel_mask is not None:
            presence = (
                channel_mask
                if channel_mask.dim() == 2
                else channel_mask.unsqueeze(0).expand(B, -1)
            )
        else:
            presence = torch.ones(B, self.num_channels, device=x.device, dtype=x.dtype)
        feature_list.append(presence)

        # ============================================================
        # HJORTH PARAMETERS (computed on-the-fly)
        # ============================================================
        if self.include_hjorth and self.num_eeg_channels > 0:
            hjorth = self._compute_hjorth_on_the_fly(x)
            eeg_indices_tensor = cast(torch.Tensor, self.eeg_indices_tensor)
            hjorth_eeg = hjorth[:, eeg_indices_tensor, :]  # [B, num_eeg, 2]
            feature_list.append(hjorth_eeg.reshape(B, -1))

        # ============================================================
        # BAND POWER RATIOS (computed on-the-fly)
        # ============================================================
        if (
            self.include_extra_ratios
            and self.num_eeg_channels > 0
            and band_powers_eeg is not None
        ):
            ratios = self._compute_ratios_on_the_fly(band_powers_eeg)
            feature_list.append(ratios)

        # N3 criterion features removed — caused N3 over-recall.

        # Concatenate all features
        all_features = torch.cat(feature_list, dim=-1)

        # CRITICAL FIX: Clamp features to prevent extreme values from causing
        # downstream overflow in bfloat16 projections. Different features have
        # different scales, so use a generous range that preserves information
        # while preventing numerical issues.
        # - Band powers after log1p: typically 0-15
        # - Hjorth params: typically 0-10
        # - Ratios: typically -10 to +10 after clipping
        # Using ±50 as safe range for bfloat16 (max ~3.4e38, but safe exp range ~±88)
        all_features = torch.clamp(all_features, min=-50.0, max=50.0)

        # Sanitize any remaining NaN/Inf (map to boundary values, not zero)
        all_features = torch.nan_to_num(
            all_features, nan=0.0, posinf=50.0, neginf=-50.0
        )
        all_features = self._normalize_features(all_features)
        all_features = torch.nan_to_num(
            all_features, nan=0.0, posinf=50.0, neginf=-50.0
        )

        return all_features

    def _extract_yasa_features(
        self,
        x: torch.Tensor,
        channel_mask: torch.Tensor | None,
        precomputed: dict[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        """
        Extract YASA-inspired features from precomputed or compute real-time.

        Args:
            x: [B, C, T] input signals (used for real-time fallback)
            channel_mask: [B, C] optional presence mask
            precomputed: Dict with precomputed features:
                - 'hjorth': [B, C, 2] - mobility, complexity
                - 'spectral': [B, C, 6] - slow, fdelta, theta, alpha, sigma, beta
                - 'ratios': [B, 4] - d/t, d/s, d/b, t/a
                Note: 'statistical' is deprecated and ignored

        Returns:
            [B, yasa_feature_dim] feature tensor
        """
        B = x.size(0)
        device = x.device
        dtype = x.dtype

        # If no YASA features are enabled, return empty tensor
        if self.yasa_feature_dim == 0:
            return torch.zeros(B, 0, device=device, dtype=dtype)

        # Try to use precomputed features
        if precomputed is not None and self.use_precomputed:
            features = []

            # Get EEG channel indices tensor
            eeg_idx = cast(torch.Tensor, self.eeg_indices_tensor)

            if self.include_statistical and "statistical" in precomputed:
                stat = precomputed["statistical"]  # [B, C, 5]
                if stat.device != device:
                    stat = stat.to(device=device, dtype=dtype)
                # Select only EEG channels
                stat_eeg = stat[:, eeg_idx, :]  # [B, num_eeg, 5]
                features.append(stat_eeg.reshape(B, -1))

            if self.include_hjorth and "hjorth" in precomputed:
                hjorth = precomputed["hjorth"]  # [B, C, 2]
                if hjorth.device != device:
                    hjorth = hjorth.to(device=device, dtype=dtype)
                hjorth_eeg = hjorth[:, eeg_idx, :]  # [B, num_eeg, 2]
                features.append(hjorth_eeg.reshape(B, -1))

            if self.include_slow_band and "spectral" in precomputed:
                spectral = precomputed["spectral"]  # [B, C, 6]
                if spectral.device != device:
                    spectral = spectral.to(device=device, dtype=dtype)
                # Slow band is index 0
                slow = spectral[:, eeg_idx, 0]  # [B, num_eeg]
                features.append(slow)

            if self.include_extra_ratios and "ratios" in precomputed:
                ratios = precomputed["ratios"]  # [B, 4]
                if ratios.device != device:
                    ratios = ratios.to(device=device, dtype=dtype)
                features.append(ratios)

            if features:
                result = torch.cat(features, dim=-1)
                # Sanitize any NaN/Inf values
                result = torch.nan_to_num(result, nan=0.0, posinf=10.0, neginf=-10.0)
                return result

        # Fallback: compute features in real-time using PyTorch implementation
        return self._compute_yasa_realtime(x, channel_mask)

    def _compute_yasa_realtime(
        self,
        x: torch.Tensor,
        channel_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Compute YASA features in real-time using PyTorch.

        This is a fallback when precomputed features are not available.
        Uses lazy-loaded YASAFeatureExtractorTorch.

        Args:
            x: [B, C, T] input signals
            channel_mask: [B, C] optional presence mask

        Returns:
            [B, yasa_feature_dim] feature tensor
        """
        B = x.size(0)
        device = x.device
        dtype = x.dtype

        # Lazy-load the YASA torch extractor
        if self._yasa_torch_extractor is None:
            from spectra.models.yasa_features import YASAFeatureExtractorTorch

            self._yasa_torch_extractor = YASAFeatureExtractorTorch(
                fs=float(self.fs),
                epoch_sec=float(self.epoch_sec),
            ).to(device)

        # Ensure extractor is on correct device
        if (
            next(self._yasa_torch_extractor.parameters(), torch.tensor(0.0)).device
            != device
        ):
            self._yasa_torch_extractor = self._yasa_torch_extractor.to(device)

        # Compute all YASA features
        yasa_dict = self._yasa_torch_extractor(x, channel_mask)

        # Extract the features we need
        features = []
        eeg_idx = cast(torch.Tensor, self.eeg_indices_tensor)

        if self.include_statistical and "statistical" in yasa_dict:
            stat = yasa_dict["statistical"]  # [B, C, 5]
            stat_eeg = stat[:, eeg_idx, :]  # [B, num_eeg, 5]
            features.append(stat_eeg.reshape(B, -1))

        if self.include_hjorth and "hjorth" in yasa_dict:
            hjorth = yasa_dict["hjorth"]  # [B, C, 2]
            hjorth_eeg = hjorth[:, eeg_idx, :]  # [B, num_eeg, 2]
            features.append(hjorth_eeg.reshape(B, -1))

        if self.include_slow_band and "spectral" in yasa_dict:
            spectral = yasa_dict["spectral"]  # [B, C, 6]
            slow = spectral[:, eeg_idx, 0]  # [B, num_eeg]
            features.append(slow)

        if self.include_extra_ratios and "ratios" in yasa_dict:
            features.append(yasa_dict["ratios"])  # [B, 4]

        if features:
            result = torch.cat(features, dim=-1)
            result = torch.nan_to_num(result, nan=0.0, posinf=10.0, neginf=-10.0)
            return result

        return torch.zeros(B, 0, device=device, dtype=dtype)


class N1FocusedAttention(nn.Module):
    """
    Focused N1 attention with additive modulation.

    Uses 6 core N1-specific features to modulate the main feature representation
    through additive attention (more stable than multiplicative).

    Features (from extract_n1_features):
        1. Theta/alpha ratio - N1 sweet spot is 1.0-2.0
        2. Alpha CV - fragmentation marker (high CV = transitioning)
        3. Theta emergence - theta relative to total power
        4. Vertex wave score - brief high-amplitude transients
        5. SEM score - slow eye movements (N1 specific)

    Key design decisions:
        - Additive attention instead of multiplicative (more stable gradients)
        - Learnable scale parameter controls attention strength
        - Unified normalization with main features (no scale mismatch)

    Args:
        d_model: Dimension of the main feature representation
        n1_feature_dim: Dimension of N1-specific features (default: 5)

    Input:
        x: [B, L, d_model] or [B, d_model] main features
        n1_features: [B, n1_feature_dim] N1-specific features

    Output:
        [B, L, d_model] or [B, d_model] modulated features (same shape as input)
    """

    def __init__(self, d_model: int, n1_feature_dim: int = 5):
        super().__init__()

        # Project N1 features to model dimension
        self.n1_query = nn.Linear(n1_feature_dim, d_model)

        # Gate mechanism for additive attention
        # Input: concatenated [main_features, n1_context] → output: attention weights
        self.gate = nn.Linear(d_model * 2, d_model)

        # Learnable scale for additive modulation
        # Initialized small (0.1) to start with minimal N1 influence
        self.scale = nn.Parameter(torch.tensor(0.1))

        # Layer normalization for stability
        self.norm = nn.LayerNorm(d_model)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """
        Handle backward compatibility for checkpoints with confidence_net.

        Newer checkpoints may have confidence_net weights that don't exist in this
        simpler implementation. We silently ignore them to allow loading.
        """
        # Remove confidence_net weights from state_dict if present (they don't exist here)
        keys_to_remove = []
        for key in list(state_dict.keys()):
            if key.startswith(prefix + "confidence_net"):
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del state_dict[key]
            if not torch.compiler.is_compiling():
                import logging

                logger = logging.getLogger(__name__)
                logger.info(
                    f"[BACKWARD COMPAT] Removed '{key}' from checkpoint "
                    f"(confidence_net not present in this simpler N1FocusedAttention)"
                )

        # Handle scale parameter value differences
        scale_key = prefix + "scale"
        if scale_key in state_dict:
            old_scale = state_dict[scale_key]
            # Newer checkpoints may have scale initialized to 0.05, reset to 0.1 for this version
            if old_scale.item() < 0.08:  # Likely from newer version with 0.05 default
                state_dict[scale_key] = torch.tensor(0.1)
                if not torch.compiler.is_compiling():
                    import logging

                    logger = logging.getLogger(__name__)
                    logger.info(
                        f"[BACKWARD COMPAT] Reset N1 attention scale from {old_scale.item():.4f} to 0.1 "
                        f"(newer checkpoint had smaller default)"
                    )

        # Handle N1 feature-dimension changes (legacy checkpoints: 6 -> current: 5).
        n1_query_key = prefix + "n1_query.weight"
        if n1_query_key in state_dict:
            ckpt_weight = state_dict[n1_query_key]
            current_weight = self.n1_query.weight
            if (
                isinstance(ckpt_weight, torch.Tensor)
                and isinstance(current_weight, torch.Tensor)
                and ckpt_weight.ndim == 2
                and current_weight.ndim == 2
                and ckpt_weight.shape != current_weight.shape
            ):
                ckpt_out, ckpt_in = ckpt_weight.shape
                cur_out, cur_in = current_weight.shape
                if ckpt_out == cur_out:
                    if ckpt_in > cur_in:
                        # Legacy 6-dim N1 features: drop removed trailing feature column.
                        state_dict[n1_query_key] = ckpt_weight[:, :cur_in]
                    elif ckpt_in < cur_in:
                        std = (2.0 / (cur_in + cur_out)) ** 0.5
                        pad = (
                            torch.randn(
                                cur_out,
                                cur_in - ckpt_in,
                                device=ckpt_weight.device,
                                dtype=ckpt_weight.dtype,
                            )
                            * std
                        )
                        state_dict[n1_query_key] = torch.cat([ckpt_weight, pad], dim=1)
                    if not torch.compiler.is_compiling():
                        import logging

                        logger = logging.getLogger(__name__)
                        logger.info(
                            f"[BACKWARD COMPAT] Resized '{n1_query_key}' from "
                            f"{tuple(ckpt_weight.shape)} to "
                            f"{tuple(state_dict[n1_query_key].shape)}"
                        )

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x: torch.Tensor, n1_features: torch.Tensor) -> torch.Tensor:
        """
        Apply additive N1-focused attention to main features.

        Args:
            x: [B, L, d_model] or [B, d_model] main features
            n1_features: [B, n1_feature_dim] N1-specific features

        Returns:
            [B, L, d_model] or [B, d_model] modulated features
        """
        # Project N1 features to model dimension
        n1_context = self.n1_query(n1_features)  # [B, d_model]

        # Handle both [B, d_model] and [B, L, d_model] inputs
        if x.dim() == 3:
            # [B, L, d_model] - expand n1_context to match
            n1_context_expanded = n1_context.unsqueeze(1).expand_as(
                x
            )  # [B, L, d_model]
        else:
            # [B, d_model] - no expansion needed
            n1_context_expanded = n1_context

        # Concatenate main features with N1 context
        combined = torch.cat(
            [x, n1_context_expanded], dim=-1
        )  # [B, L, 2*d_model] or [B, 2*d_model]

        # Compute attention weights (tanh gives [-1, 1] range for additive modulation)
        attention_weight = torch.tanh(
            self.gate(combined)
        )  # [B, L, d_model] or [B, d_model]

        # Additive modulation with learned scale
        # x + scale * attention allows both boosting and suppression
        # Scale controls how much N1 features influence the output
        modulated = x + self.scale * attention_weight

        # Apply layer normalization for stability
        return self.norm(modulated)


class BottleneckProjection(nn.Module):
    """
    Bottleneck projection with residual for feature dimension transformation.

    Designed for projecting low-dimensional engineered features (e.g., 16-dim)
    to high-dimensional CNN feature space (e.g., 256-dim) with:
    1. Expansion to learn feature interactions
    2. Compression to target dimension
    3. Residual connection for gradient flow

    Args:
        in_dim: Input feature dimension
        out_dim: Output feature dimension
        expansion: Expansion factor for hidden dimension (default: 2)
        dropout: Dropout probability (default: 0.1)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        expansion: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        hidden_dim = out_dim * expansion

        # Main pathway: expand → nonlinearity → compress
        self.bottleneck = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

        # Residual pathway
        self.residual_proj = (
            nn.Linear(in_dim, out_dim, bias=False)
            if in_dim != out_dim
            else nn.Identity()
        )

        # Post-addition normalization
        self.norm = nn.LayerNorm(out_dim)

        # Stable initialization
        self._init_weights()

    def _init_weights(self):
        # Standard initialization for gradient flow
        # Previous gain=0.1 was too small, causing weak gradients through bottleneck
        # Using gain=1.0 (standard) with He initialization for GELU activations
        for layer in self.bottleneck:
            if isinstance(layer, nn.Linear):
                # He initialization (fan_out) for GELU activation
                nn.init.kaiming_normal_(
                    layer.weight, mode="fan_out", nonlinearity="relu"
                )
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        if isinstance(self.residual_proj, nn.Linear):
            nn.init.xavier_uniform_(self.residual_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., in_dim] input features (any leading dimensions)

        Returns:
            [..., out_dim] projected features
        """
        return self.norm(self.bottleneck(x) + self.residual_proj(x))


class SequentialFeatureFusion(nn.Module):
    """
    Streamlined feature fusion with single cross-attention and gated residual.

    Simplified from the original 3-stage design to reduce over-clamping and
    improve signal preservation. Key changes:
    - Single cross-attention stage (CNN queries engineered features)
    - Cleaner residual path: CNN features preserved, gate modulates attention only
    - Trust LayerNorm for normalization instead of hard clamping
    - Uses nn.MultiheadAttention for efficiency and compatibility

    Architecture:
        1. Project engineered features to CNN dimension
        2. Cross-attention: CNN attends to projected engineered features
        3. Gated residual: fused = cnn + gate * attention_output
        4. LayerNorm for stability

    Args:
        cnn_dim: Dimension of CNN features
        feat_dim: Dimension of engineered features (before projection)
        num_heads: Number of attention heads (default: 4)
        dropout: Dropout rate (default: 0.1)
    """

    def __init__(
        self, cnn_dim: int, feat_dim: int, num_heads: int = 4, dropout: float = 0.1
    ):
        super().__init__()

        # Project engineered features to CNN dimension
        self.feat_proj = nn.Linear(feat_dim, cnn_dim)

        # Single cross-attention layer (CNN queries engineered features)
        # Using nn.MultiheadAttention for efficiency and Flash Attention support
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=cnn_dim,
            num_heads=num_heads,  # Match or exceed transformer heads
            kdim=cnn_dim,
            vdim=cnn_dim,
            dropout=dropout,
            batch_first=True,
        )

        # Gated residual: gate modulates attention contribution only
        # Bottleneck design to prevent parameter explosion while allowing
        # engineered features to directly influence gating decision
        bottleneck = cnn_dim // 2
        self.gate = nn.Sequential(
            nn.Linear(cnn_dim * 3, bottleneck),  # cnn + eng_proj + attn_out
            nn.GELU(),
            nn.Linear(bottleneck, cnn_dim),
            nn.Sigmoid(),
        )

        # Final normalization for stability
        self.norm = nn.LayerNorm(cnn_dim)

    def forward(self, cnn_feats: torch.Tensor, eng_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cnn_feats: [B, L, cnn_dim] CNN features
            eng_feats: [B, L, feat_dim] Engineered features

        Returns:
            fused_feats: [B, L, cnn_dim] Fused features
        """
        # Project engineered features to match CNN dimension
        eng_proj = self.feat_proj(eng_feats)  # [B, L, cnn_dim]

        # Cross-attention: CNN attends to engineered features
        # query=CNN features, key/value=projected engineered features
        attn_out, _ = self.cross_attn(
            query=cnn_feats, key=eng_proj, value=eng_proj
        )  # [B, L, cnn_dim]

        # Gated fusion: gate controls how much attention contributes
        # Include eng_proj directly so engineered features have independent influence
        gate_input = torch.cat(
            [cnn_feats, eng_proj, attn_out], dim=-1
        )  # [B, L, 3*cnn_dim]
        gate = self.gate(gate_input)  # [B, L, cnn_dim]

        # Residual connection: CNN features preserved, gate modulates attention only
        fused = cnn_feats + gate * attn_out

        # Normalize for stability (trust LayerNorm, no hard clamping)
        return self.norm(fused)


class VariancePreservingRMSNorm(nn.Module):
    """RMSNorm without centering + explicit variance re-injection.

    Unlike standard LayerNorm which removes variance information during
    normalization, this module preserves variance as an explicit feature.
    This is critical for N2/N3 discrimination where delta power variance
    is a key distinguishing characteristic.

    Why this combination:
    1. No centering -> preserves relative feature magnitudes (N2 transients vs N3 sustained)
    2. RMS scaling -> numerical stability without destroying variance ratio
    3. Variance side channel -> explicit signal the model can learn to use

    Args:
        dim: Feature dimension to normalize
        eps: Small constant for numerical stability
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

        # Variance pathway - projects scalar variance back to feature space
        self.var_proj = nn.Sequential(
            nn.Linear(1, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, dim),
        )
        # Initialize gate small for conservative initial contribution
        self.var_gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute variance BEFORE normalization (this is what we want to preserve)
        var = x.var(dim=-1, keepdim=True)  # [B, ..., 1]

        # RMSNorm (no mean centering)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        normalized = (x / rms) * self.weight

        # Re-inject variance as learned embedding
        var_embed = self.var_proj(var)  # [B, ..., dim]

        return normalized + self.var_gate * var_embed


class TemporalEngineeredFeatures(nn.Module):
    """Compute comprehensive engineered sleep features at each temporal position.

    Instead of computing a single feature vector per epoch, this module
    uses sliding window STFT to compute spectral features at each temporal
    position. This enables position-specific K/V in cross-attention,
    allowing the model to attend to specific transient events (K-complexes,
    spindles) at their actual temporal locations.

    Features computed at each temporal position:
    - Band powers: slow, fdelta, theta, alpha, sigma, beta (per EEG channel)
    - Band variances: delta_var, sigma_var (for N2/N3 discrimination)
    - Hjorth parameters: mobility, complexity (per EEG channel)
    - Band ratios: delta/theta, delta/sigma, delta/beta, theta/alpha
    - Temporal markers: spindle_density_proxy, muscle_tone_proxy

    Args:
        fs: Sampling frequency (Hz)
        target_temporal_dim: Target output temporal dimension to match CNN
        window_sec: STFT window size in seconds (default: 2.0)
        hop_sec: STFT hop size in seconds (default: 0.5)
        num_eeg_channels: Number of EEG channels to process (default: 2)
        include_hjorth: Include Hjorth parameters (default: True)
        include_ratios: Include band power ratios (default: True)
        include_variances: Include temporal variance features (default: True)
    """

    BANDS = {
        "slow": (0.4, 1.0),
        "fdelta": (1.0, 4.0),
        "theta": (4.0, 7.0),  # Matches YASA
        "alpha": (8.0, 13.0),
        "sigma": (11.0, 16.0),
        "beta": (13.0, 30.0),
    }

    def __init__(
        self,
        fs: float = 128.0,
        target_temporal_dim: int = 47,
        window_sec: float = 2.0,
        hop_sec: float = 0.5,
        num_eeg_channels: int = 2,
        include_hjorth: bool = True,
        include_ratios: bool = True,
        include_variances: bool = True,
    ):
        super().__init__()
        self.fs = fs
        self.target_temporal_dim = target_temporal_dim
        self.num_eeg_channels = num_eeg_channels
        self.include_hjorth = include_hjorth
        self.include_ratios = include_ratios
        self.include_variances = include_variances

        # STFT parameters
        self.n_fft = int(window_sec * fs)
        if self.n_fft % 2 != 0:
            self.n_fft += 1
        self.hop_length = int(hop_sec * fs)

        # Hjorth window (shorter for better temporal resolution)
        self.hjorth_window = int(1.0 * fs)  # 1 second window
        self.hjorth_hop = int(0.25 * fs)  # 0.25 second hop

        # Register Hann window
        self.window: torch.Tensor
        self.register_buffer("window", torch.hann_window(self.n_fft))

        # Pre-compute frequency bins
        self.freqs: torch.Tensor
        freqs = torch.fft.rfftfreq(self.n_fft, d=1.0 / fs)
        self.register_buffer("freqs", freqs)

        # Band masks registered below
        self.mask_slow: torch.Tensor
        self.mask_fdelta: torch.Tensor
        self.mask_theta: torch.Tensor
        self.mask_alpha: torch.Tensor
        self.mask_sigma: torch.Tensor
        self.mask_beta: torch.Tensor

        # Pre-compute band masks
        for band_name, (low, high) in self.BANDS.items():
            mask = (freqs >= low) & (freqs <= high)
            self.register_buffer(f"mask_{band_name}", mask)

        # Calculate output dimension
        # Band powers: 6 bands * num_eeg_channels
        n_band_features = len(self.BANDS) * num_eeg_channels
        # Variances: 2 (delta_var, sigma_var) * num_eeg_channels
        n_variance_features = 2 * num_eeg_channels if include_variances else 0
        # Hjorth: 2 (mobility, complexity) * num_eeg_channels
        n_hjorth_features = 2 * num_eeg_channels if include_hjorth else 0
        # Ratios: 4 (delta/theta, delta/sigma, delta/beta, theta/alpha)
        n_ratio_features = 4 if include_ratios else 0
        # Temporal markers: 2 (spindle_proxy, muscle_proxy)
        n_temporal_markers = 2

        self.out_features = (
            n_band_features
            + n_variance_features
            + n_hjorth_features
            + n_ratio_features
            + n_temporal_markers
        )

        # Learnable projection to refine and compress features
        self.projection = nn.Sequential(
            nn.Linear(self.out_features, self.out_features),
            nn.LayerNorm(self.out_features),
            nn.GELU(),
            nn.Linear(self.out_features, self.out_features),
        )

    @torch_compile_disable
    def forward(
        self,
        x: torch.Tensor,
        channel_mask: torch.Tensor | None = None,
        eeg_indices: list[int] | None = None,
    ) -> torch.Tensor:
        """Compute temporal engineered features.

        Args:
            x: [B*L, C, T] raw waveform data
            channel_mask: Optional channel presence mask
            eeg_indices: Indices of EEG channels (default: first num_eeg_channels)

        Returns:
            features: [B*L, T', F] features at each temporal position
                where T' = target_temporal_dim, F = out_features
        """
        B, C, T = x.shape
        device = x.device
        dtype = x.dtype
        eps = 1e-10

        # Default to first num_eeg_channels if not specified
        if eeg_indices is None:
            eeg_indices = list(range(min(self.num_eeg_channels, C)))

        feature_list = []

        # Compute expected n_frames from STFT parameters (before any loops)
        # STFT with center=True pads by n_fft//2 on each side
        n_frames = (T + self.n_fft) // self.hop_length + 1

        # === BAND POWERS (per EEG channel) ===
        band_powers = {}  # Store for ratio computation
        for ch_idx in eeg_indices[: self.num_eeg_channels]:
            if ch_idx >= C:
                # Pad with zeros if channel doesn't exist
                ch_signal = torch.zeros(B, T, device=device, dtype=dtype)
            else:
                ch_signal = x[:, ch_idx, :]  # [B, T]

            # Compute STFT
            stft_out = _compile_safe_stft(
                ch_signal,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=self.window,
                return_complex=True,
                center=True,
                pad_mode="reflect",
            )  # [B, n_freqs, n_frames]

            power = stft_out.abs().pow(2)  # [B, n_freqs, n_frames]
            # Update n_frames from actual STFT output (more accurate than formula)
            if ch_idx == eeg_indices[0]:
                n_frames = power.shape[-1]

            # Extract each band power
            for band_name in self.BANDS.keys():
                mask = getattr(self, f"mask_{band_name}")
                band_power = power[:, mask, :].mean(dim=1)  # [B, n_frames]
                log_power = torch.log10(band_power + eps)
                feature_list.append(log_power)

                # Store for ratios (use first EEG channel only)
                if ch_idx == eeg_indices[0]:
                    band_powers[band_name] = band_power

        # === BAND VARIANCES (for N2/N3 discrimination) ===
        if self.include_variances:
            for ch_idx in eeg_indices[: self.num_eeg_channels]:
                if ch_idx >= C:
                    ch_signal = torch.zeros(B, T, device=device, dtype=dtype)
                else:
                    ch_signal = x[:, ch_idx, :]

                stft_out = _compile_safe_stft(
                    ch_signal,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    window=self.window,
                    return_complex=True,
                    center=True,
                    pad_mode="reflect",
                )
                power = stft_out.abs().pow(2)

                # Delta variance (fdelta band)
                fdelta_power = power[:, self.mask_fdelta, :].mean(dim=1)
                delta_var = self._sliding_variance(fdelta_power, window_size=3)
                feature_list.append(torch.log10(delta_var + eps))

                # Sigma variance (spindle variability)
                sigma_power = power[:, self.mask_sigma, :].mean(dim=1)
                sigma_var = self._sliding_variance(sigma_power, window_size=3)
                feature_list.append(torch.log10(sigma_var + eps))

        # === HJORTH PARAMETERS (per EEG channel) ===
        if self.include_hjorth:
            for ch_idx in eeg_indices[: self.num_eeg_channels]:
                if ch_idx >= C:
                    ch_signal = torch.zeros(B, T, device=device, dtype=dtype)
                else:
                    ch_signal = x[:, ch_idx, :]

                mobility, complexity = self._compute_hjorth_temporal(ch_signal)

                # Interpolate Hjorth features to match STFT n_frames
                # Hjorth uses different windowing, so dimensions may differ
                if mobility.shape[1] != n_frames:
                    mobility = F.interpolate(
                        mobility.unsqueeze(1),  # [B, 1, hjorth_frames]
                        size=n_frames,
                        mode="linear",
                        align_corners=False,
                    ).squeeze(
                        1
                    )  # [B, n_frames]
                    complexity = F.interpolate(
                        complexity.unsqueeze(1),
                        size=n_frames,
                        mode="linear",
                        align_corners=False,
                    ).squeeze(1)

                feature_list.append(mobility)
                feature_list.append(complexity)

        # === BAND RATIOS ===
        if self.include_ratios and band_powers:
            # delta/theta ratio
            delta_theta = torch.log10(
                (band_powers["fdelta"] + eps) / (band_powers["theta"] + eps)
            )
            feature_list.append(delta_theta)

            # delta/sigma ratio (N2/N3 discrimination)
            delta_sigma = torch.log10(
                (band_powers["fdelta"] + eps) / (band_powers["sigma"] + eps)
            )
            feature_list.append(delta_sigma)

            # delta/beta ratio
            delta_beta = torch.log10(
                (band_powers["fdelta"] + eps) / (band_powers["beta"] + eps)
            )
            feature_list.append(delta_beta)

            # theta/alpha ratio (N1 detection)
            theta_alpha = torch.log10(
                (band_powers["theta"] + eps) / (band_powers["alpha"] + eps)
            )
            feature_list.append(theta_alpha)

        # === TEMPORAL MARKERS ===
        # Spindle density proxy (sigma power relative to neighbors)
        if "sigma" in band_powers:
            sigma_zscore = self._local_zscore(band_powers["sigma"])
            feature_list.append(sigma_zscore)
        else:
            feature_list.append(torch.zeros(B, n_frames, device=device, dtype=dtype))

        # Muscle tone proxy (high frequency power from last channel, assumed EMG)
        if C > 2:  # Has EMG channel
            emg_signal = x[:, -1, :]  # Last channel assumed EMG
            stft_emg = _compile_safe_stft(
                emg_signal,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=self.window,
                return_complex=True,
                center=True,
                pad_mode="reflect",
            )
            emg_power = stft_emg.abs().pow(2)
            # High-frequency EMG (use beta band as proxy)
            muscle_tone = emg_power[:, self.mask_beta, :].mean(dim=1)
            feature_list.append(torch.log10(muscle_tone + eps))
        else:
            feature_list.append(torch.zeros(B, n_frames, device=device, dtype=dtype))

        # Stack all features [B, n_frames, F]
        features = torch.stack(feature_list, dim=-1)

        # Interpolate to match target temporal dimension
        current_frames = features.shape[1]
        if current_frames != self.target_temporal_dim:
            features = F.interpolate(
                features.permute(0, 2, 1),  # [B, F, n_frames]
                size=self.target_temporal_dim,
                mode="linear",
                align_corners=False,
            ).permute(
                0, 2, 1
            )  # [B, T', F]

        # Apply projection and sanitize
        features = self.projection(features)
        features = torch.nan_to_num(features, nan=0.0, posinf=10.0, neginf=-10.0)
        features = torch.clamp(features, min=-20.0, max=20.0)

        return features

    def _sliding_variance(self, x: torch.Tensor, window_size: int = 3) -> torch.Tensor:
        """Compute sliding window variance."""
        pad = window_size // 2
        x_padded = F.pad(x, (pad, pad), mode="reflect")

        x_mean = F.avg_pool1d(
            x_padded.unsqueeze(1), kernel_size=window_size, stride=1
        ).squeeze(1)

        x2_mean = F.avg_pool1d(
            (x_padded**2).unsqueeze(1), kernel_size=window_size, stride=1
        ).squeeze(1)

        variance = (x2_mean - x_mean**2).clamp(min=0)
        return variance

    def _local_zscore(self, x: torch.Tensor, window_size: int = 5) -> torch.Tensor:
        """Compute local z-score for detecting peaks/transients."""
        pad = window_size // 2
        x_padded = F.pad(x, (pad, pad), mode="reflect")

        x_mean = F.avg_pool1d(
            x_padded.unsqueeze(1), kernel_size=window_size, stride=1
        ).squeeze(1)

        x2_mean = F.avg_pool1d(
            (x_padded**2).unsqueeze(1), kernel_size=window_size, stride=1
        ).squeeze(1)

        x_std = torch.sqrt((x2_mean - x_mean**2).clamp(min=1e-8))
        zscore = (x - x_mean) / x_std
        return zscore.clamp(min=-5.0, max=5.0)

    def _compute_hjorth_temporal(
        self, signal: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Hjorth parameters in sliding windows.

        Args:
            signal: [B, T] time series

        Returns:
            mobility: [B, n_frames] Hjorth mobility at each position
            complexity: [B, n_frames] Hjorth complexity at each position
        """
        B, T = signal.shape
        device = signal.device
        dtype = signal.dtype

        window = self.hjorth_window
        hop = self.hjorth_hop
        n_frames = (T - window) // hop + 1

        if n_frames <= 0:
            # Signal too short, return zeros
            return (
                torch.zeros(B, 1, device=device, dtype=dtype),
                torch.zeros(B, 1, device=device, dtype=dtype),
            )

        # Unfold signal into windows [B, n_frames, window]
        signal_unfolded = signal.unfold(dimension=1, size=window, step=hop)

        # First derivative
        diff1 = torch.diff(signal_unfolded, dim=-1)
        # Second derivative
        diff2 = torch.diff(diff1, dim=-1)

        # Variance of signal, first diff, second diff
        var0 = signal_unfolded.var(dim=-1).clamp(min=1e-10)
        var1 = diff1.var(dim=-1).clamp(min=1e-10)
        var2 = diff2.var(dim=-1).clamp(min=1e-10)

        # Hjorth mobility = sqrt(var(diff1) / var(signal))
        mobility = torch.sqrt(var1 / var0)

        # Hjorth complexity = mobility(diff1) / mobility(signal)
        mobility_diff1 = torch.sqrt(var2 / var1)
        complexity = mobility_diff1 / mobility.clamp(min=1e-10)

        # Clamp to reasonable range
        mobility = mobility.clamp(min=0.0, max=10.0)
        complexity = complexity.clamp(min=0.0, max=10.0)

        return mobility, complexity


class TemporalFeatureFusion(nn.Module):
    """
    Fuse engineered sleep features into CNN temporal representations via cross-attention.

    This module enriches the temporal feature sequence from the CNN encoder with
    domain-specific engineered features (spectral bands, spindle density, etc.).
    This allows downstream attention to leverage physiological priors when
    attending to temporal positions.

    Architecture:
        1. Project engineered features (scalar per epoch) to match CNN dimension
        2. Cross-attention: temporal positions attend to the engineered feature summary
        3. Gated residual: cnn_temporal + gate * cross_attention_output
        4. LayerNorm for stability

    Args:
        cnn_dim: Dimension of CNN temporal features (temporal_dim from encoder)
        eng_dim: Dimension of engineered features (from SleepFeatureExtractor.out_dim)
        num_heads: Number of cross-attention heads (default: 4)
        dropout: Dropout probability (default: 0.1)

    Input/Output:
        cnn_temporal: [B, T', D_cnn] temporal features from CNN encoder
        eng_feats: [B, F_eng] engineered features (one vector per epoch)
        returns: [B, T', D_cnn] enriched temporal features
    """

    def __init__(
        self,
        cnn_dim: int,
        eng_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_variance_preserving_norm: bool = True,
        use_temporal_engineered_features: bool = False,
        temporal_eng_dim: int | None = None,
    ):
        super().__init__()

        self.cnn_dim = cnn_dim
        self.eng_dim = eng_dim
        self.use_variance_preserving_norm = use_variance_preserving_norm
        self.use_temporal_engineered_features = use_temporal_engineered_features

        # FiLM-style modulation: gamma * x + beta
        # This replaces degenerate single-token cross-attention which essentially
        # just broadcasts the engineered features to all positions with uniform weight
        # FiLM is more efficient and equally expressive for this use case

        # Gamma projection: produces multiplicative modulation factors
        self.gamma_proj = nn.Sequential(
            nn.Linear(eng_dim, cnn_dim * 2),
            nn.LayerNorm(cnn_dim * 2),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(cnn_dim * 2, cnn_dim),
        )

        # Beta projection: produces additive modulation factors
        self.beta_proj = nn.Sequential(
            nn.Linear(eng_dim, cnn_dim * 2),
            nn.LayerNorm(cnn_dim * 2),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(cnn_dim * 2, cnn_dim),
        )

        # Gated residual connection
        # Gate takes both original and modulated features to decide fusion weight
        self.gate = nn.Sequential(
            nn.Linear(cnn_dim * 2, cnn_dim),
            nn.Sigmoid(),
        )

        # Cross-attention for temporal engineered features
        # When temporal features are provided at each position, use cross-attention
        # instead of FiLM to allow position-specific fusion
        if use_temporal_engineered_features and temporal_eng_dim is not None:
            self.temporal_cross_attn = nn.MultiheadAttention(
                embed_dim=cnn_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            # Project temporal features to cnn_dim for K/V
            self.temporal_kv_proj = nn.Linear(temporal_eng_dim, cnn_dim)
            # Gate for blending cross-attention output
            self.temporal_gate = nn.Sequential(
                nn.Linear(cnn_dim * 2, cnn_dim),
                nn.Sigmoid(),
            )
            # Initialize temporal gate to be conservative
            temporal_gate_linear = self.temporal_gate[0]
            if (
                isinstance(temporal_gate_linear, nn.Linear)
                and temporal_gate_linear.bias is not None
            ):
                nn.init.constant_(temporal_gate_linear.bias, -2.0)
        else:
            self.temporal_cross_attn = None
            self.temporal_kv_proj = None
            self.temporal_gate = None

        # Layer normalization for stability
        # Use variance-preserving norm to retain N2/N3 discriminative information
        if use_variance_preserving_norm:
            self.norm = VariancePreservingRMSNorm(cnn_dim)
        else:
            self.norm = nn.LayerNorm(cnn_dim)

        # Initialize gate to be conservative (small initial contribution)
        self._init_gate_bias()

        # Initialize gamma to be close to 1 (identity) and beta close to 0
        self._init_film_params()

    def _init_gate_bias(self):
        """Initialize gate to output small values initially."""
        # Last linear in gate should output small values
        # sigmoid(-2) ~ 0.12, so initial contribution is modest
        gate_linear = self.gate[0]
        if isinstance(gate_linear, nn.Linear) and gate_linear.bias is not None:
            nn.init.constant_(gate_linear.bias, -2.0)

    def _init_film_params(self):
        """Initialize FiLM parameters for identity-like initial behavior."""
        # Gamma should output values close to 0 initially (so 1 + gamma ≈ 1)
        # Beta should output values close to 0 initially
        for proj in [self.gamma_proj, self.beta_proj]:
            # Find the last Linear layer
            for layer in reversed(proj):
                if isinstance(layer, nn.Linear):
                    nn.init.zeros_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                    break

    @torch_compile_disable
    def forward(
        self,
        cnn_temporal: torch.Tensor,
        eng_feats: torch.Tensor,
        temporal_eng_feats: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Enrich temporal features with engineered feature information.

        Two modes of operation:
        1. FiLM mode (default): Uses per-epoch features to modulate all positions equally
        2. Temporal mode: Uses position-specific features with cross-attention

        FiLM (Feature-wise Linear Modulation) applies affine transformations
        conditioned on the engineered features: modulated = gamma * x + beta

        Args:
            cnn_temporal: [B, T', D_cnn] temporal features from CNN (before pooling)
            eng_feats: [B, F_eng] engineered features (per-epoch summary)
            temporal_eng_feats: [B, T', F_temp] optional temporal engineered features

        Returns:
            enriched: [B, T', D_cnn] temporal features enriched with domain knowledge
        """
        # CRITICAL: Clamp inputs FIRST to prevent extreme values, then sanitize NaN
        cnn_temporal = torch.clamp(cnn_temporal, min=-100.0, max=100.0)
        cnn_temporal = torch.nan_to_num(
            cnn_temporal, nan=0.0, posinf=100.0, neginf=-100.0
        )

        eng_feats = torch.clamp(eng_feats, min=-50.0, max=50.0)
        eng_feats = torch.nan_to_num(eng_feats, nan=0.0, posinf=50.0, neginf=-50.0)

        # === TEMPORAL CROSS-ATTENTION MODE ===
        # When temporal features are provided, use cross-attention for position-specific fusion
        if (
            temporal_eng_feats is not None
            and self.temporal_cross_attn is not None
            and self.temporal_kv_proj is not None
            and self.temporal_gate is not None
        ):
            temporal_eng_feats = torch.clamp(temporal_eng_feats, min=-50.0, max=50.0)
            temporal_eng_feats = torch.nan_to_num(
                temporal_eng_feats, nan=0.0, posinf=50.0, neginf=-50.0
            )

            # Project temporal features to cnn_dim for K/V
            kv = self.temporal_kv_proj(temporal_eng_feats)  # [B, T', D_cnn]

            # Cross-attention: cnn_temporal queries the temporal engineered features
            # Q = cnn_temporal, K = V = projected temporal features
            attn_out, _ = self.temporal_cross_attn(
                query=cnn_temporal,
                key=kv,
                value=kv,
            )  # [B, T', D_cnn]

            attn_out = torch.clamp(attn_out, min=-100.0, max=100.0)
            attn_out = torch.nan_to_num(attn_out, nan=0.0, posinf=100.0, neginf=-100.0)

            # Gated residual for cross-attention
            gate_input = torch.cat([cnn_temporal, attn_out], dim=-1)
            temporal_gate = self.temporal_gate(gate_input)

            # Blend cross-attention output
            enriched = cnn_temporal + temporal_gate * (attn_out - cnn_temporal)

        # === FiLM MODE (default) ===
        else:
            # FiLM modulation: compute gamma and beta from engineered features
            gamma = self.gamma_proj(eng_feats)  # [B, D_cnn]
            beta = self.beta_proj(eng_feats)  # [B, D_cnn]

            # Clamp FiLM parameters
            gamma = torch.clamp(gamma, min=-10.0, max=10.0)
            gamma = torch.nan_to_num(gamma, nan=0.0, posinf=10.0, neginf=-10.0)
            beta = torch.clamp(beta, min=-50.0, max=50.0)
            beta = torch.nan_to_num(beta, nan=0.0, posinf=50.0, neginf=-50.0)

            # Broadcast to temporal dimension: [B, D_cnn] -> [B, 1, D_cnn]
            gamma = gamma.unsqueeze(1)  # [B, 1, D_cnn]
            beta = beta.unsqueeze(1)  # [B, 1, D_cnn]

            # Apply FiLM: modulated = (1 + gamma) * cnn_temporal + beta
            # Using (1 + gamma) ensures identity-like behavior when gamma ≈ 0
            modulated = (1.0 + gamma) * cnn_temporal + beta  # [B, T', D_cnn]

            # Clamp modulated output
            modulated = torch.clamp(modulated, min=-100.0, max=100.0)
            modulated = torch.nan_to_num(
                modulated, nan=0.0, posinf=100.0, neginf=-100.0
            )

            # Gated residual: learn how much to incorporate engineered info
            gate_input = torch.cat(
                [cnn_temporal, modulated], dim=-1
            )  # [B, T', 2*D_cnn]
            gate = self.gate(gate_input)  # [B, T', D_cnn]

            # Fuse: preserve original + gated modulation contribution
            enriched = cnn_temporal + gate * (modulated - cnn_temporal)

        # Normalize for stability
        enriched = self.norm(enriched)

        # Final clamp and sanitize
        enriched = torch.clamp(enriched, min=-100.0, max=100.0)
        return torch.nan_to_num(enriched, nan=0.0, posinf=100.0, neginf=-100.0)


class HybridFeatureExtractor(nn.Module):
    """
    Combines CNN-learned features with engineered sleep features.

    This module wraps an existing CNN feature extractor and optionally adds
    engineered sleep-specific features alongside the learned features.

    Args:
        cnn_extractor: Base CNN feature extractor (e.g., FlexibleAsymmetricEpochCNN)
        use_engineered: Whether to extract and concatenate engineered features
        fs: Sampling frequency for engineered features
        epoch_sec: Epoch duration for engineered features
        num_channels: Number of channels to expect (default: 3 for EEG, EOG, EMG)
        fusion_mode: How to combine features:
            - 'concat': Simple concatenation
            - 'gated': Learned gating mechanism
            - 'learned_weight': Learnable scalar weights
            - 'sequential': Progressive 3-stage fusion with cross-attention
        normalize: Normalization strategy for engineered features ('none', 'standardize', 'minmax')

    Output:
        [B, F] where:
        - F = cnn_dim + engineered_dim for 'concat', 'gated', 'learned_weight'
        - F = cnn_dim for 'sequential' mode
    """

    @staticmethod
    def _sanitize_tensor(tensor: torch.Tensor) -> torch.Tensor:
        """
        Sanitize tensor values to prevent NaN/Inf propagation.

        Uses adaptive scaling instead of hard clamping to preserve gradients:
        - If max |value| > 100, scale entire tensor to fit within [-100, 100]
        - This maintains gradient flow (no zero gradients from clamping)
        - NaN/Inf values are replaced after scaling
        """
        if not torch.is_floating_point(tensor):
            return tensor

        # First handle NaN/Inf (must happen before scaling to avoid issues)
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1e6, neginf=-1e6)

        # Adaptive scaling: if out of range, scale to fit (preserves gradients)
        max_abs = tensor.abs().max()
        if max_abs > 100:
            # Scale factor to bring max value to 100
            scale = 100.0 / max_abs.clamp(min=1e-6)
            tensor = tensor * scale

        return tensor

    def __init__(
        self,
        cnn_extractor: nn.Module,
        use_engineered: bool = False,
        fs: int = 128,
        epoch_sec: int = 30,
        num_channels: int = 3,
        fusion_mode: str = "concat",
        normalize: str = "none",
        channel_names: list[str] | None = None,
    ):
        super().__init__()
        self.cnn_extractor = cnn_extractor
        self.use_engineered = use_engineered
        self.fusion_mode = fusion_mode

        # Torch.compile compatibility:
        # - CNN extractor: Fully compilable (convolutions, batch norm, etc.)
        # - Engineered features: Partially compilable
        #   * STFT/ISTFT operations are excluded via @torch_compile_disable decorators
        #   * Other operations (band power computation, feature fusion) can be compiled
        #
        # With PyTorch 2.1+, partial compilation works well with reduce-overhead mode:
        # - Compiled parts run in CUDA graphs for maximum performance
        # - STFT operations run in eager mode (minimal overhead ~1-2%)
        #
        # Set supports_torch_compile=True to enable partial compilation.
        # Models with engineered features will see ~80-90% of operations compiled,
        # with STFT operations running in eager mode for correctness.
        cnn_compile_flag = bool(
            getattr(self.cnn_extractor, "supports_torch_compile", True)
        )
        self.supports_torch_compile = cnn_compile_flag  # Enable partial compilation

        if use_engineered:
            self.sleep_features = SleepFeatureExtractor(
                fs=fs,
                epoch_sec=epoch_sec,
                learnable_bands=True,
                num_channels=num_channels,
                normalize=normalize,
                channel_names=channel_names,
            )

            cnn_out_dim: Any = cnn_extractor.out_dim
            cnn_dim = (
                int(cnn_out_dim.item())
                if isinstance(cnn_out_dim, torch.Tensor)
                else int(cnn_out_dim)
            )
            eng_dim = int(self.sleep_features.out_dim)

            if fusion_mode == "concat":
                self.out_dim = cnn_dim + eng_dim
            elif fusion_mode == "gated":
                # Gated fusion: learned weights for each feature set
                self.gate = nn.Sequential(
                    nn.Linear(cnn_dim + eng_dim, cnn_dim + eng_dim), nn.Sigmoid()
                )
                self.out_dim = cnn_dim + eng_dim
            elif fusion_mode == "learned_weight":
                # Learnable scalar weights for each feature set
                self.cnn_weight = nn.Parameter(torch.ones(1))
                self.eng_weight = nn.Parameter(torch.ones(1))
                self.out_dim = cnn_dim + eng_dim
            elif fusion_mode == "sequential":
                # Progressive 3-stage fusion with cross-attention
                # Project engineered features to match CNN dimension for fusion
                # BottleneckProjection provides: expansion → nonlinearity → compression
                # with residual connection for better gradient flow
                self.feat_projection = BottleneckProjection(
                    in_dim=eng_dim,
                    out_dim=cnn_dim,
                    expansion=2,
                    dropout=0.1,
                )
                # Gated additive fusion for single-epoch features (L=1)
                # Cross-attention with L=1 is meaningless, so use simple gated fusion
                self.fusion_gate = nn.Sequential(
                    nn.Linear(cnn_dim * 2, cnn_dim),
                    nn.Sigmoid(),
                )
                # Output dimension is just cnn_dim (additive fusion preserves cnn_dim)
                self.out_dim = cnn_dim
            else:
                raise ValueError(f"Unknown fusion_mode: {fusion_mode}")

            # Feature normalization before fusion to handle scale mismatch
            # CNN features: large magnitude (learned from raw signals)
            # Engineered features: small magnitude (spectral bands ~0-100, ratios ~0-10)
            self.cnn_norm = nn.LayerNorm(cnn_dim)
            self.eng_norm = nn.LayerNorm(eng_dim)
        else:
            self.sleep_features = None
            cnn_out_dim: Any = cnn_extractor.out_dim
            self.out_dim = (
                int(cnn_out_dim.item())
                if isinstance(cnn_out_dim, torch.Tensor)
                else int(cnn_out_dim)
            )

        self._warned_fallback_wave_raw = (
            False  # Track if we've warned about missing wave_raw
        )

    def gradient_checkpointing_enable(self, granularity: str = "block"):
        """Enable gradient checkpointing in the CNN extractor if supported.

        Args:
            granularity: Checkpoint boundary policy for encoders that expose
                configurable checkpoint granularity.
        """
        fn = getattr(self.cnn_extractor, "gradient_checkpointing_enable", None)
        if callable(fn):
            if granularity == "block":
                fn()
            else:
                fn(granularity=granularity)
        return self

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing in CNN extractor if supported."""
        fn = getattr(self.cnn_extractor, "gradient_checkpointing_disable", None)
        if callable(fn):
            fn()
        return self

    @property
    def temporal_dim(self) -> int:
        """
        Return the temporal dimension from the underlying CNN extractor.

        This is the pre-pooling feature dimension needed for dual-stream
        architectures. Falls back to out_dim if the underlying extractor
        doesn't have temporal_dim.
        """
        if hasattr(self.cnn_extractor, "temporal_dim"):
            return int(self.cnn_extractor.temporal_dim)  # type: ignore[return-value]
        return int(self.cnn_extractor.out_dim)  # type: ignore[return-value]

    @property
    def out_features(self) -> int:
        """
        Return the output feature dimension.

        This property provides a consistent interface for downstream modules
        regardless of fusion mode:
        - 'concat', 'gated', 'learned_weight': cnn_dim + eng_dim
        - 'sequential': cnn_dim only

        Returns:
            Output feature dimension after fusion.
        """
        return int(self.out_dim)

    @torch_compile_disable
    def forward_temporal(
        self,
        x: torch.Tensor | dict[str, Any],
        channel_mask: torch.Tensor | None = None,
        engineered_features: torch.Tensor | None = None,
        recording_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass returning pre-pooling temporal features.

        Note: Decorated with @torch_compile_disable to prevent graph breaks
        when this method is called from compiled code. The underlying CNN
        extractor's forward_temporal is the main compute path.

        Delegates to the underlying CNN extractor's forward_temporal method
        if available. This is required for dual-stream architectures that
        need intra-epoch temporal features.

        Args:
            x: Either:
               - [B, C, T] input waveform tensor
               - dict with 'wave_raw' and 'wave' keys
            channel_mask: Optional [B, C] presence mask (for missing channels)
            engineered_features: Optional [B, E] engineered features (unused here)

        Returns:
            [B, T', D] temporal features where T' is reduced temporal dimension
            and D is the temporal_dim of the underlying CNN.

        Raises:
            AttributeError: If underlying CNN extractor doesn't support forward_temporal
        """
        # Handle dict input
        if isinstance(x, dict):
            wave_for_cnn = x.get("wave")
            if wave_for_cnn is None:
                wave_for_cnn = x.get("wave_raw")
            if wave_for_cnn is None:
                raise ValueError(
                    "Input mapping must contain 'wave' or 'wave_raw' tensors"
                )
        else:
            wave_for_cnn = x

        # Sanitize input
        wave_for_cnn = self._sanitize_tensor(wave_for_cnn)

        # Delegate to underlying CNN extractor
        forward_temporal_fn = getattr(self.cnn_extractor, "forward_temporal", None)
        if callable(forward_temporal_fn):
            return cast(
                torch.Tensor,
                forward_temporal_fn(
                    wave_for_cnn,
                    channel_mask=channel_mask,
                    **(
                        {"recording_index": recording_index}
                        if recording_index is not None
                        else {}
                    ),
                ),
            )
        raise AttributeError(
            f"{self.cnn_extractor.__class__.__name__} does not have forward_temporal method. "
            "Dual-stream intra-epoch attention requires an epoch encoder with forward_temporal support."
        )

    @torch_compile_disable
    def forward(
        self,
        x: torch.Tensor | dict[str, Any],
        channel_mask: torch.Tensor | None = None,
        recording_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass combining CNN and engineered features.

        Note: Decorated with @torch_compile_disable because this method calls
        SleepFeatureExtractor which uses STFT operations that cannot be compiled.
        Running this in eager mode prevents graph breaks and NaN issues.

        CRITICAL: TWO-LEVEL NORMALIZATION
        Both CNN and engineered features use the SAME IQR-normalized waveforms.
        The separate feature normalization layer (standardization) is applied
        to the extracted feature VALUES, not the waveforms.

        Level 1: Waveform Normalization (IQR-based) - BOTH paths
            - Input waveforms are IQR-normalized (median-centered, IQR-scaled)
            - Handles inter-recording amplitude variability
            - Makes signals comparable across different recordings/equipment

        Level 2: Feature Value Normalization (Standardization) - feature values only
            - Takes the extracted feature values (band powers, spindle counts, etc.)
            - Standardizes them to zero-mean, unit-variance
            - Makes different feature types (e.g., alpha power vs spindle count) comparable
            - Handled by SleepFeatureExtractor._normalize_features()

        Args:
            x: Either:
               - [B, C, T] IQR-normalized input waveform
               - dict with 'wave_raw' and 'wave' keys (both contain IQR-normalized data)
            channel_mask: [B, C] or [C] optional channel mask

        Returns:
            [B, F] combined features
        """
        # Handle dict input (backward compatibility: both keys contain same IQR-normalized data)
        engineered_metadata: dict | None = None
        precomputed_features: dict[str, torch.Tensor] | None = None
        if isinstance(x, dict):
            # ``wave`` is the finite CNN input. ``wave_raw`` optionally preserves
            # physical amplitudes for engineered feature extraction.
            wave_for_cnn = x.get("wave")
            wave_for_features = x.get("wave_raw")
            # Extract precomputed YASA features if available
            precomputed_candidate = x.get("precomputed_features")
            if isinstance(precomputed_candidate, dict):
                precomputed_features = cast(
                    dict[str, torch.Tensor], precomputed_candidate
                )
            if wave_for_cnn is None and wave_for_features is None:
                raise ValueError(
                    "Input mapping must contain 'wave' or 'wave_raw' tensors"
                )
            if wave_for_cnn is None:
                wave_for_cnn = wave_for_features
            if wave_for_features is None:
                wave_for_features = wave_for_cnn

                # Warn about missing wave_raw (once per model instance)
                if not self._warned_fallback_wave_raw:
                    warnings.warn(
                        "HybridFeatureExtractor: Input dict missing 'wave_raw' key. "
                        "Falling back to 'wave' for feature extraction. "
                        "This may degrade sleep staging accuracy if 'wave' contains normalized data. "
                        "Fix: Ensure ModelWithPreproc provides 'wave_raw' in input dict.",
                        category=UserWarning,
                        stacklevel=2,
                    )
                    self._warned_fallback_wave_raw = True

            meta_candidate: dict[str, torch.Tensor | list | tuple | str] = {}
            for key in ("subject_ids", "subject_id", "subjects", "subject"):
                if key in x:
                    meta_candidate[key] = x[key]
            if meta_candidate:
                engineered_metadata = meta_candidate
            else:
                metadata_candidate = x.get("metadata")
                if isinstance(metadata_candidate, dict):
                    engineered_metadata = cast(dict, metadata_candidate)
        else:
            # Tensor input: already IQR-normalized and shared across both branches
            wave_for_cnn = x
            wave_for_features = x

        if not isinstance(wave_for_cnn, torch.Tensor) or not isinstance(
            wave_for_features, torch.Tensor
        ):
            raise ValueError(
                "Expected tensor inputs for both CNN and engineered feature branches"
            )

        # Downstream convolutions cannot tolerate NaN/Inf. Sanitize once per branch
        # so both CNN and engineered paths see finite waveforms regardless of input quirks.
        wave_for_cnn = self._sanitize_tensor(wave_for_cnn)
        wave_for_features = self._sanitize_tensor(wave_for_features)

        if not self.use_engineered or self.sleep_features is None:
            # Extract CNN features from channel-embedded IQR-normalized data
            cnn_feats = self.cnn_extractor(
                wave_for_cnn,
                channel_mask=channel_mask,
                **(
                    {"recording_index": recording_index}
                    if recording_index is not None
                    else {}
                ),
            )  # [B, F_cnn]
            cnn_feats = self._sanitize_tensor(cnn_feats)
            return cnn_feats

        # Extract engineered features from SAME IQR-normalized data
        # The separate feature normalization layer (standardization) is applied
        # internally by SleepFeatureExtractor to the extracted feature VALUES
        eng_feats = self.sleep_features(
            wave_for_features,
            channel_mask=channel_mask,
            metadata=engineered_metadata,
            precomputed_features=precomputed_features,
        )  # [B, F_eng]
        eng_feats = self._sanitize_tensor(eng_feats)

        # Extract CNN features from channel-embedded IQR-normalized data.
        # If the CNN supports feature guidance, pass engineered features for FiLM-style modulation.
        cnn_feats = self.cnn_extractor(
            wave_for_cnn,
            channel_mask=channel_mask,
            engineered_features=eng_feats,
            **(
                {"recording_index": recording_index}
                if recording_index is not None
                else {}
            ),
        )  # [B, F_cnn]
        cnn_feats = self._sanitize_tensor(cnn_feats)

        # Apply feature normalization to handle scale mismatch before fusion
        # CNN features: large magnitude, Engineered features: small magnitude
        cnn_feats_norm = self.cnn_norm(cnn_feats)
        eng_feats_norm = self.eng_norm(eng_feats)

        # Combine features based on fusion mode
        if self.fusion_mode == "concat":
            return self._sanitize_tensor(
                torch.cat([cnn_feats_norm, eng_feats_norm], dim=-1)
            )

        elif self.fusion_mode == "gated":
            combined = torch.cat([cnn_feats_norm, eng_feats_norm], dim=-1)
            gate = self.gate(combined)
            fused = combined * gate
            return self._sanitize_tensor(fused)

        elif self.fusion_mode == "learned_weight":
            # Weight each feature set and concatenate
            weighted_cnn = cnn_feats_norm * self.cnn_weight
            weighted_eng = eng_feats_norm * self.eng_weight
            return self._sanitize_tensor(
                torch.cat([weighted_cnn, weighted_eng], dim=-1)
            )

        elif self.fusion_mode == "sequential":
            # Gated additive fusion for single-epoch features
            # Project engineered features to match CNN dimension
            eng_feats_proj = self.feat_projection(eng_feats_norm)  # [B, F_cnn]
            eng_feats_proj = self._sanitize_tensor(eng_feats_proj)

            # Simple gated additive fusion (L=1 attention is meaningless)
            # fused = cnn + gate * eng_proj
            gate = self.fusion_gate(torch.cat([cnn_feats_norm, eng_feats_proj], dim=-1))
            fused = cnn_feats_norm + gate * eng_feats_proj

            return self._sanitize_tensor(fused)

        return self._sanitize_tensor(torch.cat([cnn_feats, eng_feats], dim=-1))
