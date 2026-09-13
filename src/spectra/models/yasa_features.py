"""YASA-inspired feature extraction for sleep staging.

This module provides efficient, GPU-compatible feature extraction matching
YASA's feature set for sleep staging. Features are computed per-epoch per-channel.

Features implemented:
1. Statistical: std, iqr, skewness, kurtosis, zero_crossings
2. Hjorth: mobility, complexity
3. Spectral: slow (0.4-1 Hz), fdelta (1-4 Hz), theta, alpha, sigma, beta
4. Ratios: delta/theta, delta/sigma, delta/beta, alpha/theta, theta/alpha
5. Temporal: theta/alpha ratio, spindle density, slow wave count, rem power, muscle tone, rem/tone ratio

All features can be computed on GPU (torch) or CPU (numpy).

Usage:
    # NumPy (for preprocessing)
    computer = YASAFeatureComputer(fs=128.0)
    features = computer.compute_all(raw_signals)  # raw_signals: [n_epochs, C, T]

    # PyTorch (for real-time fallback)
    extractor = YASAFeatureExtractorTorch(fs=128.0)
    features = extractor(x)  # x: [B, C, T]
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
from scipy import signal as scipy_signal
from scipy import stats as scipy_stats

# Torch.compile compatibility: Prevent STFT compilation
try:
    _torch_compile_disable = torch.compiler.disable
    _COMPILE_DISABLE_AVAILABLE = True
except (ImportError, AttributeError):
    _torch_compile_disable = None
    _COMPILE_DISABLE_AVAILABLE = False


def torch_compile_disable[FuncT: Callable[..., Any]](func: FuncT) -> FuncT:
    """Typed decorator wrapper around torch.compiler.disable."""
    if _COMPILE_DISABLE_AVAILABLE and _torch_compile_disable is not None:
        return cast(FuncT, _torch_compile_disable(func))
    return func


@torch_compile_disable
def _compile_safe_stft(x_flat, n_fft, hop_length, window, **kwargs):
    """Wrapper around torch.stft that prevents torch.compile issues.

    TorchInductor has known bugs with complex FFT operations that can produce
    NaN or fail during backward pass compilation with complex64 dtype errors.
    """
    return torch.stft(
        x_flat, n_fft=n_fft, hop_length=hop_length, window=window, **kwargs
    )


# =============================================================================
# NumPy Implementation (for Zarr preprocessing)
# =============================================================================


class YASAFeatureComputer:
    """Compute YASA-inspired features from raw PSG signals (NumPy).

    This class computes features from raw (unnormalized) signals in microvolts,
    matching YASA's approach. Features are computed per-epoch per-channel.

    Args:
        fs: Sampling frequency in Hz (default: 128.0)
        epoch_sec: Epoch duration in seconds (default: 30.0)
        welch_window_sec: Welch periodogram window size in seconds (default: 5.0)
        welch_overlap: Welch periodogram overlap fraction (default: 0.5)

    Example:
        >>> computer = YASAFeatureComputer(fs=128.0)
        >>> raw_signals = np.random.randn(100, 3, 3840)  # [n_epochs, C, T]
        >>> features = computer.compute_all(raw_signals)
        >>> features['hjorth'].shape       # (100, 3, 2)
        >>> features['spectral'].shape     # (100, 3, 6)
        >>> features['ratios'].shape       # (100, 4)
        >>> features['temporal'].shape     # (100, 3)
    """

    # Frequency band definitions (Hz)
    BANDS = {
        "slow": (0.4, 1.0),  # Slow oscillations (N3 specific)
        "fdelta": (1.0, 4.0),  # Fast delta
        "theta": (4.0, 7.0),  # Theta (was 4-8, YASA uses 4-8 but we use 4-7)
        "alpha": (8.0, 13.0),  # Alpha
        "sigma": (11.0, 16.0),  # Sigma (spindles)
        "beta": (13.0, 30.0),  # Beta
    }

    # Order of bands in output array
    BAND_ORDER = ["slow", "fdelta", "theta", "alpha", "sigma", "beta"]

    def __init__(
        self,
        fs: float = 128.0,
        epoch_sec: float = 30.0,
        welch_window_sec: float = 5.0,
        welch_overlap: float = 0.5,
    ):
        self.fs = fs
        self.epoch_sec = epoch_sec
        self.n_samples = int(fs * epoch_sec)
        self.welch_window_sec = welch_window_sec
        self.welch_overlap = welch_overlap

        # Welch parameters
        self.nperseg = int(welch_window_sec * fs)
        self.noverlap = int(self.nperseg * welch_overlap)

    def compute_statistical(self, x: np.ndarray) -> np.ndarray:
        """Compute statistical features for all epochs.

        DEPRECATED: Statistical features are no longer used in the default pipeline.
        This method is kept for backward compatibility but will be removed in a future version.

        Features:
            0: std - Standard deviation
            1: iqr - Interquartile range (P75 - P25)
            2: skewness - Distribution asymmetry
            3: kurtosis - Distribution tail weight (excess kurtosis)
            4: zero_crossings - Count of sign changes (normalized by epoch length)

        Args:
            x: [n_epochs, C, T] raw signal in microvolts

        Returns:
            [n_epochs, C, 5] statistical features
        """
        n_epochs, n_channels, n_samples = x.shape
        result = np.zeros((n_epochs, n_channels, 5), dtype=np.float32)

        for ep in range(n_epochs):
            for ch in range(n_channels):
                signal = x[ep, ch, :]

                # Standard deviation
                std_val = np.std(signal)
                result[ep, ch, 0] = std_val

                # Interquartile range
                q75, q25 = np.percentile(signal, [75, 25])
                result[ep, ch, 1] = q75 - q25

                # Skewness and Kurtosis: skip if variance is too low to avoid
                # "catastrophic cancellation" warnings on near-constant signals
                if std_val > 1e-10:
                    result[ep, ch, 2] = scipy_stats.skew(signal, nan_policy="omit")
                    result[ep, ch, 3] = scipy_stats.kurtosis(signal, nan_policy="omit")
                else:
                    # Near-constant signal: skewness=0 (symmetric), kurtosis=0 (normal-like)
                    result[ep, ch, 2] = 0.0
                    result[ep, ch, 3] = 0.0

                # Zero crossings (normalized by epoch length for comparability)
                # Count sign changes
                sign_changes = np.diff(np.sign(signal))
                nzc = np.sum(sign_changes != 0)
                # Normalize to crossings per second
                result[ep, ch, 4] = nzc / self.epoch_sec

        # Handle NaN/Inf
        result = np.nan_to_num(result, nan=0.0, posinf=10.0, neginf=-10.0)

        return result

    def compute_hjorth(self, x: np.ndarray) -> np.ndarray:
        """Compute Hjorth parameters for all epochs.

        Hjorth parameters are time-domain measures of signal dynamics:
            - Mobility: Related to mean frequency (sqrt of ratio of variances)
            - Complexity: Bandwidth of the signal (ratio of mobilities)

        Features:
            0: mobility - sqrt(var(dx) / var(x))
            1: complexity - mobility(dx) / mobility(x)

        Args:
            x: [n_epochs, C, T] raw signal in microvolts

        Returns:
            [n_epochs, C, 2] Hjorth parameters (mobility, complexity)
        """
        n_epochs, n_channels, n_samples = x.shape
        result = np.zeros((n_epochs, n_channels, 2), dtype=np.float32)

        for ep in range(n_epochs):
            for ch in range(n_channels):
                signal = x[ep, ch, :]

                # First derivative
                dx = np.diff(signal)
                # Second derivative
                ddx = np.diff(dx)

                # Variances (with small epsilon to avoid division by zero)
                var_x = np.var(signal) + 1e-10
                var_dx = np.var(dx) + 1e-10
                var_ddx = np.var(ddx) + 1e-10

                # Mobility = sqrt(var(dx) / var(x))
                mobility_x = np.sqrt(var_dx / var_x)
                result[ep, ch, 0] = mobility_x

                # Mobility of first derivative
                mobility_dx = np.sqrt(var_ddx / var_dx)

                # Complexity = mobility(dx) / mobility(x)
                complexity = mobility_dx / (mobility_x + 1e-10)
                result[ep, ch, 1] = complexity

        # Handle NaN/Inf
        result = np.nan_to_num(result, nan=0.0, posinf=10.0, neginf=-10.0)

        return result

    def compute_spectral_bands(self, x: np.ndarray) -> np.ndarray:
        """Compute spectral band powers using Welch periodogram.

        Uses scipy.signal.welch with 5s window and 50% overlap (matching YASA).
        Band powers are computed as log-transformed absolute power (log10(µV²/Hz)).

        Features (band order):
            0: slow (0.4-1 Hz) - Slow oscillations, N3 specific
            1: fdelta (1-4 Hz) - Fast delta
            2: theta (4-7 Hz) - Theta
            3: alpha (8-13 Hz) - Alpha
            4: sigma (11-16 Hz) - Sigma/spindles
            5: beta (13-30 Hz) - Beta

        Args:
            x: [n_epochs, C, T] raw signal in microvolts

        Returns:
            [n_epochs, C, 6] log-transformed band powers
        """
        n_epochs, n_channels, n_samples = x.shape
        n_bands = len(self.BAND_ORDER)
        result = np.zeros((n_epochs, n_channels, n_bands), dtype=np.float32)

        for ep in range(n_epochs):
            for ch in range(n_channels):
                signal = x[ep, ch, :]

                # Compute Welch periodogram
                freqs, psd = scipy_signal.welch(
                    signal,
                    fs=self.fs,
                    nperseg=self.nperseg,
                    noverlap=self.noverlap,
                    window="hann",
                    scaling="density",
                )

                # Extract band powers
                for band_idx, band_name in enumerate(self.BAND_ORDER):
                    low, high = self.BANDS[band_name]

                    # Find frequency indices for this band
                    band_mask = (freqs >= low) & (freqs <= high)

                    if np.any(band_mask):
                        # Integrate power in band (trapezoidal)
                        band_power = np.trapezoid(psd[band_mask], freqs[band_mask])
                        # Log transform (add small epsilon to avoid log(0))
                        result[ep, ch, band_idx] = np.log10(band_power + 1e-10)
                    else:
                        result[ep, ch, band_idx] = -10.0  # Very low power

        # Handle NaN/Inf
        result = np.nan_to_num(result, nan=-10.0, posinf=10.0, neginf=-10.0)

        return result

    def compute_power_ratios(
        self,
        spectral: np.ndarray,
        eeg_channel_idx: int = 0,
    ) -> np.ndarray:
        """Compute inter-band power ratios.

        Ratios are computed from the spectral band powers of a single EEG channel.
        Uses delta = slow + fdelta for ratio computations.

        Features:
            0: delta/theta - (slow + fdelta) / theta
            1: delta/sigma - (slow + fdelta) / sigma
            2: delta/beta - (slow + fdelta) / beta
            3: theta/alpha - theta / alpha

        Note: alpha/theta was removed (redundant with theta/alpha).

        Args:
            spectral: [n_epochs, C, 6] spectral band powers (from compute_spectral_bands)
            eeg_channel_idx: Which channel to use for ratios (default: 0, first EEG)

        Returns:
            [n_epochs, 4] power ratios
        """
        n_epochs = spectral.shape[0]
        result = np.zeros((n_epochs, 4), dtype=np.float32)

        # Band indices in spectral array
        IDX_SLOW = 0
        IDX_FDELTA = 1
        IDX_THETA = 2
        IDX_ALPHA = 3
        IDX_SIGMA = 4
        IDX_BETA = 5

        for ep in range(n_epochs):
            # Get band powers for this epoch (already log-transformed)
            # Convert back to linear for ratio computation
            bands = spectral[ep, eeg_channel_idx, :]
            bands_linear = np.power(10, bands)  # Undo log10

            slow = bands_linear[IDX_SLOW]
            fdelta = bands_linear[IDX_FDELTA]
            theta = bands_linear[IDX_THETA]
            alpha = bands_linear[IDX_ALPHA]
            sigma = bands_linear[IDX_SIGMA]
            beta = bands_linear[IDX_BETA]

            # Combined delta = slow + fdelta
            delta_total = slow + fdelta

            # Compute ratios (with epsilon to avoid division by zero)
            eps = 1e-10

            # delta/theta
            result[ep, 0] = np.log10((delta_total + eps) / (theta + eps))
            # delta/sigma
            result[ep, 1] = np.log10((delta_total + eps) / (sigma + eps))
            # delta/beta
            result[ep, 2] = np.log10((delta_total + eps) / (beta + eps))
            # theta/alpha
            result[ep, 3] = np.log10((theta + eps) / (alpha + eps))

        # Handle NaN/Inf
        result = np.nan_to_num(result, nan=0.0, posinf=10.0, neginf=-10.0)

        return result

    def _bandpass_filter(
        self,
        x: np.ndarray,
        low_freq: float,
        high_freq: float,
        order: int = 4,
    ) -> np.ndarray:
        """Apply bandpass filter using scipy.

        Args:
            x: [n_epochs, C, T] or [T] signal
            low_freq: Low cutoff frequency (Hz)
            high_freq: High cutoff frequency (Hz)
            order: Filter order (default: 4)

        Returns:
            Filtered signal with same shape as input
        """
        nyq = self.fs / 2.0
        low = low_freq / nyq
        high = high_freq / nyq

        # Clamp to valid range
        low = max(0.001, min(low, 0.99))
        high = max(low + 0.01, min(high, 0.99))

        ba = scipy_signal.butter(order, [low, high], btype="band", output="ba")
        b, a = cast(tuple[np.ndarray, np.ndarray], ba)
        return scipy_signal.filtfilt(b, a, x, axis=-1).astype(np.float32)

    def compute_spindle_features(self, x: np.ndarray) -> np.ndarray:
        """Compute spindle-related features from sigma band envelope.

        Features (per channel):
            0: mean - Mean envelope (spindle density proxy)
            1: std - Envelope variability (burstiness)
            2: max - Peak envelope (spindle intensity)

        Args:
            x: [n_epochs, C, T] raw signal in microvolts

        Returns:
            [n_epochs, C, 3] spindle features
        """
        n_epochs, n_channels, n_samples = x.shape
        result = np.zeros((n_epochs, n_channels, 3), dtype=np.float32)

        # Filter in sigma band (11-16 Hz)
        for ep in range(n_epochs):
            for ch in range(n_channels):
                signal = x[ep, ch, :]

                try:
                    sigma_filtered = self._bandpass_filter(signal, 11.0, 16.0)

                    # Compute envelope (squared + smoothing)
                    envelope = sigma_filtered**2
                    # Moving average smoothing (0.5s window)
                    window_size = max(1, int(0.5 * self.fs))
                    kernel = np.ones(window_size) / window_size
                    envelope = np.convolve(envelope, kernel, mode="same")

                    # Extract statistics
                    result[ep, ch, 0] = np.mean(envelope)
                    result[ep, ch, 1] = np.std(envelope)
                    result[ep, ch, 2] = np.max(envelope)
                except Exception:
                    # Filter failed, use zeros
                    pass

        result = np.nan_to_num(result, nan=0.0, posinf=10.0, neginf=0.0)
        return result

    def compute_slow_wave_features(self, x: np.ndarray) -> np.ndarray:
        """Compute slow wave activity features (N3 marker).

        Features (per channel):
            0: mean - Mean envelope (slow wave density)
            1: std - Envelope variability
            2: max - Peak envelope (slow wave intensity)

        Args:
            x: [n_epochs, C, T] raw signal in microvolts

        Returns:
            [n_epochs, C, 3] slow wave features
        """
        n_epochs, n_channels, n_samples = x.shape
        result = np.zeros((n_epochs, n_channels, 3), dtype=np.float32)

        # Filter in slow wave band (0.5-2 Hz)
        for ep in range(n_epochs):
            for ch in range(n_channels):
                signal = x[ep, ch, :]

                try:
                    slow_filtered = self._bandpass_filter(signal, 0.5, 2.0, order=2)

                    # Compute envelope (absolute + smoothing)
                    envelope = np.abs(slow_filtered)
                    # Moving average smoothing (1.0s window)
                    window_size = max(1, int(1.0 * self.fs))
                    kernel = np.ones(window_size) / window_size
                    envelope = np.convolve(envelope, kernel, mode="same")

                    # Extract statistics
                    result[ep, ch, 0] = np.mean(envelope)
                    result[ep, ch, 1] = np.std(envelope)
                    result[ep, ch, 2] = np.max(envelope)
                except Exception:
                    # Filter failed, use zeros
                    pass

        result = np.nan_to_num(result, nan=0.0, posinf=10.0, neginf=0.0)
        return result

    def compute_spindle_count_and_frequency(
        self,
        x: np.ndarray,
        eeg_channel_idx: int = 0,
    ) -> np.ndarray:
        """Detect individual spindles and compute count, frequency, and amplitude.

        Uses sigma-band (11-16 Hz) filtering with Hilbert envelope and
        threshold-based detection (mean + 1.5*std), filtering for 0.5-2.0s duration.

        Features (3 total, from a single EEG channel):
            0: spindle_count - Number of detected spindles in epoch
            1: mean_spindle_freq - Mean instantaneous frequency of spindles (Hz)
            2: spindle_amplitude - Mean peak amplitude of detected spindles

        Args:
            x: [n_epochs, C, T] raw signal in microvolts
            eeg_channel_idx: Which channel to use (default: 0)

        Returns:
            [n_epochs, 3] spindle count, frequency, amplitude
        """
        from scipy import ndimage

        n_epochs = x.shape[0]
        n_channels = x.shape[1]
        eeg_idx = min(eeg_channel_idx, n_channels - 1)
        result = np.zeros((n_epochs, 3), dtype=np.float32)

        min_dur_samples = int(0.5 * self.fs)
        max_dur_samples = int(2.0 * self.fs)

        for ep in range(n_epochs):
            signal = x[ep, eeg_idx, :]
            try:
                filtered = self._bandpass_filter(signal, 11.0, 16.0)
                analytic: np.ndarray = np.asarray(scipy_signal.hilbert(filtered))
                envelope: np.ndarray = np.abs(analytic)

                threshold = float(np.mean(envelope) + 1.5 * np.std(envelope))
                above = (envelope > threshold).astype(np.int32)

                label_result: tuple[np.ndarray, int] = ndimage.label(above)  # type: ignore[assignment]
                labeled = label_result[0]
                n_regions = label_result[1]

                spindle_count = 0
                freqs_list: list[float] = []
                amps_list: list[float] = []

                for region_id in range(1, n_regions + 1):
                    region_mask = labeled == region_id
                    dur = int(region_mask.sum())
                    if dur < min_dur_samples or dur > max_dur_samples:
                        continue

                    spindle_count += 1
                    amps_list.append(float(np.max(envelope[region_mask])))

                    # Instantaneous frequency from Hilbert phase
                    phase = np.unwrap(np.angle(analytic[region_mask]))
                    if len(phase) > 1:
                        inst_freq = np.diff(phase) / (2.0 * np.pi / self.fs)
                        valid_freq = inst_freq[(inst_freq > 8) & (inst_freq < 20)]
                        if len(valid_freq) > 0:
                            freqs_list.append(float(np.mean(valid_freq)))

                result[ep, 0] = spindle_count
                result[ep, 1] = float(np.mean(freqs_list)) if freqs_list else 0.0
                result[ep, 2] = float(np.mean(amps_list)) if amps_list else 0.0
            except Exception:
                pass

        result = np.nan_to_num(result, nan=0.0, posinf=10.0, neginf=0.0)
        return result

    def compute_slow_wave_metrics(
        self,
        x: np.ndarray,
        eeg_channel_idx: int = 0,
    ) -> np.ndarray:
        """Compute slow wave metrics from 0.5-2 Hz band.

        Detects negative half-waves via zero crossings in the slow-wave band
        and computes amplitude, slope, and density metrics.

        Features (3 total, from a single EEG channel):
            0: swa_amplitude - Mean peak-to-peak amplitude of detected slow waves
            1: swa_slope - Mean amplitude/duration slope of slow waves
            2: swa_density - Number of slow waves per epoch

        Args:
            x: [n_epochs, C, T] raw signal in microvolts
            eeg_channel_idx: Which channel to use (default: 0)

        Returns:
            [n_epochs, 3] slow wave amplitude, slope, density
        """
        n_epochs = x.shape[0]
        n_channels = x.shape[1]
        eeg_idx = min(eeg_channel_idx, n_channels - 1)
        result = np.zeros((n_epochs, 3), dtype=np.float32)

        for ep in range(n_epochs):
            signal = x[ep, eeg_idx, :]
            try:
                filtered = self._bandpass_filter(signal, 0.5, 2.0, order=2)

                # Find zero crossings (negative-to-positive and positive-to-negative)
                sign = np.sign(filtered)
                sign_changes = np.diff(sign)
                neg_to_pos = np.where(sign_changes > 0)[0]  # negative -> positive
                pos_to_neg = np.where(sign_changes < 0)[0]  # positive -> negative

                if len(pos_to_neg) < 1 or len(neg_to_pos) < 1:
                    continue

                amplitudes: list[float] = []
                slopes: list[float] = []

                # For each positive-to-negative crossing, find the next negative-to-positive
                for p2n in pos_to_neg:
                    # Find the next neg_to_pos after this pos_to_neg
                    candidates = neg_to_pos[neg_to_pos > p2n]
                    if len(candidates) == 0:
                        continue
                    n2p = candidates[0]

                    # This is one negative half-wave from p2n to n2p
                    half_wave = filtered[p2n : n2p + 1]
                    if len(half_wave) < 2:
                        continue

                    trough = float(np.min(half_wave))
                    ptp = float(abs(trough))  # peak-to-trough amplitude
                    dur_sec = len(half_wave) / self.fs

                    # Filter for plausible slow waves (duration 0.25-1.0s, amplitude > 10 µV)
                    if 0.25 <= dur_sec <= 1.0 and ptp > 10.0:
                        amplitudes.append(ptp)
                        slopes.append(ptp / dur_sec if dur_sec > 0 else 0.0)

                result[ep, 0] = float(np.mean(amplitudes)) if amplitudes else 0.0
                result[ep, 1] = float(np.mean(slopes)) if slopes else 0.0
                result[ep, 2] = float(len(amplitudes))
            except Exception:
                pass

        result = np.nan_to_num(result, nan=0.0, posinf=10.0, neginf=0.0)
        return result

    def compute_n3_criterion_features(
        self,
        x: np.ndarray,
        eeg_channel_idx: int = 0,
        ptp_thresholds: tuple[float, ...] = (2.0, 2.5, 3.0, 3.5, 4.0),
        occupancy_threshold: float = 0.20,
    ) -> np.ndarray:
        """Compute multi-threshold N3 occupancy features from normalized EEG.

        This feature family approximates the AASM N3 >=20% slow-wave occupancy rule
        using relative peak-to-peak thresholds on IQR-normalized signals.

        Features (2 per threshold, interleaved):
            [occ_t0, soft_t0, occ_t1, soft_t1, ...]
            - occ_ti: Fraction of epoch samples occupied by qualifying slow waves
            - soft_ti: Sigmoid soft indicator for occupancy >= occupancy_threshold

        Args:
            x: [n_epochs, C, T] IQR-normalized signal (NOT raw microvolts)
            eeg_channel_idx: Which channel to use (default: 0)
            ptp_thresholds: Peak-to-peak thresholds in normalized units
            occupancy_threshold: Target occupancy fraction (AASM default: 0.20)

        Returns:
            [n_epochs, 2 * len(ptp_thresholds)] occupancy + soft indicators
        """
        n_epochs = x.shape[0]
        n_channels = x.shape[1]
        n_samples = x.shape[2]
        eeg_idx = min(eeg_channel_idx, n_channels - 1)
        n_thresholds = len(ptp_thresholds)
        result = np.zeros((n_epochs, 2 * n_thresholds), dtype=np.float32)

        steepness = 20.0
        min_dur_samples = int(0.5 * self.fs)
        max_dur_samples = int(2.0 * self.fs)

        for ep in range(n_epochs):
            signal = x[ep, eeg_idx, :]
            try:
                filtered = self._bandpass_filter(signal, 0.5, 2.0, order=2)

                # Full wave cycles are defined by consecutive positive->negative zero crossings
                sign = np.sign(filtered)
                sign_changes = np.diff(sign)
                pos_to_neg = np.where(sign_changes < 0)[0]

                if len(pos_to_neg) < 2:
                    continue

                wave_ptps: list[float] = []
                wave_lengths: list[float] = []

                for idx in range(len(pos_to_neg) - 1):
                    start = int(pos_to_neg[idx])
                    end = int(pos_to_neg[idx + 1])
                    wave_len = end - start
                    if wave_len <= 0:
                        continue

                    if wave_len < min_dur_samples or wave_len > max_dur_samples:
                        continue

                    segment = filtered[start:end]
                    if len(segment) < 2:
                        continue

                    ptp = float(np.max(segment) - np.min(segment))
                    wave_ptps.append(ptp)
                    wave_lengths.append(float(wave_len))

                if not wave_ptps:
                    continue

                wave_ptps_arr = np.asarray(wave_ptps, dtype=np.float32)
                wave_lengths_arr = np.asarray(wave_lengths, dtype=np.float32)

                for thr_idx, threshold in enumerate(ptp_thresholds):
                    qualifying_samples = float(
                        wave_lengths_arr[wave_ptps_arr >= threshold].sum()
                    )
                    occupancy = qualifying_samples / max(float(n_samples), 1.0)
                    occupancy = float(np.clip(occupancy, 0.0, 1.0))
                    n3_soft = float(
                        1.0
                        / (
                            1.0
                            + np.exp(
                                -steepness * (occupancy - float(occupancy_threshold))
                            )
                        )
                    )

                    out_idx = 2 * thr_idx
                    result[ep, out_idx] = occupancy
                    result[ep, out_idx + 1] = n3_soft
            except Exception:
                pass

        result = np.nan_to_num(result, nan=0.0, posinf=10.0, neginf=0.0)
        return result

    def compute_temporal_features(
        self,
        x: np.ndarray,
        spectral: np.ndarray,
        eog_channel_idx: int = 1,
        emg_channel_idx: int = 2,
        eeg_channel_idx: int = 0,
    ) -> np.ndarray:
        """Compute temporal features used by SleepFeatureExtractor.

        These features complement YASA spectral features with time-domain analysis.

        Features (3 total):
            0: spindle_density - Mean sigma envelope (from first EEG channel)
            1: rem_power - log1p(REM band power from EOG)
            2: muscle_tone - log1p(EMG high-freq power)

        Note: theta_alpha_ratio is computed in ratios (theta/alpha).
        Note: slow_wave_count and rem_tone_ratio were removed to reduce feature set.

        Args:
            x: [n_epochs, C, T] raw signal in microvolts
            spectral: [n_epochs, C, 6] spectral band powers (from compute_spectral_bands)
            eog_channel_idx: EOG channel index (default: 1)
            emg_channel_idx: EMG channel index (default: 2)
            eeg_channel_idx: EEG channel index (default: 0, unused but kept for API compatibility)

        Returns:
            [n_epochs, 3] temporal features
        """
        n_epochs, n_channels, n_samples = x.shape
        result = np.zeros((n_epochs, 3), dtype=np.float32)

        # Clamp channel indices to valid range
        eog_idx = min(eog_channel_idx, n_channels - 1)
        emg_idx = min(emg_channel_idx, n_channels - 1)
        eeg_idx = min(eeg_channel_idx, n_channels - 1)

        # Precompute spindle features (slow wave features removed)
        spindle_features = self.compute_spindle_features(x)  # [n_epochs, C, 3]

        for ep in range(n_epochs):
            # 0. Spindle density (mean envelope from first EEG channel)
            result[ep, 0] = spindle_features[ep, eeg_idx, 0]

            # 1-2. REM/EMG features
            try:
                # REM power from EOG (0.5-2 Hz)
                eog_signal = x[ep, eog_idx, :]
                rem_filtered = self._bandpass_filter(eog_signal, 0.5, 2.0, order=2)
                rem_power = np.mean(rem_filtered**2)

                # Muscle tone from EMG (30-50 Hz)
                emg_signal = x[ep, emg_idx, :]
                tone_filtered = self._bandpass_filter(emg_signal, 30.0, 50.0)
                muscle_tone = np.mean(tone_filtered**2)

                # Log transforms
                result[ep, 1] = np.log1p(rem_power * 1e6)
                result[ep, 2] = np.log1p(muscle_tone * 1e6)
            except Exception:
                # Filter failed, use zeros
                pass

        result = np.nan_to_num(result, nan=0.0, posinf=10.0, neginf=-10.0)
        return result

    def compute_all(
        self,
        x: np.ndarray,
        eeg_channel_idx: int = 0,
        eog_channel_idx: int = 1,
        emg_channel_idx: int = 2,
    ) -> dict[str, np.ndarray]:
        """Compute all YASA-inspired features.

        Args:
            x: [n_epochs, C, T] raw signal in microvolts
            eeg_channel_idx: Which channel to use for EEG features (default: 0)
            eog_channel_idx: EOG channel index for REM features (default: 1)
            emg_channel_idx: EMG channel index for muscle tone (default: 2)

        Returns:
            Dict with keys:
                'hjorth': [n_epochs, C, 2] - mobility, complexity
                'spectral': [n_epochs, C, 6] - slow, fdelta, theta, alpha, sigma, beta
                'ratios': [n_epochs, 4] - d/t, d/s, d/b, t/a
                'temporal': [n_epochs, 3] - spindle, rem, tone
                'spindle': [n_epochs, C, 3] - sigma envelope stats
                'slow_wave': [n_epochs, C, 3] - slow-wave envelope stats
                'spindle_count_freq': [n_epochs, 3] - count, freq, amplitude
                'slow_wave_metrics': [n_epochs, 3] - amplitude, slope, density
                'n3_criterion': [n_epochs, 10] - multi-threshold occupancy + soft flags

        Note: Statistical features have been removed from the default feature set.
        """
        hjorth = self.compute_hjorth(x)
        spectral = self.compute_spectral_bands(x)
        ratios = self.compute_power_ratios(spectral, eeg_channel_idx)
        temporal = self.compute_temporal_features(
            x, spectral, eog_channel_idx, emg_channel_idx, eeg_channel_idx
        )
        spindle = self.compute_spindle_features(x)
        slow_wave = self.compute_slow_wave_features(x)
        spindle_count_freq = self.compute_spindle_count_and_frequency(
            x, eeg_channel_idx
        )
        slow_wave_metrics = self.compute_slow_wave_metrics(x, eeg_channel_idx)
        # NOTE: compute_n3_criterion_features is designed for IQR-normalized input and
        # relative PTP thresholds. If compute_all is called with raw microvolts, effective
        # thresholds differ. This is acceptable when features are recomputed from normalized
        # data at training time, or when preprocessing provides normalized data for this path.
        n3_criterion = self.compute_n3_criterion_features(x, eeg_channel_idx)

        return {
            "hjorth": hjorth,
            "spectral": spectral,
            "ratios": ratios,
            "temporal": temporal,
            "spindle": spindle,
            "slow_wave": slow_wave,
            "spindle_count_freq": spindle_count_freq,
            "slow_wave_metrics": slow_wave_metrics,
            "n3_criterion": n3_criterion,
        }


# =============================================================================
# PyTorch Implementation (for real-time fallback)
# =============================================================================


class YASAFeatureExtractorTorch(nn.Module):
    """PyTorch implementation for real-time YASA feature extraction.

    This is a fallback for when precomputed features are not available.
    Note: Features computed from normalized signals will be approximations
    since raw amplitude information is lost.

    Args:
        fs: Sampling frequency in Hz (default: 128.0)
        epoch_sec: Epoch duration in seconds (default: 30.0)
        include_statistical: Include statistical features (default: False, deprecated)
        include_hjorth: Include Hjorth parameters (default: True)
        include_spectral: Include spectral bands (default: True)
        include_ratios: Include power ratios (default: True)

    Example:
        >>> extractor = YASAFeatureExtractorTorch(fs=128.0)
        >>> x = torch.randn(8, 3, 3840)  # [B, C, T]
        >>> features = extractor(x)
        >>> features['hjorth'].shape       # torch.Size([8, 3, 2])
        >>> features['spectral'].shape     # torch.Size([8, 3, 6])
    """

    # Same band definitions as NumPy version
    BANDS = {
        "slow": (0.4, 1.0),
        "fdelta": (1.0, 4.0),
        "theta": (4.0, 7.0),
        "alpha": (8.0, 13.0),
        "sigma": (11.0, 16.0),
        "beta": (13.0, 30.0),
    }
    BAND_ORDER = ["slow", "fdelta", "theta", "alpha", "sigma", "beta"]

    def __init__(
        self,
        fs: float = 128.0,
        epoch_sec: float = 30.0,
        include_statistical: bool = False,  # Deprecated, disabled by default
        include_hjorth: bool = True,
        include_spectral: bool = True,
        include_ratios: bool = True,
    ):
        super().__init__()
        self.fs = fs
        self.epoch_sec = epoch_sec
        self.n_samples = int(fs * epoch_sec)
        self.include_statistical = include_statistical
        self.include_hjorth = include_hjorth
        self.include_spectral = include_spectral
        self.include_ratios = include_ratios

        # STFT parameters for spectral features
        # Use n_fft ~ 5 seconds for comparable resolution to Welch
        self.n_fft = int(5.0 * fs)
        if self.n_fft % 2 != 0:
            self.n_fft += 1  # Make even for efficiency
        self.hop_length = self.n_fft // 2

        # Register Hann window as buffer
        self.register_buffer("window", torch.hann_window(self.n_fft))

        # Pre-compute frequency bins
        freqs = torch.fft.rfftfreq(self.n_fft, d=1.0 / fs)
        self.register_buffer("freqs", freqs)

        # Pre-compute band masks
        for band_name, (low, high) in self.BANDS.items():
            mask = (freqs >= low) & (freqs <= high)
            self.register_buffer(f"mask_{band_name}", mask)

        # Clip threshold for robust scaling (matches preprocessing in batch_edf_to_zarrfp32.py)
        self._robust_scale_clip = 20.0

    def _robust_scale_and_clip(
        self, x: torch.Tensor, dim: int = 0, clip_threshold: float | None = None
    ) -> torch.Tensor:
        """Apply robust scaling (median/IQR normalization) with clipping.

        This matches the normalization used in batch_edf_to_zarrfp32.py for
        precomputed features. Applied per-feature across the batch dimension.

        Args:
            x: Input tensor
            dim: Dimension to compute statistics over (default: 0 = batch)
            clip_threshold: Clip normalized values to ±threshold (default: 20.0)

        Returns:
            Normalized and clipped tensor
        """
        if clip_threshold is None:
            clip_threshold = self._robust_scale_clip

        # Compute median and IQR across the specified dimension
        median = torch.median(x, dim=dim, keepdim=True).values
        q75 = torch.quantile(x, 0.75, dim=dim, keepdim=True)
        q25 = torch.quantile(x, 0.25, dim=dim, keepdim=True)
        iqr = (q75 - q25).clamp(min=1e-6)  # Avoid division by zero

        # Normalize: (x - median) / IQR
        normalized = (x - median) / iqr

        # Clip to ±threshold
        clipped = torch.clamp(normalized, -clip_threshold, clip_threshold)

        # Sanitize any remaining NaN/Inf
        return torch.nan_to_num(
            clipped, nan=0.0, posinf=clip_threshold, neginf=-clip_threshold
        )

    def _compute_statistical(self, x: torch.Tensor) -> torch.Tensor:
        """Compute statistical features.

        Args:
            x: [B, C, T] signal

        Returns:
            [B, C, 5] statistical features
        """
        B, C, T = x.shape
        device = x.device
        dtype = x.dtype

        result = torch.zeros(B, C, 5, device=device, dtype=dtype)

        # 0: Standard deviation
        result[:, :, 0] = x.std(dim=-1)

        # 1: IQR (P75 - P25)
        q75 = torch.quantile(x, 0.75, dim=-1)
        q25 = torch.quantile(x, 0.25, dim=-1)
        result[:, :, 1] = q75 - q25

        # 2: Skewness
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True).clamp(min=1e-10)
        z = (x - mean) / std
        result[:, :, 2] = (z**3).mean(dim=-1)

        # 3: Kurtosis (excess)
        result[:, :, 3] = (z**4).mean(dim=-1) - 3.0

        # 4: Zero crossings (normalized)
        signs = torch.sign(x)
        sign_diff = torch.diff(signs, dim=-1)
        nzc = (sign_diff != 0).sum(dim=-1).float()
        result[:, :, 4] = nzc / self.epoch_sec

        # Sanitize
        result = torch.nan_to_num(result, nan=0.0, posinf=10.0, neginf=-10.0)

        return result

    def _compute_hjorth(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Hjorth parameters.

        Args:
            x: [B, C, T] signal

        Returns:
            [B, C, 2] Hjorth parameters (mobility, complexity)
        """
        B, C, T = x.shape
        device = x.device
        dtype = x.dtype

        result = torch.zeros(B, C, 2, device=device, dtype=dtype)

        # First derivative
        dx = torch.diff(x, dim=-1)
        # Second derivative
        ddx = torch.diff(dx, dim=-1)

        # Variances
        var_x = x.var(dim=-1).clamp(min=1e-10)
        var_dx = dx.var(dim=-1).clamp(min=1e-10)
        var_ddx = ddx.var(dim=-1).clamp(min=1e-10)

        # Mobility = sqrt(var(dx) / var(x))
        mobility_x = torch.sqrt(var_dx / var_x)
        result[:, :, 0] = mobility_x

        # Mobility of dx
        mobility_dx = torch.sqrt(var_ddx / var_dx)

        # Complexity = mobility(dx) / mobility(x)
        result[:, :, 1] = mobility_dx / mobility_x.clamp(min=1e-10)

        # Sanitize
        result = torch.nan_to_num(result, nan=0.0, posinf=10.0, neginf=-10.0)

        return result

    def _compute_spectral(self, x: torch.Tensor) -> torch.Tensor:
        """Compute spectral band powers using STFT.

        Computes log10-transformed band powers. No batch-level normalization
        is applied so that the same epoch always produces the same features
        regardless of batch composition.

        Args:
            x: [B, C, T] signal

        Returns:
            [B, C, 6] log-scaled band powers (slow, fdelta, theta, alpha, sigma, beta)
        """
        B, C, T = x.shape
        device = x.device
        dtype = x.dtype

        n_bands = len(self.BAND_ORDER)
        result = torch.zeros(B, C, n_bands, device=device, dtype=dtype)

        # Flatten B and C for batched STFT
        x_flat = x.reshape(B * C, T)

        # Compute STFT (using compile-safe wrapper to avoid torch.compile issues)
        stft_out = _compile_safe_stft(
            x_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True,
            center=True,
            pad_mode="reflect",
        )

        # Power spectrum: |STFT|^2
        power = stft_out.abs().pow(2)  # [B*C, n_freqs, n_frames]

        # Average over time frames
        power_avg = power.mean(dim=-1)  # [B*C, n_freqs]

        # Reshape back
        power_avg = power_avg.reshape(B, C, -1)  # [B, C, n_freqs]

        # Extract band powers
        for band_idx, band_name in enumerate(self.BAND_ORDER):
            mask = getattr(self, f"mask_{band_name}")
            if mask.sum() > 0:
                # Sum power in band
                band_power = power_avg[:, :, mask].sum(dim=-1)
                # Log transform
                result[:, :, band_idx] = torch.log10(band_power.clamp(min=1e-10))
            else:
                result[:, :, band_idx] = -10.0

        # Sanitize
        result = torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=-10.0)

        return result

    def _compute_ratios(
        self,
        spectral: torch.Tensor,
        eeg_channel_idx: int = 0,
    ) -> torch.Tensor:
        """Compute power ratios from spectral features.

        Args:
            spectral: [B, C, 6] spectral band powers
            eeg_channel_idx: Which channel to use for ratios

        Returns:
            [B, 4] power ratios (alpha/theta removed, redundant with theta/alpha)
        """
        B = spectral.size(0)
        device = spectral.device
        dtype = spectral.dtype

        result = torch.zeros(B, 4, device=device, dtype=dtype)

        # Get bands for EEG channel (convert from log10)
        bands = spectral[:, eeg_channel_idx, :]  # [B, 6]
        bands_linear = torch.pow(10, bands)

        slow = bands_linear[:, 0]
        fdelta = bands_linear[:, 1]
        theta = bands_linear[:, 2]
        alpha = bands_linear[:, 3]
        sigma = bands_linear[:, 4]
        beta = bands_linear[:, 5]

        delta_total = slow + fdelta
        eps = 1e-10

        # Log ratios (alpha/theta removed - redundant with theta/alpha)
        result[:, 0] = torch.log10((delta_total + eps) / (theta + eps))
        result[:, 1] = torch.log10((delta_total + eps) / (sigma + eps))
        result[:, 2] = torch.log10((delta_total + eps) / (beta + eps))
        result[:, 3] = torch.log10((theta + eps) / (alpha + eps))

        # Sanitize
        result = torch.nan_to_num(result, nan=0.0, posinf=10.0, neginf=-10.0)

        return result

    @torch_compile_disable
    def forward(
        self,
        x: torch.Tensor,
        channel_mask: torch.Tensor | None = None,
        eeg_channel_idx: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Extract YASA features from signal.

        Args:
            x: [B, C, T] input signal
            channel_mask: [B, C] optional binary mask (1=present, 0=missing)
            eeg_channel_idx: Which channel to use for power ratios

        Returns:
            Dict with keys (depending on config):
                'hjorth': [B, C, 2]
                'spectral': [B, C, 6]
                'ratios': [B, 4]
                Note: 'statistical' is deprecated and disabled by default
        """
        result = {}

        # Apply channel mask if provided
        if channel_mask is not None:
            x = x * channel_mask.unsqueeze(-1)

        if self.include_statistical:
            result["statistical"] = self._compute_statistical(x)

        if self.include_hjorth:
            result["hjorth"] = self._compute_hjorth(x)

        if self.include_spectral:
            result["spectral"] = self._compute_spectral(x)

        if self.include_ratios and self.include_spectral:
            result["ratios"] = self._compute_ratios(result["spectral"], eeg_channel_idx)

        return result

    def get_flat_features(
        self,
        x: torch.Tensor,
        channel_mask: torch.Tensor | None = None,
        eeg_channel_indices: list[int] | None = None,
    ) -> torch.Tensor:
        """Extract features and return as flat vector.

        Convenience method that extracts features for EEG channels only
        and returns a flat feature vector suitable for concatenation.

        Args:
            x: [B, C, T] input signal
            channel_mask: [B, C] optional binary mask
            eeg_channel_indices: List of EEG channel indices to use

        Returns:
            [B, F] flat feature vector
        """
        if eeg_channel_indices is None:
            eeg_channel_indices = [0]

        features = self.forward(x, channel_mask, eeg_channel_idx=eeg_channel_indices[0])
        B = x.size(0)

        eeg_idx = torch.tensor(eeg_channel_indices, device=x.device)
        parts = []

        if "statistical" in features:
            stat = features["statistical"][:, eeg_idx, :]  # [B, num_eeg, 5]
            parts.append(stat.reshape(B, -1))

        if "hjorth" in features:
            hjorth = features["hjorth"][:, eeg_idx, :]  # [B, num_eeg, 2]
            parts.append(hjorth.reshape(B, -1))

        if "spectral" in features:
            # Just include slow band as extra (other bands covered elsewhere)
            slow = features["spectral"][:, eeg_idx, 0]  # [B, num_eeg]
            parts.append(slow)

        if "ratios" in features:
            parts.append(features["ratios"])  # [B, 5]

        if parts:
            return torch.cat(parts, dim=-1)
        return torch.zeros(B, 0, device=x.device, dtype=x.dtype)


# =============================================================================
# Utility Functions
# =============================================================================


def get_feature_names(
    n_eeg_channels: int = 2,
    include_statistical: bool = False,  # Deprecated, disabled by default
    include_hjorth: bool = True,
    include_slow_band: bool = True,
    include_extra_ratios: bool = True,
    include_n3_criterion: bool = True,
) -> list[str]:
    """Get human-readable feature names.

    Note: Statistical features are deprecated and disabled by default.

    Returns:
        List of feature names in order.
    """
    names = []

    stat_names = ["std", "iqr", "skew", "kurt", "nzc"]
    hjorth_names = ["mobility", "complexity"]

    if include_statistical:
        # Deprecated: statistical features are no longer used
        for ch in range(n_eeg_channels):
            for stat in stat_names:
                names.append(f"eeg{ch}_{stat}")

    if include_hjorth:
        for ch in range(n_eeg_channels):
            for h in hjorth_names:
                names.append(f"eeg{ch}_{h}")

    if include_slow_band:
        for ch in range(n_eeg_channels):
            names.append(f"eeg{ch}_slow")

    if include_extra_ratios:
        ratio_names = ["delta_theta", "delta_sigma", "delta_beta", "theta_alpha"]
        names.extend(ratio_names)

    if include_n3_criterion:
        for ptp_thresh in (2.0, 2.5, 3.0, 3.5, 4.0):
            names.append(f"n3_occ_ptp_{ptp_thresh:g}")
            names.append(f"n3_soft_ptp_{ptp_thresh:g}")

    return names


def compute_yasa_features_batch(
    x: np.ndarray,
    fs: float = 128.0,
    epoch_sec: float = 30.0,
    eeg_channel_idx: int = 0,
    n_jobs: int = 1,
) -> dict[str, np.ndarray]:
    """Convenience function to compute all YASA features.

    Args:
        x: [n_epochs, C, T] raw signal in microvolts
        fs: Sampling frequency in Hz
        epoch_sec: Epoch duration in seconds
        eeg_channel_idx: Which channel to use for power ratios
        n_jobs: Not used (for API compatibility)

    Returns:
        Dict with feature arrays (including 'n3_criterion' occupancy features).
    """
    computer = YASAFeatureComputer(fs=fs, epoch_sec=epoch_sec)
    return computer.compute_all(x, eeg_channel_idx=eeg_channel_idx)
