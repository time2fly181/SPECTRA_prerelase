"""Model wrappers with embedded preprocessing.

This module provides wrappers for existing PSG models that include the
EmbeddedPreproc normalization module as the first layer. This ensures
identical preprocessing at training and inference time.
"""

from __future__ import annotations

import warnings
from typing import Any, cast

import torch
import torch.nn as nn

from spectra.preprocessing import EmbeddedPreproc, ProcCfg

__all__ = ["ModelWithPreproc", "wrap_model_with_preprocessing"]


class ModelWithPreproc(nn.Module):
    """Wrapper that adds EmbeddedPreproc as first layer before a model.

    This wrapper handles:
    - Normalization with EmbeddedPreproc module
    - Proper reshaping for context windows
    - Calibration interface
    - Optional pass-through mode for precomputed normalized data

    Args:
        model: The base model (e.g., ContextNet, TransformerContextNet)
        preprocessor: EmbeddedPreproc instance (pre-calibrated)
        bypass_preprocessing: If True, skip preprocessing (for precomputed normalized data)

    Example:
        >>> from spectra.models import ContextNet
        >>> from spectra.preprocessing import ProcCfg, EmbeddedPreproc
        >>>
        >>> # Create base model
        >>> base_model = ContextNet(n_channels=5, ...)
        >>>
        >>> # Create preprocessor
        >>> cfg = ProcCfg()
        >>> channel_types = ["c3_a2", "c4_a1", "eog_l", "eog_r", "emg"]
        >>> preprocessor = EmbeddedPreproc(cfg, channel_types)
        >>>
        >>> # Calibrate on sample data
        >>> x_sample = torch.randn(8, 5, 3000)  # (B, C, T)
        >>> preprocessor.calibrate(x_sample)
        >>>
        >>> # Wrap model
        >>> model = ModelWithPreproc(base_model, preprocessor)
        >>>
        >>> # Forward pass with raw data
        >>> x_raw = torch.randn(4, 21, 5, 3000)  # (B, L, C, T) in μV
        >>> logits = model(x_raw)
        >>>
        >>> # Or use with precomputed normalized data (bypass preprocessing)
        >>> model_bypass = ModelWithPreproc(base_model, preprocessor, bypass_preprocessing=True)
        >>> x_norm = torch.randn(4, 21, 5, 3000)  # Already normalized
        >>> logits = model_bypass(x_norm)
    """

    def __init__(
        self,
        model: nn.Module,
        preprocessor: EmbeddedPreproc,
        bypass_preprocessing: bool = False,
        use_raw_for_features: bool = False,
    ):
        super().__init__()
        self.model = model
        self.preprocessor = preprocessor
        self.bypass_preprocessing = bypass_preprocessing
        self.use_raw_for_features = use_raw_for_features
        self._warned_fallback_normstats = (
            False  # Track if we've warned about missing norm_stats
        )

    def _preprocess_inputs(
        self, x_raw, presence_mask: torch.Tensor | None = None
    ) -> dict:
        """Preprocess raw inputs and prepare model inputs.

        Args:
            x_raw: Raw signal tensor or dict with 'wave' key
                - If bypass_preprocessing=False: expects raw μV data
                - If bypass_preprocessing=True: expects already-normalized data
                Shape: (B, L, C, T) for context models
                   or (B, C, T) for single-epoch models
            presence_mask: Optional binary mask indicating present channels (C,)

        Returns:
            Dictionary with preprocessed inputs ready for model
        """
        # Handle dict inputs (extract wave tensor and optional presence_mask)
        if isinstance(x_raw, dict):
            wave_tensor = x_raw["wave"]
            # If presence_mask kwarg not provided, try extracting from dict
            if presence_mask is None and "presence_mask" in x_raw:
                presence_mask = x_raw["presence_mask"]
        else:
            wave_tensor = x_raw

        # Detect input shape
        if wave_tensor.ndim == 4:
            # Context window input: (B, L, C, T)
            B, L, C, T = wave_tensor.shape
            has_context = True
        elif wave_tensor.ndim == 3:
            # Single epoch input: (B, C, T)
            B, C, T = wave_tensor.shape
            L = 1
            has_context = False
            wave_tensor = wave_tensor.unsqueeze(1)  # (B, 1, C, T)
        else:
            raise ValueError(f"Expected 3D or 4D input, got shape {wave_tensor.shape}")

        # Apply normalization (or pass through if bypassed)
        if self.bypass_preprocessing:
            # Data is already normalized (e.g., from precomputed zarr store)
            # Just reshape to maintain consistency
            x_norm = wave_tensor
        else:
            # Reshape for preprocessing: (B*L, C, T)
            x_flat = wave_tensor.reshape(B * L, C, T)
            preprocessor_mask = presence_mask
            if presence_mask is not None and presence_mask.ndim == 3:
                if presence_mask.shape != (B, L, C):
                    raise ValueError(
                        "Epoch-specific presence_mask must have shape "
                        f"{(B, L, C)}, got {tuple(presence_mask.shape)}"
                    )
                preprocessor_mask = presence_mask.reshape(B * L, C)

            # Apply normalization
            x_norm = self.preprocessor(x_flat, channel_mask=preprocessor_mask)

            # Reshape back to context shape: (B, L, C, T)
            x_norm = x_norm.reshape(B, L, C, T)

        # Remove context dimension if it wasn't in original input
        if not has_context:
            x_norm = x_norm.squeeze(1)  # (B, C, T)

        # CRITICAL: WAVEFORM SELECTION FOR ENGINEERED FEATURES
        #
        # New behavior with use_raw_for_features flag:
        #
        # When use_raw_for_features=True (NEW):
        #   - Feature extraction uses denormalized raw μV waveforms
        #   - Preserves absolute amplitude information for delta power (N3 detection)
        #   - If bypass_preprocessing=False: Use original wave_tensor (should be raw μV)
        #   - If bypass_preprocessing=True: Denormalize using stored IQR stats
        #
        # When use_raw_for_features=False (DEFAULT - backward compatible):
        #   - Feature extraction uses IQR-normalized waveforms (same as CNN)
        #   - Maintains existing behavior for models trained with normalized waveforms
        #   - Works across different recordings/equipment due to normalization
        #
        # TWO-LEVEL NORMALIZATION (when use_raw_for_features=False):
        # Level 1: Waveform Normalization (IQR-based) - BOTH CNN and feature extraction
        # Level 2: Feature Value Normalization (Standardization) - only for extracted features
        #   (handled internally by SleepFeatureExtractor._normalize_features())

        wave_raw_for_features = None

        if self.use_raw_for_features:
            # NEW: Provide raw μV waveforms for feature extraction
            if self.bypass_preprocessing:
                # Data is already normalized. Try to denormalize it.
                if isinstance(x_raw, dict) and "norm_stats" in x_raw:
                    # Reconstruct raw data: raw = norm * iqr + median
                    stats = x_raw["norm_stats"]
                    median = stats["median"]  # [B, C, 1]
                    iqr = stats["iqr"]  # [B, C, 1]

                    # Handle shapes for broadcasting
                    if wave_tensor.ndim == 4:
                        # Context: [B, L, C, T] -> Stats: [B, C, 1] -> [B, 1, C, 1]
                        median = median.unsqueeze(1)
                        iqr = iqr.unsqueeze(1)

                    # Denormalize: raw = norm * iqr + median
                    wave_raw_for_features = wave_tensor * iqr + median
                else:
                    # Try to denormalize using preprocessor's stored calibration stats
                    if hasattr(self.preprocessor, "median_") and hasattr(
                        self.preprocessor, "iqr_"
                    ):
                        median = cast(torch.Tensor, self.preprocessor.median_)  # [C, 1]
                        iqr = cast(torch.Tensor, self.preprocessor.iqr_)  # [C, 1]

                        # Reshape for broadcasting: [C, 1] -> [1, C, 1]
                        median = median.unsqueeze(0)
                        iqr = iqr.unsqueeze(0)

                        # Handle context dimension
                        if wave_tensor.ndim == 4:
                            # [1, C, 1] -> [1, 1, C, 1]
                            median = median.unsqueeze(0)
                            iqr = iqr.unsqueeze(0)

                        # Denormalize
                        wave_raw_for_features = wave_tensor * iqr + median
                    else:
                        # Fallback: Use normalized data (warn once)
                        wave_raw_for_features = x_norm
                        if not self._warned_fallback_normstats:
                            warnings.warn(
                                "ModelWithPreproc: use_raw_for_features=True but cannot denormalize. "
                                "Missing 'norm_stats' in input and preprocessor not calibrated. "
                                "Falling back to normalized data for feature extraction. "
                                "This degrades N3 detection accuracy (delta power features need absolute amplitude).",
                                category=UserWarning,
                                stacklevel=2,
                            )
                            self._warned_fallback_normstats = True
            else:
                # bypass_preprocessing=False: Input should be raw μV data
                wave_raw_for_features = wave_tensor
        else:
            # DEFAULT: Use IQR-normalized waveforms for feature extraction (backward compatible)
            # This is the old behavior where features receive normalized data
            wave_raw_for_features = x_norm

        if presence_mask is not None:
            model_input = {
                "wave": x_norm,  # IQR-normalized data for CNN
                "wave_raw": wave_raw_for_features,  # Raw (uV) data for engineered features
                "presence_mask": presence_mask,
            }
        else:
            # Pass IQR-normalized data for both CNN and engineered features
            model_input = {
                "wave": x_norm,
                "wave_raw": wave_raw_for_features,
            }

        # Forward passthrough keys from the input dict that we don't consume
        # here but the wrapped model needs (e.g., the epoch-validity mask used
        # for loop-boundary protection).
        _PASSTHROUGH_KEYS = {
            "epoch_valid_mask",
        }
        if isinstance(x_raw, dict):
            for key in _PASSTHROUGH_KEYS:
                if key in x_raw and key not in model_input:
                    model_input[key] = x_raw[key]

        return model_input

    def forward(
        self,
        x_raw: torch.Tensor | dict,
        presence_mask: torch.Tensor | None = None,
        predict_all: bool = False,
        recurrent_refinement_steps: int | None = None,
    ) -> torch.Tensor:
        """Forward pass with embedded preprocessing.

        Args:
            x_raw: Raw signal tensor or dict with 'wave' key in μV
                Shape: (B, L, C, T) for context models
                   or (B, C, T) for single-epoch models
            presence_mask: Optional binary mask indicating present channels (C,)
                1 = channel present, 0 = channel missing
            predict_all: If True, return predictions for all epochs in context window.
                If False (default), return only center epoch prediction.

        Returns:
            Model output (typically logits)
            - If predict_all=False: [B, num_classes]
            - If predict_all=True: [B, L, num_classes]
        """
        model_input = self._preprocess_inputs(x_raw, presence_mask)

        forward_kwargs: dict[str, Any] = {}
        if predict_all:
            forward_kwargs["predict_all"] = True
        if recurrent_refinement_steps is not None:
            forward_kwargs["recurrent_refinement_steps"] = recurrent_refinement_steps
        try:
            output = self.model(model_input, **forward_kwargs)
        except TypeError:
            if forward_kwargs:
                output = self.model(model_input)
            else:
                raise
        return output

    def forward_features(
        self,
        x_raw: torch.Tensor | dict,
        presence_mask: torch.Tensor | None = None,
        all_positions: bool = False,
    ) -> torch.Tensor:
        """Forward pass that returns features before classifier.

        Used for contrastive learning where we need backbone features.

        Args:
            x_raw: Raw signal tensor or dict with 'wave' key in μV
                Shape: (B, L, C, T) for context models
                   or (B, C, T) for single-epoch models
            presence_mask: Optional binary mask indicating present channels (C,)
            all_positions: If True, return features for all positions

        Returns:
            Feature tensor (shape depends on model and all_positions flag)
        """
        model_input = self._preprocess_inputs(x_raw, presence_mask)

        # Call forward_features on underlying model
        if hasattr(self.model, "forward_features"):
            return cast(Any, self.model).forward_features(
                model_input, all_positions=all_positions
            )
        else:
            raise AttributeError(
                f"Underlying model {type(self.model).__name__} has no 'forward_features' method"
            )

    def calibrate(
        self,
        x_raw: torch.Tensor | dict,
        presence_mask: torch.Tensor | None = None,
    ) -> None:
        """Calibrate the preprocessor on sample data.

        Args:
            x_raw: Raw signal tensor or dict with 'wave' key
                - If bypass_preprocessing=False: expects raw μV data
                - If bypass_preprocessing=True: no-op (already normalized)
                Shape: (B, C, T) or (B, L, C, T)
            presence_mask: Optional binary mask indicating present channels (C,)
                1 = channel present, 0 = channel missing
        """
        # Skip calibration if preprocessing is bypassed
        if self.bypass_preprocessing:
            return

        # Handle dict inputs
        if isinstance(x_raw, dict):
            wave_tensor = x_raw["wave"]
        else:
            wave_tensor = x_raw

        # Extract single-epoch data for calibration
        if wave_tensor.ndim == 4:
            # Take center epoch: (B, L, C, T) -> (B, C, T)
            center_idx = wave_tensor.size(1) // 2
            x_sample = wave_tensor[:, center_idx, :, :]
        else:
            x_sample = wave_tensor

        self.preprocessor.calibrate(x_sample, channel_mask=presence_mask)

    def get_preprocessing_config(self) -> ProcCfg:
        """Get the preprocessing configuration."""
        return self.preprocessor.cfg

    @property
    def feature_dim(self) -> int:
        """Forward feature_dim from underlying model."""
        if hasattr(self.model, "feature_dim"):
            return cast(int, self.model.feature_dim)
        raise AttributeError(
            f"Underlying model {type(self.model).__name__} has no 'feature_dim' attribute"
        )

    @property
    def classifier(self):
        """Forward classifier from underlying model."""
        if hasattr(self.model, "classifier"):
            return self.model.classifier
        raise AttributeError(
            f"Underlying model {type(self.model).__name__} has no 'classifier' attribute"
        )


def wrap_model_with_preprocessing(
    model: nn.Module,
    proc_cfg: ProcCfg,
    channel_types: list[str],
    calibration_data: torch.Tensor | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    bypass_preprocessing: bool = False,
    use_raw_for_features: bool = False,
) -> ModelWithPreproc:
    """Convenience function to wrap a model with preprocessing.

    Args:
        model: Base model to wrap
        proc_cfg: Preprocessing configuration
        channel_types: List of channel type strings (e.g., ["c3_a2", "eog_l", "emg"])
        calibration_data: Optional calibration data (B, C, T) in μV
            If provided, preprocessor will be calibrated immediately
        device: Device for preprocessor buffers
        dtype: Data type for preprocessor buffers
        bypass_preprocessing: If True, skip preprocessing (for precomputed normalized data)
        use_raw_for_features: If True, provide denormalized raw μV waveforms to engineered
            features instead of IQR-normalized waveforms. Improves delta power features for N3.

    Returns:
        ModelWithPreproc instance

    Example:
        >>> from spectra.models import ContextNet
        >>> from spectra.preprocessing import ProcCfg
        >>>
        >>> base_model = ContextNet(n_channels=5, ...)
        >>> cfg = ProcCfg()
        >>> channel_types = ["c3_a2", "c4_a1", "eog_l", "eog_r", "emg"]
        >>>
        >>> # Wrap and calibrate for raw data
        >>> calibration_batch = torch.randn(8, 5, 3000)
        >>> model = wrap_model_with_preprocessing(
        ...     base_model, cfg, channel_types,
        ...     calibration_data=calibration_batch
        ... )
        >>>
        >>> # Or wrap for precomputed normalized data
        >>> model_norm = wrap_model_with_preprocessing(
        ...     base_model, cfg, channel_types,
        ...     bypass_preprocessing=True
        ... )
    """
    # Create preprocessor
    preprocessor = EmbeddedPreproc(proc_cfg, channel_types, device=device, dtype=dtype)

    # Calibrate if data provided and not bypassing
    if calibration_data is not None and not bypass_preprocessing:
        preprocessor.calibrate(calibration_data)

    # Wrap model
    wrapped = ModelWithPreproc(
        model,
        preprocessor,
        bypass_preprocessing=bypass_preprocessing,
        use_raw_for_features=use_raw_for_features,
    )

    return wrapped
