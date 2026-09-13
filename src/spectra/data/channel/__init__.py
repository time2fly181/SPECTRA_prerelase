"""Channel processing utilities for PSG data.

This subpackage provides utilities for:
- Channel metadata encoding and token generation
- Channel label normalization and matching
- Linear re-referencing and derivation
- Channel substitution and aliasing
- Montage signature derivation
"""

from __future__ import annotations

from .metadata import (
    POSITION_PAD_IDX,
    POSITION_TO_INDEX,
    POSITION_UNKNOWN_IDX,
    REFERENCE_PAD_IDX,
    REFERENCE_TO_INDEX,
    REFERENCE_UNKNOWN_IDX,
    TYPE_PAD_IDX,
    TYPE_TO_INDEX,
    TYPE_UNKNOWN_IDX,
    encode_channel_metadata_tokens,
)
from .montage import (
    derive_montage_signature,
)
from .normalization import (
    PSGNormalizer,
    infer_channel_type,
    normalize_channel_label,
    normalize_robust,
)
from .rereferencing import (
    DerivationInfo,
    build_derivation_matrix,
    build_montage_transform,
    can_derive_channel,
    find_derivation_path,
    normalize_electrode,
    parse_channel_derivation,
)
from .substitutions import (
    REFERENCE_SUBSTITUTIONS,
    build_substitution_map,
    build_substitution_map_with_rereferencing,
    find_canonical_name,
    find_substitute_channel,
    normalize_for_substitution,
    select_5ch_for_model,
    select_channels_by_modality,
)

__all__ = [
    "TYPE_PAD_IDX",
    "TYPE_UNKNOWN_IDX",
    "TYPE_TO_INDEX",
    "POSITION_PAD_IDX",
    "POSITION_UNKNOWN_IDX",
    "POSITION_TO_INDEX",
    "REFERENCE_PAD_IDX",
    "REFERENCE_UNKNOWN_IDX",
    "REFERENCE_TO_INDEX",
    "encode_channel_metadata_tokens",
    "normalize_channel_label",
    "normalize_robust",
    "infer_channel_type",
    "PSGNormalizer",
    "derive_montage_signature",
    "parse_channel_derivation",
    "can_derive_channel",
    "find_derivation_path",
    "build_derivation_matrix",
    "build_montage_transform",
    "DerivationInfo",
    "normalize_electrode",
    "REFERENCE_SUBSTITUTIONS",
    "normalize_for_substitution",
    "find_canonical_name",
    "find_substitute_channel",
    "build_substitution_map",
    "build_substitution_map_with_rereferencing",
    "select_channels_by_modality",
    "select_5ch_for_model",
]
