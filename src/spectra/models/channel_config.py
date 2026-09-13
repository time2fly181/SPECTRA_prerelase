"""Channel configuration for PSG models.

Provides channel-type inference, contralateral pair detection, and
configuration loading from channels_canon.json files.

Key Features:
    - Auto-detect channel type (EEG/EOG/EMG/ECG) from naming patterns
    - Parse bipolar derivations to extract electrode/reference
    - Infer hemisphere from electrode naming (odd=left, even=right)
    - Find contralateral pairs for inter-channel coherence modeling
    - Load configuration from JSON files

Usage:
    # From JSON file
    config = CanonicalChannelSet.from_json('channels_canon.json')

    # From list of channel names
    config = CanonicalChannelSet.from_names([
        'C3-M2', 'C4-M1', 'F3-M2', 'F4-M1',
        'LOC-A2', 'ROC-A1',
        'EMG1-EMG2',
    ])

    # Access channel info
    print(f"EEG channels: {config.n_eeg}")
    print(f"Contralateral pairs: {config.contralateral_pairs}")
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import torch


class ChannelType(Enum):
    """PSG channel types with distinct physiological characteristics."""

    EEG = "eeg"
    EOG = "eog"
    EMG = "emg"
    ECG = "ecg"
    UNKNOWN = "unknown"


@dataclass
class ChannelConfig:
    """Configuration for a single canonical channel.

    Attributes:
        name: Full channel name (e.g., "C3-M2")
        channel_type: Inferred channel type
        index: Position in canonical channel list
        frequency_range: Typical frequency band (Hz)
        typical_amplitude_iqr: Normalized amplitude range
        electrode: Primary electrode (e.g., "C3")
        reference: Reference electrode (e.g., "M2")
        hemisphere: Laterality ("left", "right", "midline", or None)
        contralateral_idx: Index of contralateral pair channel
    """

    name: str
    channel_type: ChannelType
    index: int

    # Physiological characteristics
    frequency_range: tuple[float, float] = (0.5, 35.0)
    typical_amplitude_iqr: float = 1.0

    # Reference information
    electrode: str | None = None
    reference: str | None = None
    hemisphere: str | None = None

    # Related channels
    contralateral_idx: int | None = None

    @classmethod
    def from_name(cls, name: str, index: int) -> ChannelConfig:
        """Parse channel name to create configuration.

        Args:
            name: Channel name (e.g., "C3-M2", "LOC-A2", "EMG1-EMG2")
            index: Position in channel list

        Returns:
            Configured ChannelConfig instance
        """
        channel_type = cls._infer_type(name)
        electrode, reference = cls._parse_derivation(name)
        hemisphere = cls._infer_hemisphere(electrode)
        freq_range = cls._get_frequency_range(channel_type)

        return cls(
            name=name,
            channel_type=channel_type,
            index=index,
            frequency_range=freq_range,
            electrode=electrode,
            reference=reference,
            hemisphere=hemisphere,
        )

    @staticmethod
    def _infer_type(name: str) -> ChannelType:
        """Infer channel type from name patterns.

        Uses common PSG naming conventions:
        - EOG: Contains 'EOG', 'LOC', 'ROC', 'E1', 'E2'
        - EMG: Contains 'EMG', 'CHIN', 'MENT', 'SUBMENT'
        - ECG: Contains 'ECG', 'EKG'
        - EEG: Standard 10-20 electrode names
        """
        name_upper = name.upper()

        # EOG patterns
        if any(p in name_upper for p in ["EOG", "LOC", "ROC", "E1", "E2"]):
            return ChannelType.EOG

        # EMG patterns
        if any(p in name_upper for p in ["EMG", "CHIN", "MENT", "SUBMENT"]):
            return ChannelType.EMG

        # ECG patterns
        if any(p in name_upper for p in ["ECG", "EKG"]):
            return ChannelType.ECG

        # Generic EEG patterns (EEG, EEG1, EEG2, etc.)
        # Must check BEFORE specific 10-20 electrodes
        if re.match(r"^EEG\d*$", name_upper):
            return ChannelType.EEG

        # EEG patterns (10-20 system electrodes)
        eeg_electrodes = [
            "FP1",
            "FP2",
            "F3",
            "F4",
            "F7",
            "F8",
            "FZ",
            "C3",
            "C4",
            "CZ",
            "T3",
            "T4",
            "T5",
            "T6",
            "T7",
            "T8",
            "P3",
            "P4",
            "PZ",
            "O1",
            "O2",
            "OZ",
            "A1",
            "A2",
            "M1",
            "M2",
        ]
        for elec in eeg_electrodes:
            if elec in name_upper:
                return ChannelType.EEG

        return ChannelType.UNKNOWN

    @staticmethod
    def _parse_derivation(name: str) -> tuple[str | None, str | None]:
        """Parse bipolar derivation into electrode and reference.

        Handles common separators: '-', '_', ' '

        Args:
            name: Channel derivation string

        Returns:
            (electrode, reference) tuple
        """
        for sep in ["-", "_", " "]:
            if sep in name:
                parts = name.split(sep)
                if len(parts) >= 2:
                    return parts[0].strip(), parts[1].strip()
        return name, None

    @staticmethod
    def _infer_hemisphere(electrode: str | None) -> str | None:
        """Infer hemisphere from electrode name.

        Uses 10-20 conventions:
        - Odd numbers = left hemisphere
        - Even numbers = right hemisphere
        - 'Z' suffix = midline
        - LOC/L* = left, ROC/R* = right
        """
        if electrode is None:
            return None

        elec = electrode.upper()

        # Midline electrodes
        if elec.endswith("Z"):
            return "midline"

        # Check for trailing digit
        match = re.search(r"(\d)$", elec)
        if match:
            digit = int(match.group(1))
            return "left" if digit % 2 == 1 else "right"

        # EOG laterality
        if "LOC" in elec or elec.startswith("L"):
            return "left"
        if "ROC" in elec or elec.startswith("R"):
            return "right"

        return None

    @staticmethod
    def _get_frequency_range(channel_type: ChannelType) -> tuple[float, float]:
        """Get typical frequency range for channel type.

        EEG: 0.5-35 Hz (delta to gamma)
        EOG: 0.1-15 Hz (slow eye movements)
        EMG: 10-100 Hz (muscle activity)
        ECG: 0.5-40 Hz (cardiac cycle)
        """
        ranges = {
            ChannelType.EEG: (0.5, 35.0),
            ChannelType.EOG: (0.1, 15.0),
            ChannelType.EMG: (10.0, 100.0),
            ChannelType.ECG: (0.5, 40.0),
            ChannelType.UNKNOWN: (0.5, 35.0),
        }
        return ranges.get(channel_type, (0.5, 35.0))


@dataclass
class CanonicalChannelSet:
    """Complete canonical channel configuration.

    Provides organized access to channel configurations with
    type-based indexing and contralateral pair tracking.

    Attributes:
        channels: List of all channel configurations
        eeg_indices: Indices of EEG channels
        eog_indices: Indices of EOG channels
        emg_indices: Indices of EMG channels
        ecg_indices: Indices of ECG channels
        contralateral_pairs: List of (idx1, idx2) contralateral pairs
    """

    channels: list[ChannelConfig] = field(default_factory=list)

    # Channel type indices
    eeg_indices: list[int] = field(default_factory=list)
    eog_indices: list[int] = field(default_factory=list)
    emg_indices: list[int] = field(default_factory=list)
    ecg_indices: list[int] = field(default_factory=list)

    # Contralateral pairs for inter-channel modeling
    contralateral_pairs: list[tuple[int, int]] = field(default_factory=list)

    @classmethod
    def from_json(cls, json_path: str | Path) -> CanonicalChannelSet:
        """Load configuration from channels_canon.json file.

        Supports two JSON formats:
        1. List of channel names: ["C3-M2", "C4-M1", ...]
        2. Dict with channels key: {"channels": ["C3-M2", ...]}

        Args:
            json_path: Path to JSON file

        Returns:
            Configured CanonicalChannelSet

        Raises:
            ValueError: If JSON format is not recognized
        """
        with open(json_path) as f:
            data = json.load(f)

        if isinstance(data, list):
            channel_names = data
        elif isinstance(data, dict) and "channels" in data:
            channel_names = data["channels"]
        else:
            raise ValueError(
                "JSON must be list of channel names or dict with 'channels' key"
            )

        return cls.from_names(channel_names)

    @classmethod
    def from_names(cls, channel_names: list[str]) -> CanonicalChannelSet:
        """Create configuration from list of channel names.

        Args:
            channel_names: List of channel names

        Returns:
            Configured CanonicalChannelSet with inferred types and pairs
        """
        channels = [
            ChannelConfig.from_name(name, i) for i, name in enumerate(channel_names)
        ]

        # Build type indices
        eeg_indices = [c.index for c in channels if c.channel_type == ChannelType.EEG]
        eog_indices = [c.index for c in channels if c.channel_type == ChannelType.EOG]
        emg_indices = [c.index for c in channels if c.channel_type == ChannelType.EMG]
        ecg_indices = [c.index for c in channels if c.channel_type == ChannelType.ECG]

        # Find contralateral pairs
        contralateral_pairs = cls._find_contralateral_pairs(channels)

        # Update contralateral indices in channel configs
        for i, j in contralateral_pairs:
            channels[i].contralateral_idx = j
            channels[j].contralateral_idx = i

        return cls(
            channels=channels,
            eeg_indices=eeg_indices,
            eog_indices=eog_indices,
            emg_indices=emg_indices,
            ecg_indices=ecg_indices,
            contralateral_pairs=contralateral_pairs,
        )

    @staticmethod
    def _find_contralateral_pairs(
        channels: list[ChannelConfig],
    ) -> list[tuple[int, int]]:
        """Find contralateral channel pairs.

        Groups EEG channels by base electrode name (e.g., 'C' for C3/C4)
        and pairs channels on opposite hemispheres.

        Args:
            channels: List of channel configurations

        Returns:
            List of (left_idx, right_idx) tuples
        """
        pairs = []

        # Group EEG channels by base electrode
        electrode_groups: dict[str, list[int]] = {}
        for ch in channels:
            if ch.channel_type == ChannelType.EEG and ch.electrode:
                # Extract base (e.g., 'C3' -> 'C', 'F4' -> 'F')
                base = re.sub(r"\d+$", "", ch.electrode.upper())
                if base not in electrode_groups:
                    electrode_groups[base] = []
                electrode_groups[base].append(ch.index)

        # Find pairs within each group
        for indices in electrode_groups.values():
            if len(indices) == 2:
                ch1, ch2 = channels[indices[0]], channels[indices[1]]
                if (
                    ch1.hemisphere
                    and ch2.hemisphere
                    and ch1.hemisphere != ch2.hemisphere
                ):
                    pairs.append((indices[0], indices[1]))

        return pairs

    @property
    def n_channels(self) -> int:
        """Total number of channels."""
        return len(self.channels)

    @property
    def n_eeg(self) -> int:
        """Number of EEG channels."""
        return len(self.eeg_indices)

    @property
    def n_eog(self) -> int:
        """Number of EOG channels."""
        return len(self.eog_indices)

    @property
    def n_emg(self) -> int:
        """Number of EMG channels."""
        return len(self.emg_indices)

    @property
    def n_ecg(self) -> int:
        """Number of ECG channels."""
        return len(self.ecg_indices)

    def get_type_mask(self, channel_type: ChannelType) -> torch.Tensor:
        """Get boolean mask for channels of given type.

        Args:
            channel_type: Type to select

        Returns:
            Boolean tensor of shape [n_channels]
        """
        mask = torch.zeros(self.n_channels, dtype=torch.bool)
        indices = {
            ChannelType.EEG: self.eeg_indices,
            ChannelType.EOG: self.eog_indices,
            ChannelType.EMG: self.emg_indices,
            ChannelType.ECG: self.ecg_indices,
        }.get(channel_type, [])
        for idx in indices:
            mask[idx] = True
        return mask

    def get_channel_names(self) -> list[str]:
        """Get list of channel names in order."""
        return [ch.name for ch in self.channels]

    def get_type_names(self, channel_type: ChannelType) -> list[str]:
        """Get names of channels of a specific type."""
        indices = {
            ChannelType.EEG: self.eeg_indices,
            ChannelType.EOG: self.eog_indices,
            ChannelType.EMG: self.emg_indices,
            ChannelType.ECG: self.ecg_indices,
        }.get(channel_type, [])
        return [self.channels[i].name for i in indices]

    def __repr__(self) -> str:
        return (
            f"CanonicalChannelSet("
            f"n_channels={self.n_channels}, "
            f"eeg={self.n_eeg}, eog={self.n_eog}, "
            f"emg={self.n_emg}, ecg={self.n_ecg}, "
            f"pairs={len(self.contralateral_pairs)})"
        )


# Default 3-channel configuration (for backwards compatibility)
DEFAULT_CHANNEL_NAMES = ["EEG", "EOG", "EMG"]

# Default 5-channel configuration: 2 EEG, 2 EOG, 1 EMG
# Channel order: [EEG1, EEG2, EOG1, EOG2, EMG]
# Indices: eeg=[0,1], eog=[2,3], emg=[4]
DEFAULT_5CH_NAMES = ["EEG1", "EEG2", "EOG1", "EOG2", "EMG"]


def get_default_channel_config() -> CanonicalChannelSet:
    """Get default 3-channel configuration (EEG, EOG, EMG)."""
    return CanonicalChannelSet.from_names(DEFAULT_CHANNEL_NAMES)


def get_default_5ch_config() -> CanonicalChannelSet:
    """Get default 5-channel configuration (2 EEG, 2 EOG, 1 EMG).

    Returns a CanonicalChannelSet with:
        - eeg_indices: [0, 1]
        - eog_indices: [2, 3]
        - emg_indices: [4]

    This is the standard layout expected by models when no
    explicit channel configuration is provided.
    """
    return CanonicalChannelSet.from_names(DEFAULT_5CH_NAMES)


def create_5ch_config_from_actual_names(
    actual_channel_names: list[str | None],
) -> CanonicalChannelSet:
    """Create 5-channel config from actual channel names with fixed modality indices.

    Creates a CanonicalChannelSet with the standard 5-channel layout:
        - Indices 0-1: EEG channels
        - Indices 2-3: EOG channels
        - Index 4: EMG channel

    Uses actual channel names where available for proper type inference and
    contralateral pair detection. Falls back to generic names for missing channels.

    Args:
        actual_channel_names: List of 5 actual channel names (None for missing).
            Expected order: [EEG1, EEG2, EOG1, EOG2, EMG]

    Returns:
        CanonicalChannelSet with fixed modality indices regardless of type inference.

    Raises:
        ValueError: If list doesn't contain exactly 5 elements.
    """
    if len(actual_channel_names) != 5:
        raise ValueError(f"Expected 5 channel names, got {len(actual_channel_names)}")

    # Build channel names, using canonical placeholder for missing
    channel_names = []
    canonical_fallbacks = ["EEG1", "EEG2", "EOG1", "EOG2", "EMG"]

    for i, name in enumerate(actual_channel_names):
        if name is not None:
            channel_names.append(name)
        else:
            channel_names.append(canonical_fallbacks[i])

    # Create configuration from names (this will infer types)
    config = CanonicalChannelSet.from_names(channel_names)

    # Override modality indices to ensure correct 5-channel layout
    # Even if type inference differs, we KNOW the intended layout
    config.eeg_indices = [0, 1]
    config.eog_indices = [2, 3]
    config.emg_indices = [4]

    return config


if __name__ == "__main__":
    # Example usage
    EXAMPLE_CHANNELS = [
        "C3-M2",
        "C4-M1",  # Central EEG (contralateral pair)
        "F3-M2",
        "F4-M1",  # Frontal EEG (contralateral pair)
        "O1-M2",
        "O2-M1",  # Occipital EEG (contralateral pair)
        "LOC-A2",
        "ROC-A1",  # EOG
        "EMG1-EMG2",  # Chin EMG
    ]

    print("Creating channel configuration...")
    config = CanonicalChannelSet.from_names(EXAMPLE_CHANNELS)

    print(f"\n{config}")
    print("\nChannel details:")
    for ch in config.channels:
        print(
            f"  [{ch.index}] {ch.name}: "
            f"type={ch.channel_type.value}, "
            f"hemisphere={ch.hemisphere}, "
            f"freq={ch.frequency_range}"
        )

    print(f"\nEEG indices: {config.eeg_indices}")
    print(f"EOG indices: {config.eog_indices}")
    print(f"EMG indices: {config.emg_indices}")
    print(f"Contralateral pairs: {config.contralateral_pairs}")

    # Test mask
    eeg_mask = config.get_type_mask(ChannelType.EEG)
    print(f"\nEEG mask: {eeg_mask.tolist()}")

    print("\n✓ Channel configuration ready!")
