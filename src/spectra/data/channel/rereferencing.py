"""Linear re-referencing for EEG channels.

This module implements algebraic channel derivation through "telescoping"
bipolar montages. If target derivations aren't directly available, they can
be computed from available pairs using linear combinations. These helpers are
available to channel-utility callers; the EDF scoring pipeline disables them.

Mathematical background:
Any bipolar derivation is a linear combination of electrode potentials:
    (C4-A1) = potential(C4) - potential(A1)

If you have the right pairs, you can telescope them:
    C4-A1 = (C4-P4) + (P4-A1)    [X-Y + Y-Z = X-Z]
    C4-A1 = (C4-A2) - (A1-A2)    [X-B + B-Y = X-Y when A1-A2 is backwards]

Matrix representation:
Let v(t) be a vector of electrode potentials at time t.
Any derivation can be written as: out = M @ v
where M has +1 for positive pole, -1 for reference pole, 0 elsewhere.

Example:
    Electrodes: [C3, C4, P3, P4, A1, A2]
    Target: C4-A1
    Matrix M has row: [0, +1, 0, 0, -1, 0]

This module builds transformation matrices at load time to derive
canonical montages from whatever montage was recorded.

Example:
    >>> from spectra.data.channel import build_derivation_matrix, can_derive_channel
    >>> available = ['C4-P4', 'P4-A1', 'C3-A2']
    >>> target = 'C4-A1'
    >>> matrix, info = build_derivation_matrix(target, available)
    >>> # Apply: derived_signal = matrix @ available_signals
"""

from __future__ import annotations

import re
from collections import defaultdict

import numpy as np

__all__ = [
    "parse_channel_derivation",
    "parse_channel_derivation_ex",
    "can_derive_channel",
    "find_derivation_path",
    "build_derivation_matrix",
    "build_montage_transform",
    "DerivationInfo",
    "COMMON_REFERENCE",
]

# Graph node standing in for the amplifier's common reference. Every bare
# (monopolar) electrode in a recording is measured against it, whatever it
# physically was -- Fpz, a linked pair, or an unnamed system reference.
COMMON_REFERENCE = "__COMMON__"

# References that `parse_channel_derivation_ex` guesses for a bare electrode
# rather than reading off the channel name.
_GUESSED_REFERENCES = frozenset({"A1", "A2", "EMGREF"})


def parse_channel_derivation(channel: str) -> tuple[str, str] | None:
    """Parse a bipolar channel name into (positive, reference) electrodes.

    Thin wrapper around :func:`parse_channel_derivation_ex` that drops the
    ``inferred`` flag. Prefer the ``_ex`` variant when the caller needs to know
    whether the reference electrode was named by the channel or merely guessed.

    Args:
        channel: Channel name like 'C4-A1', 'F3-M2', 'eeg/F4_M1', 'eog/EOG1', etc.

    Returns:
        Tuple of (positive_electrode, reference_electrode) or None if not bipolar

    Examples:
        >>> parse_channel_derivation('C4-A1')
        ('C4', 'A1')
        >>> parse_channel_derivation('EEG F3-M2')
        ('F3', 'M2')
        >>> parse_channel_derivation('eeg/F4_M1')
        ('F4', 'M1')
        >>> parse_channel_derivation('F3')
        ('F3', 'A2')  # Odd electrode -> A2 reference (guessed)
        >>> parse_channel_derivation('C4')
        ('C4', 'A1')  # Even electrode -> A1 reference (guessed)
        >>> parse_channel_derivation('eog/EOG1')
        ('EOG1', 'A2')
        >>> parse_channel_derivation('EOG')
        None
    """
    parsed = parse_channel_derivation_ex(channel)
    if parsed is None:
        return None
    pos, ref, _inferred = parsed
    return (pos, ref)


def parse_channel_derivation_ex(channel: str) -> tuple[str, str, bool] | None:
    """Parse a channel name into (positive, reference, inferred).

    ``inferred`` is True when the channel name does not itself name a reference
    electrode and one had to be guessed (a bare ``C3`` assumed to be ``C3-A2``).
    That distinction is load-bearing: a monopolar recording that also ships the
    mastoid as its own channel must be derived by real subtraction, not accepted
    as an already-referenced derivation.

    Args:
        channel: Channel name like 'C4-A1', 'F3-M2', 'eeg/F4_M1', 'eog/EOG1', etc.

    Returns:
        Tuple of (positive_electrode, reference_electrode, inferred), or None if
        the name cannot be interpreted as a derivation at all.

    Examples:
        >>> parse_channel_derivation_ex('C4-A1')
        ('C4', 'A1', False)
        >>> parse_channel_derivation_ex('C4')
        ('C4', 'A1', True)
    """
    # Normalize: remove common prefixes, standardize separators
    channel = channel.strip()

    # Handle type/ or type_ prefix notation (e.g., "eeg/F4_M1", "eeg_F4_M1", "eog/EOG1", "eog_EOG1", "emg/EMG", "emg_EMG")
    # Check for type prefix with forward slash or underscore
    type_prefix_match = re.match(
        r"^(eeg|eog|emg|ecg|ekg)[/_](.+)$", channel, flags=re.IGNORECASE
    )
    if type_prefix_match:
        # Extract the channel name after the type prefix
        channel = type_prefix_match.group(2)

        # Special case: Handle mislabeled channels like "emg/ECG" (should be ecg/ECG)
        # The type prefix might not match the actual channel type - that's okay, we just strip it
        # The actual type will be inferred from the channel name itself

        # Convert underscores to hyphens for bipolar notation
        # This handles "eeg/F4_M1" -> "F4-M1"
        if "_" in channel and "-" not in channel:
            # Check if this looks like a bipolar channel (has uppercase letters on both sides of _)
            # e.g., "F4_M1" should become "F4-M1", but "EOG_1" should stay as is
            parts = channel.split("_", 1)
            if len(parts) == 2 and parts[0] and parts[1]:
                # Check if both parts look like electrode names (contain letters)
                if (
                    any(c.isalpha() for c in parts[0])
                    and any(c.isalpha() for c in parts[1])
                    and not parts[0].upper().startswith("EOG")
                    and not parts[0].upper().startswith("EMG")
                    and not parts[0].upper().startswith("ECG")
                ):
                    channel = channel.replace("_", "-", 1)

    # For EOG and EMG, we want to preserve them but normalize the format
    # "EOG horizontal" -> "EOG", "EMG submental" -> "EMG submental"

    # Remove EEG/PSG prefixes (but preserve EOG/EMG as they're channel types)
    # Apply multiple passes to handle cases like "PSG_EEG F3" -> "EEG F3" -> "F3"
    prev_channel = ""
    while prev_channel != channel:
        prev_channel = channel
        # Remove prefixes with space separator: "EEG F3", "PSG F3"
        channel = re.sub(r"^(EEG|PSG)\s+", "", channel, flags=re.IGNORECASE)
        # Remove prefixes with underscore separator: "EEG_F3", "PSG_F3"
        channel = re.sub(r"^(EEG|PSG)_", "", channel, flags=re.IGNORECASE)
        # Remove prefixes with no separator: "EEGF3" (less common)
        channel = re.sub(r"^(EEG|PSG)([A-Z]\d)", r"\2", channel, flags=re.IGNORECASE)
    channel = channel.replace("–", "-")  # Em dash to hyphen
    channel = channel.replace("—", "-")  # En dash to hyphen

    # Try to split on hyphen (most common)
    parts = channel.split("-")
    if len(parts) == 2:
        pos = parts[0].strip().upper()
        ref = parts[1].strip().upper()
        if pos and ref:
            return (pos, ref, False)

    # Try forward slash (alternative notation)
    parts = channel.split("/")
    if len(parts) == 2:
        pos = parts[0].strip().upper()
        ref = parts[1].strip().upper()
        if pos and ref:
            return (pos, ref, False)

    # Single electrode name - infer reference based on type and hemisphere
    if len(parts) == 1:
        electrode = parts[0].strip().upper()

        # Normalize E1/E2 to EOG1/EOG2 before processing
        electrode = normalize_electrode(electrode)

        # Special handling for EOG channels
        # EOG1, EOG2, etc. should reference to A2 (not hemisphere-dependent)
        if electrode.startswith("EOG"):
            # Strip descriptive terms: "vertical", "horizontal"
            electrode = re.sub(
                r"\s*(VERTICAL|HORIZONTAL)\s*", "", electrode, flags=re.IGNORECASE
            )
            electrode = electrode.strip()

            # If we're left with just "EOG" (no number, no designation), drop it
            if electrode == "EOG":
                return None

            # Check if it has a number (with space, underscore, or no separator)
            match = re.search(r"EOG[\s_]*(\d+)", electrode)
            if match:
                num = match.group(1)
                # EOG channels typically reference to A2
                return (f"EOG{num}", "A2", True)
            # EOG without number - try to extract any trailing digit
            match = re.search(r"(\d)$", electrode)
            if match:
                return (electrode, "A2", True)
            # Generic EOG name like "EOGL", "EOGR"
            # Return as-is with A2 reference
            return (electrode, "A2", True)

        # Special handling for EMG channels
        # For standalone EMG channels (ChEMG1, ChEMG2), treat as monopolar electrodes
        # referenced to a common reference, allowing rereferencing to combine them:
        # EMG1-EMG2 = (EMG1-COMMON) - (EMG2-COMMON) = ChEMG1 - ChEMG2
        if electrode.startswith("EMG") or "SUBMENT" in electrode:
            # Check for common submental/chin patterns without numbers
            # These represent the full bipolar recording
            if ("SUBMENT" in electrode or "CHIN" in electrode) and not re.search(
                r"\d", electrode
            ):
                return ("EMG1", "EMG2", True)
            # EMG with number: treat as monopolar referenced to common reference
            # This allows EMG1-EMG2 to be derived from ChEMG1 and ChEMG2
            match = re.search(r"EMG[\s_]*(\d+)", electrode)
            if match:
                num = match.group(1)
                # Treat as monopolar: EMG1-COMMON, EMG2-COMMON, etc.
                # Rereferencing can then compute: EMG1-EMG2 = (EMG1-COMMON) - (EMG2-COMMON)
                return (f"EMG{num}", "EMGREF", True)
            # Generic EMG without clear identifier - assume it's the full bipolar
            if electrode == "EMG":
                return ("EMG1", "EMG2", True)
            # Unknown EMG variant
            return None

        # EEG electrode inference based on hemisphere
        # Odd numbers (1, 3, 5, 7, 9) -> left hemisphere -> A2/M2 reference
        # Even numbers (2, 4, 6, 8) -> right hemisphere -> A1/M1 reference
        # Z (midline) -> typically A1 or A2, we'll use A2 by convention

        # Extract trailing digit if present
        match = re.search(r"(\d)$", electrode)
        if match:
            digit = int(match.group(1))
            if digit % 2 == 1:  # Odd -> left -> A2
                return (electrode, "A2", True)
            else:  # Even -> right -> A1
                return (electrode, "A1", True)
        # Check for 'Z' suffix (midline)
        elif electrode.endswith("Z"):
            return (electrode, "A2", True)  # Convention: midline -> A2

    return None


def normalize_electrode(electrode: str) -> str:
    """Normalize electrode name for matching.

    Handles common variations:
    - M1/M2 ↔ A1/A2 (mastoid references)
    - E1/E2 ↔ EOG1/EOG2 (EOG electrode naming)
    - T3/T4 ↔ T7/T8 (temporal electrodes, 10-20 vs 10-10 system)
    - FP1/FP2 ↔ Fp1/Fp2 (frontal pole case variations)

    Args:
        electrode: Raw electrode name

    Returns:
        Normalized electrode name
    """
    elec = electrode.strip().upper()

    # Handle channels named like "ChEMG1"/"ChEMG2"
    if elec.startswith("CHEMG"):
        elec = elec[2:]

    # Mastoid equivalences (different naming conventions, same location)
    mastoid_map = {
        "M1": "A1",
        "M2": "A2",
    }
    if elec in mastoid_map:
        elec = mastoid_map[elec]

    # EOG electrode equivalences (E1/E2 are shorthand for EOG1/EOG2;
    # LOC/ROC are the older left/right outer-canthus names for the same
    # electrodes and appear in MrOS visit 1, CCSHS, CFS, SOF and PNP)
    eog_map = {
        "E1": "EOG1",
        "E2": "EOG2",
        "LOC": "EOG1",
        "ROC": "EOG2",
    }
    if elec in eog_map:
        elec = eog_map[elec]

    # 10-20 to 10-10 system equivalences (old vs new naming)
    system_map = {
        "T3": "T7",
        "T4": "T8",
        "T5": "P7",
        "T6": "P8",
    }
    if elec in system_map:
        elec = system_map[elec]

    # Normalize case for frontal pole
    if elec.startswith("FP"):
        elec = "Fp" + elec[2:]

    return elec


class DerivationInfo:
    """Information about how a channel was derived."""

    def __init__(
        self,
        target: str,
        method: str,
        sources: list[str] | None = None,
        coefficients: list[float] | None = None,
        description: str | None = None,
    ):
        """Initialize derivation info.

        Args:
            target: Target channel name
            method: 'direct', 'telescoped', 'reference_swap', or 'not_derivable'
            sources: List of source channel names used
            coefficients: Coefficients for linear combination
            description: Human-readable explanation
        """
        self.target = target
        self.method = method
        self.sources = sources or []
        self.coefficients = coefficients or []
        self.description = description or method

    def __repr__(self) -> str:
        if self.method == "direct":
            return f"DerivationInfo('{self.target}': direct match)"
        elif self.sources:
            terms = [
                f"{c:+.1f}*{s}" if c != 1.0 else f"+{s}"
                for c, s in zip(self.coefficients, self.sources, strict=False)
            ]
            formula = " ".join(terms).lstrip("+")
            return f"DerivationInfo('{self.target}' = {formula})"
        return f"DerivationInfo('{self.target}': {self.method})"


def graph_endpoints(channel: str) -> tuple[str, str] | None:
    """Return the normalized (positive, reference) electrodes a channel spans.

    A bare electrode name carries no reference of its own; the reference that
    :func:`parse_channel_derivation_ex` supplied for it was guessed. Model such
    a channel as what it physically is -- monopolar against the recording's
    common reference -- so that telescoping recovers a true bipolar pair:
    ``C3-A2 = (C3-COMMON) - (A2-COMMON)``. Anchoring each bare electrode to its
    *guessed* mastoid instead would invent an edge asserting the very
    derivation we are trying to build.

    Both the graph and the path-to-coefficient reconstruction must resolve
    endpoints through this function, or the traversal directions -- and so the
    signs -- disagree with the edges.

    Args:
        channel: Raw channel name from the recording.

    Returns:
        Normalized (positive, reference) endpoints, or None if the name is not
        interpretable as a derivation.
    """
    parsed = parse_channel_derivation_ex(channel)
    if parsed is None:
        return None
    pos, ref, inferred = parsed
    pos_norm = normalize_electrode(pos)
    ref_norm = normalize_electrode(ref)
    if inferred and ref_norm in _GUESSED_REFERENCES:
        ref_norm = COMMON_REFERENCE
    return (pos_norm, ref_norm)


def build_electrode_graph(available_channels: list[str]) -> dict[str, dict[str, str]]:
    """Build a graph of electrode connections from available bipolar channels.

    The graph maps: electrode -> {connected_electrode: channel_name}
    This allows pathfinding between electrodes.

    Args:
        available_channels: List of available bipolar channel names

    Returns:
        Graph dictionary mapping electrode to connected electrodes

    Example:
        >>> channels = ['C4-P4', 'P4-A1', 'C3-A2']
        >>> graph = build_electrode_graph(channels)
        >>> graph['C4']
        {'P4': 'C4-P4'}
        >>> graph['P4']
        {'C4': 'C4-P4', 'A1': 'P4-A1'}
    """
    graph: dict[str, dict[str, str]] = defaultdict(dict)

    for channel in available_channels:
        endpoints = graph_endpoints(channel)
        if endpoints is None:
            continue

        pos_norm, ref_norm = endpoints

        # Add bidirectional edges (we can compute both directions)
        graph[pos_norm][ref_norm] = channel
        graph[ref_norm][pos_norm] = channel

    return graph


def find_monopolar_channel(
    electrode: str,
    available_channels: list[str],
) -> str | None:
    """Return the bare (monopolar) channel recording ``electrode``, if present.

    A bare channel is one whose name is a single electrode, so that
    :func:`parse_channel_derivation_ex` had to guess its reference. This is how
    we tell "the mastoid is actually in this file" from "we assumed a mastoid".

    Args:
        electrode: Electrode name to look for (normalized internally).
        available_channels: Channel names present in the recording.

    Returns:
        The matching channel name, or None if no bare channel records it.
    """
    wanted = normalize_electrode(electrode)
    for channel in available_channels:
        parsed = parse_channel_derivation_ex(channel)
        if parsed is None:
            continue
        pos, ref, inferred = parsed
        if not inferred or normalize_electrode(ref) not in _GUESSED_REFERENCES:
            continue
        if normalize_electrode(pos) == wanted:
            return channel
    return None


def _guess_would_shadow_real_reference(
    available_channel: str,
    target_ref: str,
    available_channels: list[str],
) -> bool:
    """True when accepting ``available_channel`` as already-referenced is a lie.

    Fires when the candidate is a bare electrode whose reference we merely
    guessed, *and* the reference electrode the target asks for is itself
    recorded in the file -- meaning the honest derivation is a subtraction we
    are able to perform.
    """
    parsed = parse_channel_derivation_ex(available_channel)
    if parsed is None:
        return False
    _pos, ref, inferred = parsed
    if not inferred or normalize_electrode(ref) not in _GUESSED_REFERENCES:
        return False
    return find_monopolar_channel(target_ref, available_channels) is not None


def find_derivation_path(
    target_pos: str,
    target_ref: str,
    available_channels: list[str],
) -> tuple[list[str], list[float]] | None:
    """Find a path to derive target channel from available channels.

    Uses BFS to find shortest path through electrode graph.

    Args:
        target_pos: Positive electrode of target derivation
        target_ref: Reference electrode of target derivation
        available_channels: List of available channel names

    Returns:
        Tuple of (channel_list, coefficients) or None if no path found

    Example:
        >>> # Want C4-A1, have C4-P4 and P4-A1
        >>> channels = ['C4-P4', 'P4-A1']
        >>> path = find_derivation_path('C4', 'A1', channels)
        >>> path
        (['C4-P4', 'P4-A1'], [1.0, 1.0])
        >>> # C4-A1 = (C4-P4) + (P4-A1)
    """
    pos_norm = normalize_electrode(target_pos)
    ref_norm = normalize_electrode(target_ref)

    # Check for direct match first
    for channel in available_channels:
        parsed = parse_channel_derivation(channel)
        if parsed is None:
            continue
        pos, ref = parsed
        if (
            normalize_electrode(pos) == pos_norm
            and normalize_electrode(ref) == ref_norm
        ):
            # Do not accept a guessed reference when the real one is on disk.
            if _guess_would_shadow_real_reference(
                channel, target_ref, available_channels
            ):
                continue
            return ([channel], [1.0])

    # Build electrode connection graph
    graph = build_electrode_graph(available_channels)

    if pos_norm not in graph or ref_norm not in graph:
        return None

    # BFS to find path from target_pos to target_ref
    from collections import deque

    queue = deque([(pos_norm, [])])
    visited: set[str] = {pos_norm}

    while queue:
        current, path = queue.popleft()

        if current == ref_norm:
            # Found a path!
            if not path:
                return None  # Same electrode (shouldn't happen)

            # Convert path to channels and coefficients
            # Need to track direction to compute correct signs
            channels = []
            coefficients = []

            # Reconstruct the electrode sequence to determine edge directions
            current_electrode = pos_norm
            for channel_name in path:
                endpoints = graph_endpoints(channel_name)
                if endpoints is None:
                    continue
                seg_pos_norm, seg_ref_norm = endpoints

                # Determine which direction we're traversing this edge
                # If we're going from seg_pos to seg_ref (forward), coefficient is +1
                # If we're going from seg_ref to seg_pos (backward), coefficient is -1
                if current_electrode == seg_pos_norm:
                    # Forward: current -> reference
                    coefficients.append(1.0)
                    current_electrode = seg_ref_norm
                elif current_electrode == seg_ref_norm:
                    # Backward: reference -> positive (reversed)
                    coefficients.append(-1.0)
                    current_electrode = seg_pos_norm
                else:
                    # Shouldn't happen if BFS is correct
                    coefficients.append(1.0)

                channels.append(channel_name)

            return (channels, coefficients)

        # Explore neighbors
        for neighbor, channel_name in graph[current].items():
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [channel_name]))

    return None


def can_derive_channel(
    target: str,
    available_channels: list[str],
) -> bool:
    """Check if target channel can be derived from available channels.

    Args:
        target: Target channel name (e.g., 'C4-A1')
        available_channels: List of available channel names

    Returns:
        True if derivable, False otherwise
    """
    parsed = parse_channel_derivation(target)
    if parsed is None:
        # Not a bipolar channel, can't derive
        # Check for direct match
        target_norm = target.strip().upper()
        for ch in available_channels:
            if ch.strip().upper() == target_norm:
                return True
        return False

    pos, ref = parsed
    path = find_derivation_path(pos, ref, available_channels)
    return path is not None


def build_derivation_matrix(
    target: str,
    available_channels: list[str],
) -> tuple[np.ndarray | None, DerivationInfo]:
    """Build transformation matrix to derive target from available channels.

    Args:
        target: Target channel name
        available_channels: List of available channel names

    Returns:
        Tuple of (matrix, derivation_info)
        - matrix: Shape (1, n_available) array of coefficients, or None if not derivable
        - derivation_info: Information about the derivation

    Example:
        >>> available = ['C4-P4', 'P4-A1']
        >>> matrix, info = build_derivation_matrix('C4-A1', available)
        >>> matrix
        array([[1., 1.]])
        >>> # derived = matrix @ np.vstack([C4_P4_signal, P4_A1_signal])
    """
    # Try to parse target as bipolar (handles single electrodes with inferred reference)
    parsed_target = parse_channel_derivation(target)
    if parsed_target is None:
        info = DerivationInfo(
            target=target,
            method="not_derivable",
            description="Not a bipolar channel or not available",
        )
        return None, info

    target_pos, target_ref = parsed_target
    target_pos_norm = normalize_electrode(target_pos)
    target_ref_norm = normalize_electrode(target_ref)

    # Check for direct match (string match or semantic match)
    target_norm = target.strip().upper()
    for idx, ch in enumerate(available_channels):
        # Try string match first
        if ch.strip().upper() == target_norm:
            matrix = np.zeros((1, len(available_channels)))
            matrix[0, idx] = 1.0
            info = DerivationInfo(
                target=target,
                method="direct",
                sources=[ch],
                coefficients=[1.0],
                description=f"Direct match: {ch}",
            )
            return matrix, info

        # Try semantic match (e.g., "F3" in EDF matches "F3-A2" in canonical)
        parsed_avail = parse_channel_derivation(ch)
        if parsed_avail is not None:
            avail_pos, avail_ref = parsed_avail
            avail_pos_norm = normalize_electrode(avail_pos)
            avail_ref_norm = normalize_electrode(avail_ref)

            if avail_pos_norm == target_pos_norm and avail_ref_norm == target_ref_norm:
                # `ch` is only "already referenced" because we guessed it was.
                # If the reference electrode is itself recorded, fall through to
                # the derivation path and subtract it for real.
                if _guess_would_shadow_real_reference(
                    ch, target_ref, available_channels
                ):
                    continue
                matrix = np.zeros((1, len(available_channels)))
                matrix[0, idx] = 1.0
                info = DerivationInfo(
                    target=target,
                    method="direct",
                    sources=[ch],
                    coefficients=[1.0],
                    description=f"Direct match: {ch} (inferred as {avail_pos}-{avail_ref})",
                )
                return matrix, info

    pos, ref = parsed_target

    # Find derivation path
    path_result = find_derivation_path(pos, ref, available_channels)
    if path_result is None:
        info = DerivationInfo(
            target=target,
            method="not_derivable",
            description=f"No derivation path found for {target}",
        )
        return None, info

    path_channels, path_coeffs = path_result

    # Build transformation matrix
    matrix = np.zeros((1, len(available_channels)))

    # Map path channels to indices
    channel_to_idx = {ch.strip().upper(): i for i, ch in enumerate(available_channels)}

    for ch, coeff in zip(path_channels, path_coeffs, strict=False):
        ch_norm = ch.strip().upper()
        if ch_norm in channel_to_idx:
            idx = channel_to_idx[ch_norm]
            matrix[0, idx] = coeff

    if len(path_channels) == 1:
        method = "direct"
        description = f"Direct: {path_channels[0]}"
    else:
        method = "telescoped"
        # Render the actual signs; the coefficients are what gets applied to the
        # signal, and a formula that prints "+" for a subtraction misdescribes
        # the stored channel to everything downstream.
        terms = []
        for i, (ch, coeff) in enumerate(zip(path_channels, path_coeffs, strict=False)):
            sign = "-" if coeff < 0 else "+"
            magnitude = abs(coeff)
            factor = "" if magnitude == 1.0 else f"{magnitude:g}*"
            if i == 0:
                terms.append(f"{'-' if coeff < 0 else ''}{factor}({ch})")
            else:
                terms.append(f"{sign} {factor}({ch})")
        description = f"Telescoped: {target} = {' '.join(terms)}"

    info = DerivationInfo(
        target=target,
        method=method,
        sources=path_channels,
        coefficients=path_coeffs,
        description=description,
    )

    return matrix, info


def build_montage_transform(
    target_channels: list[str],
    available_channels: list[str],
    verbose: bool = False,
) -> tuple[np.ndarray, list[DerivationInfo], list[int]]:
    """Build full transformation matrix to derive target montage from available.

    Args:
        target_channels: List of desired channel names
        available_channels: List of available channel names
        verbose: If True, print derivation information

    Returns:
        Tuple of (transform_matrix, derivation_info_list, missing_mask)
        - transform_matrix: Shape (n_target, n_available) matrix
        - derivation_info_list: List of DerivationInfo objects for each target
        - missing_mask: List of 0/1 indicating which targets are available (1) or missing (0)

    Example:
        >>> target = ['C4-A1', 'C3-A2', 'F4-A1']
        >>> available = ['C4-P4', 'P4-A1', 'C3-A2', 'F4-M1']
        >>> matrix, info, mask = build_montage_transform(target, available)
        >>> # Apply: derived_montage = matrix @ available_signals
        >>> # Check which channels are available: mask == [1, 1, 1] means all derivable
    """
    n_target = len(target_channels)
    n_available = len(available_channels)

    transform = np.zeros((n_target, n_available), dtype=np.float32)
    info_list: list[DerivationInfo] = []
    mask: list[int] = []

    for i, target in enumerate(target_channels):
        row, info = build_derivation_matrix(target, available_channels)

        if row is not None:
            transform[i, :] = row[0, :]
            mask.append(1)
            if verbose:
                print(f"✓ {info}")
        else:
            # Leave row as zeros (will be zero-padded)
            mask.append(0)
            if verbose:
                print(f"✗ {info}")

        info_list.append(info)

    return transform, info_list, mask


# Example usage and tests
if __name__ == "__main__":
    print("=" * 80)
    print("Channel Re-referencing Examples")
    print("=" * 80)

    # Example 1: Simple telescoping
    print("\n1. Simple Telescoping: C4-A1 from C4-P4 and P4-A1")
    available = ["C4-P4", "P4-A1"]
    target = "C4-A1"
    matrix, info = build_derivation_matrix(target, available)
    print(f"   {info}")
    print(f"   Matrix: {matrix}")

    # Example 2: Multiple paths
    print("\n2. Complex Montage Transformation")
    target_montage = ["C4-A1", "C3-A2", "F4-A1", "O2-A1"]
    available_montage = ["C4-P4", "P4-A1", "C3-A2", "F4-CZ", "CZ-A1", "O2-M1"]
    transform, info_list, mask = build_montage_transform(
        target_montage, available_montage, verbose=True
    )
    print(f"\n   Transform matrix shape: {transform.shape}")
    print(f"   Channel availability mask: {mask}")
    print(f"   Available: {sum(mask)}/{len(mask)} channels")

    # Example 3: Reference equivalence
    print("\n3. Reference Equivalence: A1/A2 ↔ M1/M2")
    available = ["C4-M1", "F3-M2"]
    targets = ["C4-A1", "F3-A2"]
    for tgt in targets:
        matrix, info = build_derivation_matrix(tgt, available)
        print(f"   {info}")

    print("\n" + "=" * 80)
