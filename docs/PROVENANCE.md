# Source provenance

Originally extracted from PSGStage commit `16643e0d91e669657ebcf39a45023fca8104aed0`.
The original worktree was clean when extraction began. The source repository
and the existing SPECTRA research wiki are preserved.

[source-map.json](source-map.json) maps extracted files to their source paths.
The `spectra` namespace replaces the original training-named package; learned
state-dictionary paths do not depend on that Python package rename.

The transformer constructor is specialized to the multirate asymmetric encoder.
Training loss methods, pretraining constructors, optimizer/checkpoint-saving
helpers, and unrelated encoder classes are excluded. Shared convolution,
attention, pooling, engineered-feature, preprocessing, and GUI review modules
remain where the inference runtime requires them.

`preprocessing/edf_reader.py` reuses the canonical converter's header repair,
MNE reader adapter, and polyphase resampling implementation.
`preprocessing/edf.py` supplies label-free selection/normalization using the
same channel utilities, signal-quality checks, and robust-scaling helper.
Scoring and GUI channel preview share this path. Embedded checkpoint wrappers
are reconstructed for state loading, then their base models receive the
externally normalized waveforms to avoid double normalization.

EDF scoring pins the recording's band-normalization statistics through both
sequential and materialized inference, with cleanup on exit. This closes a gap
in the original GUI scoring entry point, which did not use the existing pinning
context. Desktop settings use a separate `.spectra` directory.

On September 13, 2026, the recording-band normalization fixes were ported from
the updated PSGStage source into the standalone runtime and its band-norm
access helper. Direct windowed/sequential inference now prepares statistics;
reasoning retains them through refinement; masking and nested cleanup follow
the corrected source behavior. Only inference dependencies were added. See
[VALIDATION.md](VALIDATION.md) for regression and real-checkpoint parity results.

The legacy TTA option uses only its requested gain/noise/shift perturbations;
it no longer imports training augmentation, whose defaults also corrupted or
masked channels. Enhanced TTA and MC-dropout behavior are retained.

No Git history, training configuration, participant data, local paths,
credentials, experiment results, or checkpoint weights are imported.
