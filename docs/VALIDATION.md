# Validation

Validated on Linux, Python 3.13.8, PyTorch 2.9.1, and an NVIDIA RTX 4080.
The shipped tests use synthetic EDFs and checkpoints. No patient recordings or
pretrained weights are included. External source-parity checks are identified
separately below.

## Recording-normalization update — September 13, 2026

Ported the corrected recording-statistics lifecycle from PSGStage into the
standalone runtime. Direct windowed and sequential inference now prepare band
statistics automatically. Reasoning retains the same complete-recording
statistics through refinement. EDF scoring retains its existing outer context
and now applies both recording presence and per-epoch channel validity masks.
Nested pinning restores previous overrides, including when scoring fails.
Statistics are computed in bounded batches using signal-valid epochs and
per-modality masks; absent modalities receive explicit neutral statistics.

The fork remains inference-only. There is no supervised validation loop to
patch here; the validation-table correction remains in PSGStage.

- Source suite: **24 passed**, including four new regressions that failed before
  the port (direct/sequential preparation, reasoning lifetime, nested cleanup).
- Isolated installed wheel: **24 passed**, with the original training/model
  packages absent and imports verified to come from the installed wheel.
- Ruff passed across source and tests. Black passed on all three changed Python
  files. A full Black check flags the pre-existing formatting of `src/spectra/gui.py`;
  that unrelated file was not changed.
- Rebuilt the wheel and source distribution with `uv build`.
- An external CUDA fp32 check loaded the supplied delta-fix epoch-46 checkpoint
  in both repositories and scored the same complete 786-epoch recording with
  context half-width 10 and batch size 4. All logits were **bit-for-bit identical**
  (maximum absolute difference **0.0**); normalization overrides were cleared
  after scoring in both runtimes. Neither the recording nor its weights are
  included in this repository or its distributions.

This verifies the port on the tested paths, not cohort-level staging accuracy.
The following sections retain the original extraction audit results.

## Standalone checks

`uv sync` created a separate environment from the declared dependencies.
The test suite confirms the original `psgstage_train` and `psg_models` packages
are absent from that environment.

`uv run pytest -q` passed **20 tests**, covering:

- Checkpoint learned tensors and band-normalization extra state restored
  exactly; reloaded model logits equal pre-save logits bit-for-bit.
- Both trunk stride schedules `(1,2,2,2)` and `(2,2,2,1)` at 128 Hz.
- EMA and robust per-recording statistics for EEG and EMG; statistics affect
  inference and are cleared after normal completion and exceptions.
- Actual sequential and materialized EDF-to-export scoring, with the recording
  band statistics pinned in both paths.
- CPU inference, MC-dropout variance, and RTX 4080 fp32/bf16 inference.
- Mixed-rate direct resampling, missing channels, explicit slot selection,
  insufficient usable signal, invalid exterior epochs, and interior gaps.
- Physical-dimension and start-time EDF header repair without altering the
  supplied EDF, producing identical normalized signals to the valid source.
- Headless Qt window construction/show/close, scoring/preview channel parity,
  and review waveforms matching the chosen recorded channels.

Ruff and Black checks pass for all packaged source and tests. Targeted Pyright
checks pass for the new EDF preprocessing, TTA, and CLI modules.

`uv build` produces both a source distribution and a wheel. A separate,
non-editable installation of the wheel imports the runtime, model, preprocessing,
and GUI modules and runs CLI help from outside the repository. All **20 tests
also pass against that isolated installed wheel**, including CUDA and Qt checks.
Archive inspection
found no training packages, recordings, checkpoints, environments, or Git history.

## Comparison with the original source

An external integration check imported both source snapshots, constructed the
same multirate CNN/transformer, copied its state strictly, and compared logits
with partially missing channels. Logits were bit-for-bit equal with band
normalization disabled, EEG+EMG EMA normalization, EEG+EMG robust normalization,
and robust normalization with recording conditioning enabled.

The same check ran the original `convert_edf_to_zarr_normalized` against a
synthetic mixed-rate EDF and compared its stored tensors with SPECTRA's
`preprocess_edf` output:

| Case | Shape | Maximum absolute tensor difference | Presence/validity masks |
| --- | --- | --- | --- |
| Automatic selection | `[12,5,3840]` | 0 | Identical |
| Matched annotation crop | `[12,5,3840]` | 0 | Identical |
| Explicit layout with a missing channel | `[12,5,3840]` | 0 | Identical |

The original converter comparison is a one-time extraction audit; the shipped
tests remain independent of the original repository and do not require Zarr.

## Limits

These checks establish implementation parity on the tested inputs, not clinical
accuracy or held-out performance. No real checkpoint/EDF pair was supplied for
this extraction. macOS/MPS, Windows, interactive manual-review/export workflows,
and compiled inference were not exercised. Annotation crop bounds must match
when comparing normalized data against labeled conversion output.
