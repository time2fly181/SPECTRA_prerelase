# SPECTRA

Sleep-stage inference for EDF polysomnography recordings, with a multirate
asymmetric CNN, per-recording band normalization, and a context transformer.
Includes a desktop application for scoring, waveform review, uncertainty
inspection, manual stage corrections, and export.

This repository contains inference code and its supporting model, preprocessing,
review, and export modules. It contains no trainers, training datasets, optimizers,
pretraining pipelines, experiments, recordings, or model weights.

## Install and run

Use Python 3.12 or newer and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --locked
uv run spectra-gui
```

The desktop application also launches with `uv run python inference_gui.py`.
Select a compatible checkpoint and EDF in the Setup tab, then start scoring.
CPU, CUDA, and Apple MPS device selection are available. The default scoring
precision is fp32; sequential loading limits context-window memory use.

Command-line scoring uses the same runtime:

```bash
uv run spectra \
  --edf /path/to/recording.edf \
  --checkpoint /path/to/model.ckpt \
  --output outputs/recording \
  --device auto
```

Run `uv run spectra --help` for analysis interval, batch size, precision, and MC
dropout options. Sampling is fixed at **128 Hz** and context half-width at **10**
(**21 epochs** total); neither can be changed in the CLI, GUI, or scoring options.
Conflicting checkpoint geometry is rejected. MC dropout is available; test-time
augmentation and iterative/recurrent refinement are unsupported. The multirate
model uses Kaiser anti-alias filters.

## Checkpoints

Supply a trusted PSGStage inference/supervised checkpoint whose
`model_config.model_kwargs.epoch_encoder_variant` is `multirate_asymmetric`.
The runtime reconstructs the transformer and the saved
`multirate_asymmetric_encoder_kwargs`, including band normalization.
Models are unwrapped; all EDF waveform preprocessing runs through
`spectra.preprocessing.edf.preprocess_edf` before model inference. Recording
conditioning and embedded model preprocessing are unsupported, and checkpoints
containing either are rejected. Learned state-dictionary names are preserved. Common
compiled/distributed wrapper prefixes and legacy fixed-filter buffers retain
their compatibility handling.

Only waveform-based, epoch-token CNN + transformer inference is included.
Engineered/YASA features, feature fusion, axial and patch models, other CNN
architectures, and standalone CNN or temporal-pretraining checkpoints are
unsupported and rejected during loading. No pretrained weights are bundled, and
the included tests use synthetic signals and randomly initialized models.
Checkpoint loading uses PyTorch deserialization; load only files you trust.

Training-only auxiliary heads and source-offset state are excluded through an
explicit allowlist recorded in the checkpoint load audit. Unknown model weights
are rejected. Shared sinc filters and dilated convolution blocks remain because
the multirate encoder uses them; they do not provide alternative model paths.

## EDF preprocessing

The scoring path is implemented in
[`preprocessing/edf.py`](src/spectra/preprocessing/edf.py). The historical
[source map](docs/source-map.json) records extraction paths; it is not a parity
certificate or a record of subsequent edits.

- Select five ordered slots: `EEG1`, `EEG2`, `EOG1`, `EOG2`, `EMG`, with the
  converter's current modality priorities and mentalis/chin fallback.
- Read directly recorded channels, without algebraic rereferencing.
- With the primary reader, resample each selected channel directly from its
  native rate to **128 Hz**.
- Form **30-second / 3840-sample** epochs. Incomplete trailing epochs are omitted.
- Use signal-valid epochs to calculate per-recording, per-channel median/IQR
  scaling, clip to ±20, and recheck validity after normalization.
- Require at least 10 usable epochs per channel for normalization. Zero-fill
  unavailable channels and invalid channel epochs; preserve interior gaps.
- Trim fully invalid exterior epochs while preserving original epoch indices
  in prediction exports. Invalid/out-of-window predictions are `-1`.

No bandpass or notch filter is applied to model inputs. GUI display filters
affect waveform viewing only. Header repair and MNE fallback are retained.
The primary reader returns EDF physical values in the header's units; the MNE
adapter returns MNE values (volts for voltage channels) and exposes MNE's common
sampling rate before resampling to 128 Hz. After robust scaling, model inputs
are dimensionless. Do not interpret normalized amplitudes as microvolts.

**Analysis intervals matter:** the converter crops to the first/last annotated
epochs. Unlabeled inference cannot discover those annotation boundaries. Use
`--start-epoch` and `--end-epoch` (exclusive) to reproduce the converter's
analysis interval. Without them, statistics use the signal-valid recording.

Use GUI slot overrides, or `--canonical channels.json`, to select explicit
recorded channels. A JSON list such as
`["F4-M1", "C4-M1", "EOG1", "EOG2", "EMG"]` fixes the first two channels
and automatically selects the remaining slots. The automatic selector mirrors
the source converter; it does not enforce frontal/central anatomy.

## Python API

```python
from spectra.inference import ScoreOptions, score_recording
from spectra.preprocessing.edf import preprocess_edf

prepared = preprocess_edf("recording.edf", start_epoch=10, end_epoch=900)
print(prepared.signals_stacked.shape)  # [epochs, 5, 3840], float32

# score_recording reads and preprocesses the EDF itself.
result = score_recording(
    "recording.edf", "model.ckpt", None, "outputs/recording",
    options=ScoreOptions(batch_size=8, start_epoch=10, end_epoch=900),
)
```

The immutable class order is **Wake, N1, N2, N3, REM** (0–4). The runtime returns
predictions, probabilities, uncertainty values, validity masks, channel mapping,
and output paths. It saves a preprocessing JSON sidecar and a per-epoch channel
validity array alongside prediction exports.

## Model and data contracts

The epoch encoder processes EEG with short convolutions and rectified sinc-band
envelopes, EOG with a lower-rate convolution branch, and EMG with rectified
convolution envelopes. It fuses these branches, applies a dilated convolution
trunk, and pools each epoch into an embedding. The context transformer operates
on these embeddings. Branch widths, filter parameters, trunk strides, pooling,
attention, and classifier settings come from checkpoint metadata and compatible
state shapes. See the model constructors and `get_config()` for these settings;
do not reconstruct a trained model from assumed defaults.

| Interface | Shape and meaning |
| --- | --- |
| Prepared signals | Contiguous NumPy `float32 [N, 5, 3840]`; `N` is the retained analysis interval. |
| Recording presence | `uint8 [5]`; 1 means a selected channel survived normalization. |
| Signal validity | Boolean `[N, 5]`; true means that channel is usable in that epoch. |
| Context input | Floating tensor `[B, 21, 5, 3840]`; `B` is the number of center epochs in a batch. |
| Model output | Logits `[B, 5]`, or `[B, 21, 5]` with `predict_all=True`. |

Each context window contains ten epochs before and after its target at index 10.
Windows use only the current recording's retained analysis interval. Boundary
padding is zero and masked invalid; interior invalid epochs retain their original
positions. A channel mask and an epoch mask are distinct: an epoch is usable if
at least one channel is usable. Low-level model calls require preprocessed
waveforms; they do not read EDFs, resample, or apply median/IQR normalization.

When a checkpoint enables per-recording band normalization, the runtime computes
envelope statistics from valid epochs of this recording and pins them for the
scoring call. These statistics are separate from waveform median/IQR scaling.
The direct, sequential, and `infer_recording` entry points manage this context;
custom model forward calls must use `pinned_recording_band_statistics`. Nested
contexts and exceptions restore the previous statistics.

Center-only scoring is the default. The Python API also exposes overlap averaging
and MC pooling in `ScoreOptions`. With consistent aggregation, MC probability
pooling averages posteriors; logit pooling averages logits before softmax. These
are different estimators. Preserve the options, checkpoint, channel mapping,
analysis bounds, device, precision, package versions, and random seed when
comparing runs. MC dropout is stochastic and the scoring API does not set a seed.
Exact numerical agreement across accelerator backends or precision modes is not
guaranteed. The CLI uses its own batch-size default; inspect `--help` and
`ScoreOptions` rather than assuming CLI and Python defaults are identical.

Known limitation: `ScoreOptions.prefer_averaged` is currently not forwarded by
`score_recording`. Only the explicit `prefer_averaged=True` keyword on
`load_model_from_checkpoint` selects saved `ema.module` weights. Setting the
option field alone does not change which weights are scored.

## Outputs and reference evaluation

Outputs use the original EDF epoch grid, including excluded exterior epochs.
`score_window` is the retained half-open interval `[start, end)`. Invalid and
out-of-window epochs have prediction `-1` and zero probability, confidence, and
uncertainty rows. A zero uncertainty value on an invalid epoch is a sentinel,
not evidence of confidence.

For EDF stem `recording`, scoring writes the following files (existing files
with the same names are overwritten):

| File | Contents |
| --- | --- |
| `recording_predictions.csv` | `Epoch,Stage`; zero-based epochs, `W/N1/N2/N3/REM`, or `Unscored`. |
| `recording_probabilities.npy` | `float32 [original_n_epochs, 5]` probabilities in class order. |
| `recording_flags.npz` | Per-epoch uncertainty arrays; MC diagnostics when enabled. |
| `recording_epoch_signal_valid.npy` | Boolean `[original_n_epochs, 5]` channel validity. |
| `recording_preprocessing.json` | Channel mapping, normalization statistics, analysis bounds, and reader backend. |

Reference agreement and calibration are Python APIs used by the GUI; there is
no standalone calibration CLI. Agreement truncates the two label arrays to their
common length. Accuracy and per-stage recall count unscored model epochs as
disagreements against valid references; kappa and the confusion matrix use only
epochs where both labels are valid. Align reference epochs to the original EDF
grid before comparison.

Calibration drops invalid **labels**, not zero or invalid probability rows.
Filter unscored model rows explicitly when evaluating saved predictions:

```python
import numpy as np
from spectra.diagnostics.calibration_eval import evaluate_calibration

probabilities = np.load("outputs/recording/recording_probabilities.npy")
valid = np.load("outputs/recording/recording_epoch_signal_valid.npy").any(axis=1)
labels = np.load("recording_labels.npy")  # aligned integer labels, -1 for unknown
metrics = evaluate_calibration(probabilities[valid], labels[valid])
```

ECE weights each equal-width confidence bin by its fraction of evaluated epochs;
MCE is the largest bin gap. Bins are open on the left and closed on the right.
The multiclass Brier score sums squared errors over classes, then averages over
epochs. These are epoch-level metrics, not recording- or subject-macro averages.

## Development and packaging

```bash
uv run pytest -q
uv run ruff check src tests inference_gui.py
uv run black --check src tests inference_gui.py
uv build
```

Tests use synthetic EDFs and small randomly initialized models. They protect
preprocessing, checkpoint compatibility, fixed inference geometry, dropout, band
normalization, and GUI integration; they do not establish clinical performance
or parity on a private dataset. For a headless GUI smoke check, use
`QT_QPA_PLATFORM=offscreen uv run pytest -q tests/test_gui.py`.

Use the existing Google-style docstrings (`Args`, `Returns`, `Yields`, `Raises`,
and `Attributes` where relevant). Document scientific array shapes, units, masks,
side effects, and compatibility constraints at public boundaries. Keep simple
helpers concise and comments focused on non-obvious intent. Black and Ruff
configuration is authoritative in `pyproject.toml`. Preserve scientific checks
and supported checkpoint interfaces when cleaning up code.

The source distribution and wheel are written to `dist/`. The Python distribution
is named `spectra-sleep-staging`; its import package is `spectra`. Runtime entry
points are `spectra`, `python -m spectra`, `spectra-gui`, and `inference_gui.py`.
The `review` and `diagnostics` packages provide reference-analysis APIs; channel
utilities also support GUI waveform review. There are no dataset, training, loss,
split-generation, or standalone experiment entry points in this release.
