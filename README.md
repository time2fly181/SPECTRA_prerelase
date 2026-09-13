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
dropout options. A checkpoint's saved context half-width overrides the fallback
CLI value. Checkpoints with only `context_epochs` use that saved window length.

## Checkpoints

Supply a trusted PSGStage inference/supervised checkpoint whose
`model_config.model_kwargs.epoch_encoder_variant` is `multirate_asymmetric`.
The runtime reconstructs the transformer and the saved
`multirate_asymmetric_encoder_kwargs`, including band normalization and optional
recording conditioning. Learned state-dictionary names are preserved. Common
compiled/distributed wrapper prefixes and legacy fixed-filter buffers retain
their compatibility handling.

Other encoder architectures and standalone CNN or temporal-pretraining
checkpoints are outside this release. No pretrained weights are bundled, and
the included tests use synthetic signals and randomly initialized models.
Checkpoint loading uses PyTorch deserialization; load only files you trust.

Some older checkpoints contain auxiliary heads or source-offset state. Small
compatibility containers retain that state for reconstruction; their training
losses and source-specific prediction adjustments are absent.

## EDF preprocessing

The scoring path follows `batch_edf_to_zarr_fp32.py` from the source snapshot
identified in [provenance](docs/PROVENANCE.md):

- Select five ordered slots: `EEG1`, `EEG2`, `EOG1`, `EOG2`, `EMG`, with the
  converter's current modality priorities and mentalis/chin fallback.
- Read directly recorded channels, without algebraic rereferencing.
- Resample each selected channel directly from its native rate to **128 Hz**.
- Form **30-second / 3840-sample** epochs. Incomplete trailing epochs are omitted.
- Use signal-valid epochs to calculate per-recording, per-channel median/IQR
  scaling, clip to ±20, and recheck validity after normalization.
- Require at least 10 usable epochs per channel for normalization. Zero-fill
  unavailable channels and invalid channel epochs; preserve interior gaps.
- Trim fully invalid exterior epochs while preserving original epoch indices
  in prediction exports. Invalid/out-of-window predictions are `-1`.

No bandpass or notch filter is applied to model inputs. GUI display filters
affect waveform viewing only. Header repair and MNE fallback are retained.

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

result = score_recording(
    "recording.edf", "model.ckpt", None, "outputs/recording",
    options=ScoreOptions(batch_size=8),
)
```

The immutable class order is **Wake, N1, N2, N3, REM** (0–4). The runtime returns
predictions, probabilities, uncertainty values, validity masks, channel mapping,
and output paths. It saves a preprocessing JSON sidecar and a per-epoch channel
validity array alongside prediction exports.

## Development and packaging

```bash
uv run pytest -q
uv run ruff check src tests inference_gui.py
uv run black --check src tests inference_gui.py
uv build
```

See [validation](docs/VALIDATION.md) for checks actually performed. The source
distribution and wheel are written to `dist/`. The Python distribution is named
`spectra-sleep-staging`; its import package is `spectra`.

Licensing has not been selected. See [NOTICE](NOTICE). Recordings, checkpoints,
outputs, and environments are ignored by Git.
