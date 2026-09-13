"""Command-line EDF scoring using the same runtime as the desktop GUI."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    """Score a recording, or launch the desktop GUI with ``--gui``."""
    parser = argparse.ArgumentParser(description="SPECTRA sleep-stage inference")
    parser.add_argument(
        "--gui", action="store_true", help="Open the desktop application"
    )
    parser.add_argument("--edf", type=Path, help="Input EDF recording")
    parser.add_argument(
        "--checkpoint", type=Path, help="Multirate CNN/transformer checkpoint"
    )
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--canonical", type=Path, help="Optional JSON list of five channel slots"
    )
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda", "mps"], default="auto"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--context-half",
        type=int,
        default=15,
        help="Fallback context half-width; saved checkpoint metadata takes precedence",
    )
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument(
        "--end-epoch",
        type=int,
        default=-1,
        help="Exclusive end, or -1 for recording end",
    )
    parser.add_argument("--amp", choices=["fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=0,
        help="Enable MC dropout with this many samples; 0 disables it",
    )
    args = parser.parse_args(argv)
    if args.gui:
        from spectra.gui import main as gui_main

        gui_main()
        return
    if args.edf is None or args.checkpoint is None:
        parser.error("--edf and --checkpoint are required unless --gui is used")
    for path in (args.edf, args.checkpoint, args.canonical):
        if path is not None and not path.is_file():
            parser.error(f"File does not exist: {path}")
    if args.batch_size < 1 or args.context_half < 0 or args.mc_samples < 0:
        parser.error(
            "batch-size must be positive; context-half and mc-samples must be nonnegative"
        )
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    from spectra.inference import ScoreOptions, score_recording

    result = score_recording(
        edf_path=str(args.edf),
        checkpoint=str(args.checkpoint),
        canon_json=str(args.canonical) if args.canonical else None,
        output_dir=str(args.output),
        device=args.device,
        options=ScoreOptions(
            batch_size=args.batch_size,
            context_half=args.context_half,
            start_epoch=args.start_epoch,
            end_epoch=args.end_epoch,
            amp_mode=args.amp,
            use_mc_dropout=args.mc_samples > 0,
            mc_samples=max(1, args.mc_samples),
            sequential_loading=True,
        ),
    )
    print(f"Scored {result['score_window']} on a {result['n_epochs']}-epoch recording")
    print(f"Results: {args.output.resolve()}")
