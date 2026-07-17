"""Evaluate the released Baller2Vec–DisTraj checkpoint on prepared test data."""

from __future__ import annotations

import argparse
from pathlib import Path

from main import main as run_main


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "distraj_baller2vec_ratio1_combo_140games.pt"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the released DisTraj checkpoint on a prepared test split."
    )
    parser.add_argument("--pt_dir", required=True)
    parser.add_argument("--sample_ml_dir", required=True)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--output_dir", default="outputs/eval_checkpoint")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--strict_resume", action="store_true")
    args, extra = parser.parse_known_args()
    return args, extra


def main():
    args, extra = parse_args()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    main_args = [
        "--pt_dir",
        args.pt_dir,
        "--sample_ml_dir",
        args.sample_ml_dir,
        "--checkpoint",
        str(checkpoint),
        "--eval_only",
        "--output_dir",
        args.output_dir,
        "--games",
        "140",
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        str(args.num_workers),
        "--traj_model",
        "baller2vec",
        "--event_head",
        "disentangled",
        "--dynamic_weight",
        "1.0",
        "--use_context",
        "--use_voronoi_spatial",
    ]
    if args.device:
        main_args.extend(["--device", args.device])
    if args.strict_resume:
        main_args.append("--strict_resume")
    run_main(main_args + extra)


if __name__ == "__main__":
    main()
