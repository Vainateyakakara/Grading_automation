from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--real-data-dir", default=None)
    parser.add_argument("--replicate", type=int, default=50)
    parser.add_argument("--skip-cross-encoder", action="store_true")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    raw_dir = root / "data" / "raw"
    processed_dir = root / "data" / "processed"

    if args.real_data_dir:
        source_dir = Path(args.real_data_dir).resolve()
        if not source_dir.exists():
            raise SystemExit(f"Real data directory does not exist: {source_dir}")
    else:
        source_dir = raw_dir / "demo_sources"
        run(
            [
                "python3",
                "scripts/prepare_demo_sources.py",
                "--out-dir",
                str(source_dir),
                "--replicate",
                str(args.replicate),
            ],
            cwd=root,
        )

    merged_csv = processed_dir / "merged_asag.csv"
    run(
        [
            "python3",
            "scripts/merge_datasets.py",
            "--data-dir",
            str(source_dir),
            "--out",
            str(merged_csv),
        ],
        cwd=root,
    )

    cmd = [
        "python3",
        "scripts/train_best_model.py",
        "--data",
        str(merged_csv),
        "--ensemble-config",
        "configs/ensemble.yaml",
        "--cross-config",
        "configs/cross_encoder.yaml",
        "--out",
        str(root / "artifacts" / "best_model"),
    ]
    if args.skip_cross_encoder:
        cmd.append("--skip-cross-encoder")
    run(cmd, cwd=root)

    run(
        [
            "python3",
            "scripts/predict_best_model.py",
            "--model-dir",
            str(root / "artifacts" / "best_model"),
            "--reference",
            "The capital of France is Paris.",
            "--student",
            "Paris is the capital city of France.",
        ],
        cwd=root,
    )

    print("Best pipeline complete.")


if __name__ == "__main__":
    main()
