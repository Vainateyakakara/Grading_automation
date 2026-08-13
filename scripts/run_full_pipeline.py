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
    parser.add_argument("--real-data-dir", default=None, help="Directory containing real ASAP/SciEntsBank/Mohler/STS files")
    parser.add_argument("--replicate", type=int, default=50, help="Demo data replication factor when real data is unavailable")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    raw_dir = root / "data" / "raw"
    processed_dir = root / "data" / "processed"
    artifacts_dir = root / "artifacts" / "baseline"

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

    run(
        [
            "python3",
            "scripts/train_baseline.py",
            "--data",
            str(merged_csv),
            "--config",
            "configs/baseline.yaml",
            "--out",
            str(artifacts_dir),
        ],
        cwd=root,
    )

    run(
        [
            "python3",
            "scripts/evaluate.py",
            "--model-dir",
            str(artifacts_dir),
            "--data",
            str(merged_csv),
        ],
        cwd=root,
    )

    print("Pipeline complete.")
    print(f"Merged dataset: {merged_csv}")
    print(f"Baseline artifacts: {artifacts_dir}")


if __name__ == "__main__":
    main()
