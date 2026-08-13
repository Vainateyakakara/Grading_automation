from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from asag.dataset_adapters import merge_datasets


def _discover_files(data_dir: Path) -> list[Path]:
    exts = {".csv", ".tsv", ".json", ".jsonl", ".txt"}
    files = []
    for p in data_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return sorted(files)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="*", default=None, help="Input dataset files")
    parser.add_argument("--data-dir", default=None, help="Directory to auto-discover dataset files")
    parser.add_argument("--out", required=True, help="Merged CSV output path")
    parser.add_argument("--report", default=None, help="Optional JSON report path")
    args = parser.parse_args()

    input_paths: list[Path] = []
    if args.inputs:
        input_paths.extend(Path(p) for p in args.inputs)
    if args.data_dir:
        input_paths.extend(_discover_files(Path(args.data_dir)))

    input_paths = sorted(set(input_paths))
    if not input_paths:
        raise SystemExit("No input files provided. Use --inputs or --data-dir")

    merged = merge_datasets(input_paths)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)

    report = {
        "input_files": [str(p) for p in input_paths],
        "rows": int(len(merged)),
        "score_min": float(merged["score"].min()),
        "score_max": float(merged["score"].max()),
        "dataset_counts": merged["dataset"].value_counts().to_dict(),
    }

    report_path = Path(args.report) if args.report else out_path.with_suffix(".report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Merged dataset saved: {out_path}")
    print(f"Report saved: {report_path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
