from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SAMPLE_ROWS = [
    {
        "reference_answer": "Photosynthesis is the process by which plants convert light energy into chemical energy.",
        "student_answer": "Plants use sunlight to make food through photosynthesis.",
        "score": 5,
    },
    {
        "reference_answer": "Photosynthesis is the process by which plants convert light energy into chemical energy.",
        "student_answer": "Photosynthesis happens in animals and creates oxygen only.",
        "score": 1,
    },
    {
        "reference_answer": "The capital of France is Paris.",
        "student_answer": "Paris is the capital city of France.",
        "score": 5,
    },
    {
        "reference_answer": "The capital of France is Paris.",
        "student_answer": "France capital is Berlin.",
        "score": 0,
    },
    {
        "reference_answer": "Newton's second law states that force equals mass times acceleration.",
        "student_answer": "Force is mass multiplied by acceleration.",
        "score": 5,
    },
    {
        "reference_answer": "Newton's second law states that force equals mass times acceleration.",
        "student_answer": "Force is related to speed but not mass.",
        "score": 1,
    },
    {
        "reference_answer": "Mitosis results in two genetically identical daughter cells.",
        "student_answer": "Mitosis creates two identical daughter cells.",
        "score": 5,
    },
    {
        "reference_answer": "Mitosis results in two genetically identical daughter cells.",
        "student_answer": "Mitosis makes four different cells.",
        "score": 0,
    },
    {
        "reference_answer": "The water cycle includes evaporation, condensation, and precipitation.",
        "student_answer": "Water evaporates, forms clouds by condensation, and falls as rain.",
        "score": 5,
    },
    {
        "reference_answer": "The water cycle includes evaporation, condensation, and precipitation.",
        "student_answer": "Water cycle means water is only stored in oceans.",
        "score": 1,
    },
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="Output CSV path")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(SAMPLE_ROWS)
    # Duplicate to make train/val/test split stable for demo runs.
    df = pd.concat([df] * 20, ignore_index=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote sample dataset: {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
