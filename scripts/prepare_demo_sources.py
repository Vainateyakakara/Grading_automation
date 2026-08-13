from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_asap_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"essay_set": 1, "essay": "Plants use sunlight to produce glucose and oxygen.", "domain1_score": 10},
            {"essay_set": 1, "essay": "Photosynthesis is only done by animals.", "domain1_score": 2},
            {"essay_set": 2, "essay": "Paris is the capital of France.", "domain1_score": 12},
            {"essay_set": 2, "essay": "The capital of France is Rome.", "domain1_score": 1},
        ]
    )


def build_scientsbank_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "reference_answer": "Water boils at 100 degrees Celsius at sea level.",
                "student_answer": "At sea level water boils at 100 C.",
                "label": "correct",
            },
            {
                "reference_answer": "Water boils at 100 degrees Celsius at sea level.",
                "student_answer": "Water freezes at 100 C.",
                "label": "incorrect",
            },
        ]
    )


def build_mohler_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model_answer": "TCP is connection-oriented and reliable.",
                "student_response": "TCP provides reliable connection-oriented transport.",
                "score": 5,
            },
            {
                "model_answer": "TCP is connection-oriented and reliable.",
                "student_response": "TCP is always faster than UDP and unreliable.",
                "score": 1,
            },
        ]
    )


def build_sts_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"sentence1": "A dog is running through a field.", "sentence2": "A canine runs in a grassy area.", "score": 4.8},
            {"sentence1": "A dog is running through a field.", "sentence2": "A person is cooking dinner.", "score": 0.3},
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--replicate", type=int, default=50, help="Replication factor for demo data volume")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    asap = pd.concat([build_asap_rows()] * args.replicate, ignore_index=True)
    scb = pd.concat([build_scientsbank_rows()] * args.replicate, ignore_index=True)
    moh = pd.concat([build_mohler_rows()] * args.replicate, ignore_index=True)
    sts = pd.concat([build_sts_rows()] * args.replicate, ignore_index=True)

    asap.to_csv(out_dir / "asap_demo.tsv", sep="\t", index=False)
    scb.to_csv(out_dir / "scientsbank_demo.csv", index=False)
    moh.to_csv(out_dir / "mohler_demo.csv", index=False)
    sts.to_csv(out_dir / "sts_demo.csv", index=False)

    print(f"Demo source files written to: {out_dir}")


if __name__ == "__main__":
    main()
