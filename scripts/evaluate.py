import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from asag.data import load_and_standardize
from asag.features import transform_features
from asag.metrics import evaluate_regression


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, help="Baseline artifact directory")
    parser.add_argument("--data", required=True, help="Evaluation CSV path")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    regressor = joblib.load(model_dir / "ridge.joblib")
    vectorizer = joblib.load(model_dir / "tfidf.joblib")

    with open(model_dir / "score_range.json", "r", encoding="utf-8") as f:
        score_range = json.load(f)

    df = load_and_standardize(args.data)
    X = transform_features(df, vectorizer)
    y_true = df["score"].to_numpy()
    y_pred = regressor.predict(X)

    metrics = evaluate_regression(
        y_true,
        y_pred,
        min_score=float(score_range["min_score"]),
        max_score=float(score_range["max_score"]),
    )

    print("Evaluation metrics:", metrics)
    if "dataset" in df.columns:
        report = {}
        for dataset_name, group in df.groupby("dataset"):
            idx = group.index.to_numpy()
            m = evaluate_regression(
                y_true=y_true[idx],
                y_pred=y_pred[idx],
                min_score=float(score_range["min_score"]),
                max_score=float(score_range["max_score"]),
            )
            report[dataset_name] = m
        print("Per-dataset metrics:")
        print(pd.DataFrame(report).T.to_string())


if __name__ == "__main__":
    main()
