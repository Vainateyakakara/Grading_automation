from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from asag.baseline import train_baseline
from asag.config import load_yaml
from asag.data import load_and_standardize, split_data
from asag.features import fit_transform_features, transform_features
from asag.metrics import evaluate_regression


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--out", required=True, help="Output artifact directory")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_standardize(args.data)
    splits = split_data(
        df,
        test_size=float(cfg["test_size"]),
        val_size=float(cfg["val_size"]),
        random_state=int(cfg["random_state"]),
    )

    ngram_range = tuple(cfg["ngram_range"])
    fb = fit_transform_features(
        splits.train_df,
        max_features=int(cfg["max_tfidf_features"]),
        ngram_range=(int(ngram_range[0]), int(ngram_range[1])),
    )

    X_train = fb.X
    y_train = splits.train_df["score"].to_numpy()

    X_val = transform_features(splits.val_df, fb.vectorizer)
    y_val = splits.val_df["score"].to_numpy()

    X_test = transform_features(splits.test_df, fb.vectorizer)
    y_test = splits.test_df["score"].to_numpy()

    model = train_baseline(X_train, y_train, alpha=float(cfg["alpha"]))

    min_score = float(cfg["min_score"]) if cfg.get("min_score") is not None else float(df["score"].min())
    max_score = float(cfg["max_score"]) if cfg.get("max_score") is not None else float(df["score"].max())

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    val_metrics = evaluate_regression(y_val, val_pred, min_score=min_score, max_score=max_score)
    test_metrics = evaluate_regression(y_test, test_pred, min_score=min_score, max_score=max_score)

    joblib.dump(model.regressor, out_dir / "ridge.joblib")
    joblib.dump(fb.vectorizer, out_dir / "tfidf.joblib")

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"val": val_metrics, "test": test_metrics}, f, indent=2)

    with open(out_dir / "score_range.json", "w", encoding="utf-8") as f:
        json.dump({"min_score": min_score, "max_score": max_score}, f, indent=2)

    print("Baseline training complete")
    print("Validation metrics:", val_metrics)
    print("Test metrics:", test_metrics)
    print(f"Artifacts saved to: {out_dir}")


if __name__ == "__main__":
    main()
