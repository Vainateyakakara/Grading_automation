from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from asag.best_model import save_model_selection
from asag.config import load_yaml
from asag.cross_encoder import train_cross_encoder
from asag.data import load_and_standardize, split_data
from asag.ensemble import save_ensemble, train_similarity_ensemble


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Merged canonical CSV")
    parser.add_argument("--ensemble-config", required=True, help="Ensemble config path")
    parser.add_argument("--cross-config", required=True, help="Cross-encoder config path")
    parser.add_argument("--out", required=True, help="Output model directory")
    parser.add_argument("--skip-cross-encoder", action="store_true", help="Train ensemble only")
    args = parser.parse_args()

    ensemble_cfg = load_yaml(args.ensemble_config)
    cross_cfg = load_yaml(args.cross_config)

    df = load_and_standardize(args.data)
    splits = split_data(
        df,
        test_size=float(ensemble_cfg.get("test_size", 0.2)),
        val_size=float(ensemble_cfg.get("val_size", 0.1)),
        random_state=int(ensemble_cfg.get("random_state", 42)),
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ensemble_artifacts = train_similarity_ensemble(splits.train_df, splits.val_df, ensemble_cfg)
    save_ensemble(ensemble_artifacts, out_dir / "ensemble")

    cross_metrics = None
    if not args.skip_cross_encoder:
        try:
            ce_artifacts = train_cross_encoder(
                train_df=splits.train_df,
                val_df=splits.val_df,
                cfg=cross_cfg,
                out_dir=out_dir / "cross_encoder",
            )
            cross_metrics = ce_artifacts.val_metrics
        except Exception as exc:
            print(f"Cross-encoder training skipped due to error: {exc}")

    selection = save_model_selection(
        out_dir=out_dir,
        ensemble_metrics=ensemble_artifacts.val_metrics,
        cross_encoder_metrics=cross_metrics,
    )

    with open(out_dir / "split_sizes.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "train": len(splits.train_df),
                "val": len(splits.val_df),
                "test": len(splits.test_df),
            },
            f,
            indent=2,
        )

    print("Training complete.")
    print("Ensemble metrics:", ensemble_artifacts.val_metrics)
    print("Cross-encoder metrics:", cross_metrics)
    print("Final selected model:", selection.final_model)
    print("Artifacts:", out_dir)


if __name__ == "__main__":
    main()
