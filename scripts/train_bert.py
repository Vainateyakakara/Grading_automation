from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from asag.bert import train_bert_regressor
from asag.config import load_yaml
from asag.data import load_and_standardize, split_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Input CSV path")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--out", required=True, help="Output model directory")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    df = load_and_standardize(args.data)

    splits = split_data(
        df,
        test_size=float(cfg["test_size"]),
        val_size=float(cfg["val_size"]),
        random_state=int(cfg["random_state"]),
    )

    artifacts = train_bert_regressor(
        train_df=splits.train_df,
        val_df=splits.val_df,
        cfg=cfg,
        out_dir=args.out,
    )

    print(f"BERT model saved to: {artifacts.model_dir}")


if __name__ == "__main__":
    main()
