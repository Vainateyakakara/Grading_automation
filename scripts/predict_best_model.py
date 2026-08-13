from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from asag.best_model import BestModelPredictor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, help="Best model artifact dir")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--student", required=True)
    args = parser.parse_args()

    predictor = BestModelPredictor(args.model_dir)
    out = predictor.predict(args.reference, args.student)
    print(out)


if __name__ == "__main__":
    main()
