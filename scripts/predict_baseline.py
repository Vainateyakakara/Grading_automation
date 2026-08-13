from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from asag.inference import BaselinePredictor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--student", required=True)
    args = parser.parse_args()

    predictor = BaselinePredictor.load(args.model_dir)
    score = predictor.predict(args.reference, args.student)
    print(f"Predicted score: {score:.4f}")


if __name__ == "__main__":
    main()
