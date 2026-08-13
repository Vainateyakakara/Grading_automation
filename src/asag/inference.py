from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .features import transform_features
from .preprocess import normalize_text


@dataclass
class BaselinePredictor:
    regressor: object
    vectorizer: object
    min_score: float
    max_score: float

    @classmethod
    def load(cls, model_dir: str | Path) -> "BaselinePredictor":
        model_dir = Path(model_dir)
        regressor = joblib.load(model_dir / "ridge.joblib")
        vectorizer = joblib.load(model_dir / "tfidf.joblib")
        with open(model_dir / "score_range.json", "r", encoding="utf-8") as f:
            sr = json.load(f)
        return cls(
            regressor=regressor,
            vectorizer=vectorizer,
            min_score=float(sr["min_score"]),
            max_score=float(sr["max_score"]),
        )

    def predict(self, reference_answer: str, student_answer: str) -> float:
        df = pd.DataFrame(
            {
                "reference_answer": [normalize_text(reference_answer)],
                "student_answer": [normalize_text(student_answer)],
            }
        )
        X = transform_features(df, self.vectorizer)
        pred = float(self.regressor.predict(X)[0])
        return float(np.clip(pred, self.min_score, self.max_score))
