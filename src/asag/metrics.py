from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import mean_squared_error


def qwk_numpy(y_true, y_pred, min_rating: int | None = None, max_rating: int | None = None) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.size == 0:
        return 0.0

    if min_rating is None:
        min_rating = int(np.floor(np.min(y_true)))
    if max_rating is None:
        max_rating = int(np.ceil(np.max(y_true)))

    y_true_i = np.clip(np.rint(y_true), min_rating, max_rating).astype(int)
    y_pred_i = np.clip(np.rint(y_pred), min_rating, max_rating).astype(int)

    n = len(y_true_i)
    k = max_rating - min_rating + 1
    if k <= 1:
        return 1.0

    conf = np.zeros((k, k), dtype=float)
    for a, p in zip(y_true_i, y_pred_i):
        conf[a - min_rating, p - min_rating] += 1.0

    hist_true = conf.sum(axis=1)
    hist_pred = conf.sum(axis=0)
    expected = np.outer(hist_true, hist_pred) / max(n, 1)

    w = np.zeros((k, k), dtype=float)
    denom = float((k - 1) ** 2)
    for i in range(k):
        for j in range(k):
            w[i, j] = ((i - j) ** 2) / denom

    o = conf / max(n, 1)
    e = expected / max(n, 1)
    num = float((w * o).sum())
    den = float((w * e).sum())
    return 1.0 - (num / den) if den != 0 else 1.0


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray, min_score: float, max_score: float) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    qwk = qwk_numpy(y_true, y_pred, min_rating=int(min_score), max_rating=int(max_score))
    return {"mse": float(mse), "qwk": float(qwk)}
