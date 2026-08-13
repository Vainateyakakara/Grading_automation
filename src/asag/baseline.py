from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge


@dataclass
class BaselineModel:
    regressor: Ridge

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.regressor.predict(X)


def train_baseline(X_train: np.ndarray, y_train: np.ndarray, alpha: float) -> BaselineModel:
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    return BaselineModel(regressor=model)
