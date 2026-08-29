from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


class StackedMetaLearner:
    def __init__(self, random_state: int = 42):
        self.model = LogisticRegression(random_state=random_state, max_iter=2000)

    def fit(self, X_meta: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X_meta, y)

    def predict_proba(self, X_meta: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X_meta)

    def get_weights(self) -> np.ndarray:
        return np.concatenate([self.model.coef_.ravel(), self.model.intercept_])

    def set_weights(self, weights: np.ndarray) -> None:
        coef_size = self.model.coef_.size
        self.model.coef_ = weights[:coef_size].reshape(self.model.coef_.shape)
        self.model.intercept_ = weights[coef_size:coef_size + self.model.intercept_.size]

    def get_model(self) -> LogisticRegression:
        return self.model
