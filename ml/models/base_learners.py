from __future__ import annotations

import os
import pickle
import tempfile
from io import BytesIO
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


class LogisticRegressionWrapper:
    def __init__(self, **kwargs: Any) -> None:
        self.model = LogisticRegression(**kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def get_weights(self) -> np.ndarray:
        if not hasattr(self.model, "coef_"):
            raise ValueError("Model has not been fitted yet.")
        return np.concatenate([self.model.coef_.ravel(), self.model.intercept_])

    def set_weights(self, weights: np.ndarray) -> None:
        n_features = self.model.n_features_in_ if hasattr(self.model, "n_features_in_") else weights.shape[0] - 1
        if self.model.coef_.size == 0 if hasattr(self.model, "coef_") else True:
            self.model.coef_ = np.zeros((1, n_features), dtype=np.float64)
            self.model.intercept_ = np.zeros(1, dtype=np.float64)
            self.model.classes_ = np.array([0, 1])
        coef_size = self.model.coef_.size
        self.model.coef_ = weights[:coef_size].reshape(self.model.coef_.shape)
        self.model.intercept_ = weights[coef_size:coef_size + self.model.intercept_.size]

    def get_model(self) -> LogisticRegression:
        return self.model


class RandomForestWrapper:
    def __init__(self, **kwargs: Any) -> None:
        self.model = RandomForestClassifier(**kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def get_weights(self) -> bytes:
        return pickle.dumps(self.model)

    def set_weights(self, weights: bytes) -> None:
        self.model = pickle.loads(weights)

    def get_model(self) -> RandomForestClassifier:
        return self.model


class XGBClassifierWrapper:
    def __init__(self, **kwargs: Any) -> None:
        self.model = XGBClassifier(**kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def get_weights(self) -> bytes:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xgbmodel") as handle:
            temp_path = handle.name
        try:
            self.model.get_booster().save_model(temp_path)
            with open(temp_path, "rb") as handle:
                return handle.read()
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def set_weights(self, weights: bytes) -> None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xgbmodel") as handle:
            temp_path = handle.name
            handle.write(weights)
        try:
            self.model.load_model(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def get_model(self) -> XGBClassifier:
        return self.model


def train_base_learners(X_train: np.ndarray, y_train: np.ndarray) -> tuple[LogisticRegressionWrapper, RandomForestWrapper, XGBClassifierWrapper]:
    lr = LogisticRegressionWrapper(max_iter=2000, solver="liblinear", random_state=42)
    rf = RandomForestWrapper(n_estimators=200, random_state=42)
    xgb = XGBClassifierWrapper(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )

    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    xgb.fit(X_train, y_train)
    return lr, rf, xgb
