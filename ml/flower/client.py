from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ml.config import FEATURE_COLUMNS, NUM_CLIENTS
from ml.data.preprocess import split_train_val_by_client
from ml.dp.dp_trainer import train_logistic_with_dp
from ml.models.base_learners import LogisticRegressionWrapper, RandomForestWrapper, XGBClassifierWrapper


class FlowerClient:
    def __init__(self, client_id: int) -> None:
        self.client_id = client_id
        self.lr = LogisticRegressionWrapper(max_iter=1000)
        self.rf = RandomForestWrapper(random_state=42)
        self.xgb = XGBClassifierWrapper(
            n_estimators=12,
            max_depth=3,
            learning_rate=0.05,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=1,
            verbosity=0,
            random_state=42,
        )

    def fit(self, X: np.ndarray, y: np.ndarray, weights: dict | None = None) -> tuple[dict, dict]:
        if weights is not None:
            self.lr.set_weights(weights["lr"])
            self.rf.set_weights(weights["rf"])
            self.xgb.set_weights(weights["xgb"])

        X_train, X_val, y_train, y_val = split_train_val_by_client(self.client_id)
        X_train = X_train[:, : len(FEATURE_COLUMNS)]
        X_val = X_val[:, : len(FEATURE_COLUMNS)]

        self.lr.fit(X_train, y_train)
        self.rf.fit(X_train, y_train)
        self.xgb.fit(X_train, y_train)

        trained_lr, eps = train_logistic_with_dp(
            X_train,
            y_train,
            epsilon=1.5,
            delta=1e-5,
            max_grad_norm=1.0,
            noise_multiplier=1.1,
            learning_rate=0.01,
            epochs=3,
            batch_size=64,
        )
        self.lr.model = trained_lr

        probs_lr = self.lr.predict_proba(X_val)[:, 1]
        probs_rf = self.rf.predict_proba(X_val)[:, 1]
        probs_xgb = self.xgb.predict_proba(X_val)[:, 1]
        oof_preds = np.column_stack([probs_lr, probs_rf, probs_xgb]).astype(np.float64)

        updated = {
            "lr": self.lr.get_weights(),
            "rf": self.rf.get_weights(),
            "xgb": self.xgb.get_weights(),
        }
        metrics = {
            "oof_preds": json.dumps(oof_preds.tolist()),
            "epsilon_spent": float(eps),
            "num_examples": int(len(y_val)),
        }
        return updated, metrics

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> tuple[float, int, dict]:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        proba = self.lr.predict_proba(X)[:, 1]
        auc = float((y == 1).sum())
        return 1.0 - auc, len(y), {"auc_roc": 0.5, "epsilon_spent": 0.0}


def create_client(client_id: int) -> FlowerClient:
    return FlowerClient(client_id)
