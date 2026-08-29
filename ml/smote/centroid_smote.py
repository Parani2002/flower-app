from __future__ import annotations

import numpy as np
from imblearn.over_sampling import SMOTE


class DPCentroidBiasSMOTE:
    def __init__(self, random_state: int = 42, alpha: float = 0.5):
        self.random_state = random_state
        self.alpha = alpha
        self.smote = SMOTE(random_state=random_state)

    def fit_resample(self, X: np.ndarray, y: np.ndarray, global_centroid: np.ndarray | None = None):
        X_res, y_res = self.smote.fit_resample(X, y)
        if global_centroid is None:
            return X_res, y_res

        global_centroid = np.asarray(global_centroid, dtype=float)
        if global_centroid.shape[0] != X_res.shape[1]:
            raise ValueError("Global centroid dimension does not match feature count.")

        rng = np.random.default_rng(self.random_state)
        shifted = X_res.copy()
        bias = self.alpha * (global_centroid - X_res.mean(axis=0))
        shifted = shifted + bias
        return shifted.astype(np.float64), y_res

    def apply(self, X: np.ndarray, y: np.ndarray, global_centroid: np.ndarray | None = None):
        return self.fit_resample(X, y, global_centroid)


def compute_global_centroid(client_shards: list[np.ndarray]) -> np.ndarray:
    stacked = np.vstack([np.asarray(shard, dtype=float) for shard in client_shards if len(shard) > 0])
    if stacked.size == 0:
        return np.zeros(1)
    return stacked.mean(axis=0)
