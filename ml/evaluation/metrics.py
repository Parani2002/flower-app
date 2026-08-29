from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from ml.config import OUTPUTS_DIR

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_PATH = OUTPUTS_DIR / "metrics.json"


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> dict:
    if y_prob is None:
        y_prob = y_pred.astype(float)
    return {
        "auc_roc": float(roc_auc_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def log_round_metrics(round_number: int, client_epsilon: dict[str, float], aggregate_auc: float, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> None:
    metrics = compute_metrics(y_true, y_pred, y_prob)
    payload = {
        "round": round_number,
        "per_client_epsilon_spent": client_epsilon,
        "aggregated_auc_roc": float(aggregate_auc),
        "f1": metrics["f1"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "confusion_matrix": metrics["confusion_matrix"],
    }

    existing = []
    if METRICS_PATH.exists():
        try:
            existing = json.loads(METRICS_PATH.read_text())
        except json.JSONDecodeError:
            existing = []
    if not isinstance(existing, list):
        existing = [existing]
    existing.append(payload)
    METRICS_PATH.write_text(json.dumps(existing, indent=2))


def save_model_weights(weights: dict, round_number: int) -> Path:
    path = OUTPUTS_DIR / "model_weights" / f"weights_round_{round_number}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(weights, indent=2, default=lambda obj: obj.tolist() if hasattr(obj, "tolist") else str(obj)))
    return path
