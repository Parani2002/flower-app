from __future__ import annotations

from typing import Any

import numpy as np


class StackedFedAvg:
    def __init__(self, num_rounds: int = 5):
        self.num_rounds = num_rounds
        self.aggregated_weights: dict[str, Any] | None = None

    def aggregate_fit(self, client_updates: list[dict]) -> dict:
        if not client_updates:
            return {"lr": None, "rf": None, "xgb": None}

        aggregated = {
            "lr": np.mean([np.asarray(update["lr"], dtype=float) for update in client_updates], axis=0),
            "rf": b"".join(update["rf"] for update in client_updates),
            "xgb": b"".join(update["xgb"] for update in client_updates),
        }
        self.aggregated_weights = aggregated
        return aggregated

    def aggregate_evaluate(self, evaluations: list[dict]) -> float:
        if not evaluations:
            return 0.0
        aucs = [float(item.get("auc_roc", 0.0)) for item in evaluations]
        return float(np.mean(aucs))
