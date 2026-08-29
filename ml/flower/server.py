from __future__ import annotations

from typing import Any

import numpy as np

from ml.data.preprocess import split_train_val_by_client
from ml.evaluation.metrics import compute_metrics, log_round_metrics
from ml.flower.client import FlowerClient
from ml.flower.strategy import StackedFedAvg


class FlowerServer:
    def __init__(self, num_rounds: int, num_clients: int) -> None:
        self.num_rounds = num_rounds
        self.num_clients = num_clients

    def build_strategy(self, **kwargs: Any):
        return StackedFedAvg(num_rounds=self.num_rounds)

    def run(self) -> dict[str, Any]:
        clients = [FlowerClient(client_id) for client_id in range(self.num_clients)]
        strategy = self.build_strategy()

        round_metrics: list[dict[str, Any]] = []
        for round_number in range(1, self.num_rounds + 1):
            client_updates: list[dict] = []
            client_epsilon: dict[str, float] = {}
            val_probs: list[np.ndarray] = []
            val_targets: list[np.ndarray] = []
            val_predictions: list[np.ndarray] = []

            for client_id, client in enumerate(clients):
                X_train, X_val, y_train, y_val = split_train_val_by_client(client_id)
                updated, metrics = client.fit(X_train, y_train)
                client_updates.append(updated)
                client_epsilon[f"client_{client_id}"] = float(metrics.get("epsilon_spent", 0.0))

                lr_prob = client.lr.predict_proba(X_val)[:, 1]
                rf_prob = client.rf.predict_proba(X_val)[:, 1]
                xgb_prob = client.xgb.predict_proba(X_val)[:, 1]
                ensemble_prob = (lr_prob + rf_prob + xgb_prob) / 3.0
                ensemble_pred = (ensemble_prob >= 0.5).astype(int)

                val_probs.append(ensemble_prob)
                val_targets.append(y_val)
                val_predictions.append(ensemble_pred)

            aggregate = strategy.aggregate_fit(client_updates)
            agg_prob = np.concatenate(val_probs)
            agg_y = np.concatenate(val_targets)
            agg_pred = np.concatenate(val_predictions)
            agg_auc = float(compute_metrics(agg_y, agg_pred, agg_prob)["auc_roc"])

            log_round_metrics(round_number, client_epsilon, agg_auc, agg_y, agg_pred, agg_prob)
            round_metrics.append({
                "round": round_number,
                "aggregate_auc_roc": agg_auc,
                "per_client_epsilon_spent": client_epsilon,
                "status": "ok",
                "aggregate_keys": sorted(aggregate.keys()),
            })

        return {
            "status": "ok",
            "num_rounds": self.num_rounds,
            "num_clients": self.num_clients,
            "rounds": round_metrics,
            "last_round": round_metrics[-1] if round_metrics else None,
        }
