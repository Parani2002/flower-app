"""Compatibility wrapper for the legacy Flower client entry point.

This now delegates to the ML package under ml/ so the project no longer depends on
old demo-only logic in quickstart-pytorch.
"""

from __future__ import annotations

from ml.flower.client import FlowerClient

try:
    import torch  # noqa: F401
    from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
    from flwr.clientapp import ClientApp
except ImportError:  # pragma: no cover
    ClientApp = None
    ArrayRecord = Context = Message = MetricRecord = RecordDict = None

app = ClientApp() if ClientApp is not None else None


if app is not None:
    @app.train()
    def train_client(msg: Message, context: Context):
        client = FlowerClient(context.node_config.get("partition-id", 0))
        return client.fit(None, None, None)[1]

    @app.evaluate()
    def evaluate_client(msg: Message, context: Context):
        client = FlowerClient(context.node_config.get("partition-id", 0))
        return client.evaluate(None, None)


__all__ = ["app", "FlowerClient"]
