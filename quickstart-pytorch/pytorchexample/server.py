"""Compatibility wrapper for the legacy Flower server entry point.

The operational implementation now lives under ml/flower/server.py.
"""

from __future__ import annotations

from ml.flower.server import FlowerServer

try:
    from flwr.serverapp import ServerApp
except ImportError:  # pragma: no cover
    ServerApp = None

app = ServerApp() if ServerApp is not None else None


if app is not None:
    @app.main()
    def main(grid, context):
        server = FlowerServer(num_rounds=context.run_config.get("num-server-rounds", 5), num_clients=3)
        return server.run()


__all__ = ["app", "FlowerServer"]
