from __future__ import annotations

from ml.config import NUM_CLIENTS, NUM_ROUNDS
from ml.flower.server import FlowerServer


def run_simulation() -> dict:
    server = FlowerServer(num_rounds=NUM_ROUNDS, num_clients=NUM_CLIENTS)
    return server.run()


if __name__ == "__main__":
    print(run_simulation())
