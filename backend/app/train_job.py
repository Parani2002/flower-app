from __future__ import annotations

import subprocess
import threading
from datetime import datetime, timezone

from app.paths import ARTIFACTS, FL_ROOT, TRAIN_LOG


class TrainRunner:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.process: subprocess.Popen[str] | None = None
        self.started_at: str | None = None
        self.finished_at: str | None = None
        self.exit_code: int | None = None
        self.error: str | None = None

    def status(self) -> dict:
        with self._lock:
            running = self.process is not None and self.process.poll() is None
            if self.process is not None and not running and self.exit_code is None:
                self.exit_code = self.process.returncode
                self.finished_at = datetime.now(timezone.utc).isoformat()
            log_tail = ""
            if TRAIN_LOG.exists():
                text = TRAIN_LOG.read_text(errors="replace")
                log_tail = "\n".join(text.splitlines()[-40:])
            return {
                "running": running,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "exit_code": self.exit_code,
                "error": self.error,
                "log_tail": log_tail,
            }

    def start(self) -> dict:
        with self._lock:
            if self.process is not None and self.process.poll() is None:
                return {"ok": False, "detail": "A federated run is already in progress."}
            ARTIFACTS.mkdir(parents=True, exist_ok=True)
            log = TRAIN_LOG.open("w")
            try:
                self.process = subprocess.Popen(
                    ["flwr", "run", ".", "--stream"],
                    cwd=FL_ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            except FileNotFoundError:
                self.error = "Flower CLI (`flwr`) is not on PATH. Install the quickstart-pytorch package first."
                return {"ok": False, "detail": self.error}
            self.started_at = datetime.now(timezone.utc).isoformat()
            self.finished_at = None
            self.exit_code = None
            self.error = None
            return {"ok": True, "detail": "Federated simulation started.", "pid": self.process.pid}


runner = TrainRunner()
