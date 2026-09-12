"""Train / load the serving stack used by the product API."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from app.paths import ARTIFACTS, BUNDLE_PATH, FL_ROOT, METRICS_PATH, NET_PATH

if str(FL_ROOT) not in sys.path:
    sys.path.insert(0, str(FL_ROOT))

from pytorchexample.task import (  # noqa: E402
    Net,
    apply_smote,
    generate_meta_features,
    load_raw_dataframe,
    predict_readmission,
    raw_records_to_features,
    test,
    train_base_learners,
)
from ml.config import CLEAN_DATA_PATH
from ml.data.preprocess import get_train_data


def _load_net(path: Path) -> Net:
    net = Net()
    if path.exists():
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(path, map_location="cpu")
        net.load_state_dict(state)
    return net


def build_serving_bundle(max_rows: int = 8000) -> dict:
    """Fit local base learners + meta-MLP for API inference.

    If Flower has already saved ``final_model.pt``, those federated weights are
    used for the meta-learner. Base learners stay hospital-side in real FL; here
    we fit one serving copy so the product can score patients in a demo.
    """
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    started = time.time()

    get_train_data()
    raw = pd.read_csv(CLEAN_DATA_PATH)
    y_all = raw["readmitted"].astype(int).values
    rng = np.random.RandomState(7)
    if len(raw) > max_rows:
        pos = np.where(y_all == 1)[0]
        neg = np.where(y_all == 0)[0]
        n_pos = min(len(pos), max_rows // 3)
        n_neg = max_rows - n_pos
        idx = np.concatenate(
            [rng.choice(pos, n_pos, replace=False), rng.choice(neg, n_neg, replace=False)]
        )
        rng.shuffle(idx)
        sample = raw.iloc[idx]
    else:
        sample = raw

    y = sample["readmitted"].astype(int).values
    X = raw_records_to_features(sample)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_bal, y_bal = apply_smote(X_train, y_train)
    lr, rf, xgb, scaler = train_base_learners(X_bal, y_bal)

    net = _load_net(NET_PATH)
    device = torch.device("cpu")
    net.to(device)
    used_federated = NET_PATH.exists()

    if not used_federated:
        meta_train = generate_meta_features(X_bal, lr, rf, xgb, scaler)
        net.train()
        X_t = torch.tensor(meta_train)
        y_t = torch.tensor(y_bal, dtype=torch.float32).unsqueeze(1)
        opt = torch.optim.Adam(net.parameters(), lr=0.001)
        loss_fn = torch.nn.BCELoss()
        for _ in range(8):
            opt.zero_grad()
            loss = loss_fn(net(X_t), y_t)
            loss.backward()
            opt.step()
        torch.save(net.state_dict(), NET_PATH)

    auc, f1, recall = test(net, generate_meta_features(X_val, lr, rf, xgb, scaler), y_val, device)

    bundle = {
        "lr": lr,
        "rf": rf,
        "xgb": xgb,
        "scaler": scaler,
        "used_federated_meta": used_federated,
    }
    joblib.dump(bundle, BUNDLE_PATH)

    metrics = {
        "auc": float(auc),
        "f1": float(f1),
        "recall": float(recall),
        "used_federated_meta": used_federated,
        "train_rows": int(len(X_train)),
        "patients_in_network": int(len(raw)),
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_sec": round(time.time() - started, 1),
        "dp_note": "Client training uses Opacus DP-SGD (ε logged per round). Serving uses the aggregated meta-learner.",
        "rounds": 5,
        "hospitals": 3,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    return metrics


def load_bundle() -> dict:
    if not BUNDLE_PATH.exists():
        raise FileNotFoundError("Serving model is not published yet.")
    return joblib.load(BUNDLE_PATH)


def score_features(X: np.ndarray) -> np.ndarray:
    bundle = load_bundle()
    net = _load_net(NET_PATH)
    net.to(torch.device("cpu"))
    return predict_readmission(net, X, bundle["lr"], bundle["rf"], bundle["xgb"], bundle["scaler"])
