"""Compatibility layer for the legacy Flower entry points.

The active implementation now lives under the ml package. This file keeps the old
API names working while delegating the actual logic to the new ML modules.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from ml.config import DATASET_PATH, FEATURE_COLUMNS
from ml.data.preprocess import load_all_data, preprocess_dataset, split_train_val_by_client
from ml.models.base_learners import train_base_learners

ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts"
DATA_PATH = DATASET_PATH


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def load_raw_dataframe() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def _processed():
    X, y = preprocess_dataset()
    return X.to_numpy(dtype=np.float32), y.to_numpy(dtype=np.int32), tuple(FEATURE_COLUMNS)


def feature_columns() -> list[str]:
    return list(FEATURE_COLUMNS)


def raw_records_to_features(raw_df: pd.DataFrame) -> np.ndarray:
    raw = raw_df.copy()
    raw = raw.drop(columns=["encounter_id", "patient_nbr", "readmitted"], errors="ignore")
    raw = pd.get_dummies(raw)
    for feature in FEATURE_COLUMNS:
        if feature not in raw.columns:
            raw[feature] = 0.0
    raw = raw.reindex(columns=FEATURE_COLUMNS, fill_value=0)
    return raw.to_numpy(dtype=np.float32)


def load_data(partition_id: int, num_partitions: int):
    return split_train_val_by_client(partition_id)


def load_centralized_dataset():
    X, y = load_all_data()
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.10, random_state=0, stratify=y)
    return X_test, y_test


def apply_smote(X: np.ndarray, y: np.ndarray):
    smote = SMOTE(random_state=42)
    return smote.fit_resample(X, y)


def generate_meta_features(X, lr, rf, xgb, scaler=None) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if scaler is not None:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X
    proba_lr = lr.predict_proba(X_scaled)[:, 1]
    proba_rf = rf.predict_proba(X)[:, 1]
    proba_xgb = xgb.predict_proba(X)[:, 1]
    return np.column_stack([proba_lr, proba_rf, proba_xgb]).astype(np.float32)


def train(net: Net, meta_X: np.ndarray, y: np.ndarray, epochs: int, lr: float, device: torch.device) -> float:
    net.train()
    X_t = torch.tensor(meta_X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.float32, device=device).reshape(-1, 1)
    dataset = torch.utils.data.TensorDataset(X_t, y_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.BCELoss()
    total_loss = 0.0
    for _ in range(epochs):
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = criterion(net(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    return total_loss / max(1, epochs * len(loader))


def test(net: Net, meta_X: np.ndarray, y: np.ndarray, device: torch.device) -> tuple[float, float, float]:
    net.eval()
    with torch.no_grad():
        probs = net(torch.tensor(meta_X, dtype=torch.float32, device=device)).cpu().numpy().squeeze()
    preds = (probs >= 0.5).astype(int)
    auc = roc_auc_score(y, probs)
    f1 = f1_score(y, preds, zero_division=0)
    recall = recall_score(y, preds, zero_division=0)
    return auc, f1, recall


def predict_readmission(net: Net, X: np.ndarray, lr, rf, xgb, scaler, device: torch.device | None = None) -> np.ndarray:
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    meta = generate_meta_features(X, lr, rf, xgb, scaler)
    net.eval()
    with torch.no_grad():
        probs = net(torch.tensor(meta, dtype=torch.float32, device=device)).cpu().numpy().squeeze()
    return np.atleast_1d(probs).astype(np.float32)


__all__ = [
    "Net",
    "DATA_PATH",
    "ARTIFACTS_DIR",
    "load_raw_dataframe",
    "feature_columns",
    "raw_records_to_features",
    "load_data",
    "load_centralized_dataset",
    "apply_smote",
    "train_base_learners",
    "generate_meta_features",
    "train",
    "test",
    "predict_readmission",
]
