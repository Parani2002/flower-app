"""task.py — Diabetic Readmission Federated Stacked Ensemble"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "diabetic_data.csv"
ARTIFACTS_DIR = ROOT / "artifacts"

# ─────────────────────────────────────────
# 1. META-LEARNER NETWORK  ← replaces Net (CNN)
# ─────────────────────────────────────────
class Net(nn.Module):
    """Small MLP that takes 3 base-learner probabilities as input."""

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


# ─────────────────────────────────────────
# 2. LOAD & PARTITION
# ─────────────────────────────────────────
@lru_cache(maxsize=1)
def load_raw_dataframe() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@lru_cache(maxsize=1)
def _processed() -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Load CSV, one-hot encode, return X, y, and feature column names."""
    df = load_raw_dataframe().copy()
    df = df.drop(columns=["encounter_id", "patient_nbr"])
    y = (df["readmitted"] == "<30").astype(int).values.astype(np.int32)
    df = df.drop(columns=["readmitted"])
    df = pd.get_dummies(df)
    columns = tuple(df.columns)
    X = df.values.astype(np.float32)
    return X, y, columns


def _load_and_preprocess() -> tuple[np.ndarray, np.ndarray]:
    """Load CSV, encode, return X and y as numpy arrays."""
    X, y, _ = _processed()
    return X, y


def feature_columns() -> list[str]:
    return list(_processed()[2])


def raw_records_to_features(raw_df: pd.DataFrame) -> np.ndarray:
    """Encode original CSV-shaped rows to the same feature space as training."""
    df = raw_df.drop(columns=["encounter_id", "patient_nbr", "readmitted"], errors="ignore")
    df = pd.get_dummies(df)
    df = df.reindex(columns=feature_columns(), fill_value=0)
    return df.values.astype(np.float32)


def load_data(partition_id: int, num_partitions: int):
    """Return train/val split for this client's partition."""
    X, y = _load_and_preprocess()

    indices = np.array_split(np.arange(len(X)), num_partitions)
    idx = indices[partition_id]
    X_part, y_part = X[idx], y[idx]

    # 80/20 train-val split within the partition
    X_train, X_val, y_train, y_val = train_test_split(
        X_part, y_part, test_size=0.2, random_state=42, stratify=y_part
    )
    return X_train, X_val, y_train, y_val


def load_centralized_dataset():
    """Hold-out 10 % of full dataset for server-side global evaluation."""
    X, y = _load_and_preprocess()
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.10, random_state=0, stratify=y
    )
    return X_test, y_test


# ─────────────────────────────────────────
# 3. SMOTE
# ─────────────────────────────────────────
def apply_smote(X: np.ndarray, y: np.ndarray):
    """Apply SMOTE locally to balance classes."""
    smote = SMOTE(random_state=42)
    return smote.fit_resample(X, y)


# ─────────────────────────────────────────
# 4. BASE LEARNERS
# ─────────────────────────────────────────
def train_base_learners(X_train: np.ndarray, y_train: np.ndarray):
    """Train LR, RF, XGB on SMOTE-resampled data."""
    
    # Scale features — fixes LR convergence
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    lr  = LogisticRegression(max_iter=3000, solver="saga")  # saga handles large datasets better
    rf  = RandomForestClassifier(n_estimators=100, random_state=42)
    xgb = XGBClassifier(eval_metric="logloss", verbosity=0)

    lr.fit(X_scaled, y_train)   # ← scaled
    rf.fit(X_train, y_train)    # ← RF doesn't need scaling
    xgb.fit(X_train, y_train)   # ← XGB doesn't need scaling

    return lr, rf, xgb, scaler  # ← return scaler too


# ─────────────────────────────────────────
# 5. META-FEATURES
# ─────────────────────────────────────────
def generate_meta_features(X, lr, rf, xgb, scaler=None) -> np.ndarray:
    X_scaled = scaler.transform(X) if scaler is not None else X
    return np.column_stack([
        lr.predict_proba(X_scaled)[:, 1],  # ← scaled for LR
        rf.predict_proba(X)[:, 1],
        xgb.predict_proba(X)[:, 1],
    ]).astype(np.float32)


# ─────────────────────────────────────────
# 6. TRAIN META-LEARNER  (Opacus DP-SGD)
# ─────────────────────────────────────────
def train(
    net: Net,
    meta_X: np.ndarray,
    y: np.ndarray,
    epochs: int,
    lr: float,
    device: torch.device,
) -> float:
    """Train PyTorch meta-learner with Opacus DP-SGD (make_private — macOS safe)."""
    from opacus import PrivacyEngine

    net.train()
    X_t = torch.tensor(meta_X).to(device)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(device)

    dataset   = torch.utils.data.TensorDataset(X_t, y_t)
    loader    = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=True, drop_last=True  # drop_last required by Opacus
    )
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.BCELoss()

    # make_private instead of make_private_with_epsilon — avoids macOS hang
    privacy_engine = PrivacyEngine()
    net, optimizer, loader = privacy_engine.make_private(
        module=net,
        optimizer=optimizer,
        data_loader=loader,
        noise_multiplier=1.1,   # standard value, ε ≈ 1.0 for typical dataset sizes
        max_grad_norm=1.0,
    )

    total_loss = 0.0
    for _ in range(epochs):
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = criterion(net(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    # Log the actual epsilon spent
    epsilon = privacy_engine.get_epsilon(delta=1e-5)
    print(f"    DP guarantee: ε={epsilon:.2f}, δ=1e-5")

    return total_loss / (epochs * len(loader))


# ─────────────────────────────────────────
# 7. TEST META-LEARNER
# ─────────────────────────────────────────
def test(
    net: Net,
    meta_X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
) -> tuple[float, float, float]:
    """Return AUC, F1, recall on meta-features."""
    net.eval()
    with torch.no_grad():
        X_t   = torch.tensor(meta_X).to(device)
        probs = net(X_t).cpu().numpy().squeeze()

    preds = (probs >= 0.5).astype(int)
    auc    = roc_auc_score(y, probs)
    f1     = f1_score(y, preds, zero_division=0)
    recall = recall_score(y, preds, zero_division=0)
    return auc, f1, recall


# ─────────────────────────────────────────
# 8. SERVING BUNDLE (API / hospital product)
# ─────────────────────────────────────────
def predict_readmission(
    net: Net,
    X: np.ndarray,
    lr,
    rf,
    xgb,
    scaler,
    device: torch.device | None = None,
) -> np.ndarray:
    """Return P(readmitted < 30 days) for encoded feature rows."""
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    meta = generate_meta_features(X, lr, rf, xgb, scaler)
    net.eval()
    with torch.no_grad():
        probs = net(torch.tensor(meta).to(device)).cpu().numpy().squeeze()
    return np.atleast_1d(probs).astype(np.float32)