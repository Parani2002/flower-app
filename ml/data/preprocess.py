from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.config import ALPHA, DATASET_PATH, FEATURE_COLUMNS, NUM_CLIENTS


def _load_raw_dataframe() -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH)
    df = df.drop(columns=["encounter_id", "patient_nbr"], errors="ignore")
    df["readmitted"] = df["readmitted"].apply(lambda value: 1 if str(value) == "<30" else 0)
    return df


def preprocess_dataset() -> tuple[pd.DataFrame, pd.Series]:
    df = _load_raw_dataframe().copy()
    df = df.replace("?", pd.NA)

    for column in df.columns:
        if column == "readmitted":
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            median = df[column].median()
            df[column] = df[column].fillna(median)
        else:
            mode = df[column].mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else "missing"
            df[column] = df[column].fillna(fill_value)

    categorical_columns = [
        column for column in df.columns if column != "readmitted" and not pd.api.types.is_numeric_dtype(df[column])
    ]
    encoded = pd.get_dummies(df, columns=categorical_columns, dtype=float)

    for feature in FEATURE_COLUMNS:
        if feature not in encoded.columns:
            encoded[feature] = 0.0
    encoded = encoded[FEATURE_COLUMNS]

    y = df["readmitted"].astype(int)
    return encoded, y


def split_clients() -> list[pd.DataFrame]:
    X, y = preprocess_dataset()
    df = X.copy()
    df["__target__"] = y.to_numpy()

    rng = np.random.default_rng(42)
    class_labels = np.unique(df["__target__"].to_numpy())
    partitions: list[list[int]] = [[] for _ in range(NUM_CLIENTS)]

    for label in class_labels:
        candidate_idx = np.where(df["__target__"].to_numpy() == label)[0]
        permuted = rng.permutation(candidate_idx)
        class_proportions = rng.dirichlet(np.full(NUM_CLIENTS, ALPHA), size=1)[0]
        counts = np.floor(class_proportions * len(permuted)).astype(int)
        counts[-1] = len(permuted) - counts[:-1].sum()

        start = 0
        for client_id, count in enumerate(counts):
            end = start + count
            partitions[client_id].extend(permuted[start:end].tolist())
            start = end

    shards = [df.iloc[np.array(sorted(partitions[client_id]))].copy() for client_id in range(NUM_CLIENTS)]
    return shards


def split_train_val_by_client(client_id: int, client_shard: pd.DataFrame | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if client_shard is None:
        client_shard = split_clients()[client_id]

    X = client_shard.drop(columns=["__target__"]).copy()
    y = client_shard["__target__"].astype(int).to_numpy()
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    return X_train.to_numpy(dtype=np.float64), X_val.to_numpy(dtype=np.float64), y_train.astype(int), y_val.astype(int)


def load_all_data() -> tuple[np.ndarray, np.ndarray]:
    X, y = preprocess_dataset()
    return X.to_numpy(dtype=np.float64), y.to_numpy(dtype=int)
