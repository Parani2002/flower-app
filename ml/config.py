from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPO_ROOT / "quickstart-pytorch" / "pytorchexample" / "data" / "diabetic_data.csv"
ML_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = ML_ROOT / "outputs"
MODEL_WEIGHTS_DIR = OUTPUTS_DIR / "model_weights"
CLEAN_DATA_PATH = OUTPUTS_DIR / "cleaned_diabetic_data.csv"
SPLITS_DIR = OUTPUTS_DIR / "splits"
STATISTICS_PATH = OUTPUTS_DIR / "dataset_statistics.json"
STATISTICS_TABLE_PATH = OUTPUTS_DIR / "dataset_statistics.csv"

NUM_CLIENTS = 3
NUM_ROUNDS = 5
EPSILON = 1.5
DELTA = 1e-5
MAX_GRAD_NORM = 1.0
NOISE_MULTIPLIER = 1.1
BATCH_SIZE = 64
LOCAL_EPOCHS = 3
ALPHA = 0.5


def _build_feature_columns() -> list[str]:
    """Return the deterministic feature-order list used throughout training and SHAP."""
    df = pd.read_csv(DATASET_PATH)
    df = df.drop(columns=["encounter_id", "patient_nbr", "readmitted"], errors="ignore")
    df = df.replace("?", pd.NA)

    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            df[column] = df[column].apply(lambda value: float(value) if pd.notna(value) else float("nan"))
            median = df[column].median()
            df[column] = df[column].fillna(median)
        else:
            mode_value = df[column].mode(dropna=True)
            fill_value = mode_value.iloc[0] if not mode_value.empty else "missing"
            df[column] = df[column].fillna(fill_value)

    categorical_columns = [
        column for column in df.columns if not pd.api.types.is_numeric_dtype(df[column])
    ]
    encoded = pd.get_dummies(df, columns=categorical_columns, dtype=float)
    return list(encoded.columns)


# Populated by ml.data.preprocess after fitting its train-only encoder.
FEATURE_COLUMNS: list[str] = []
