from __future__ import annotations

from pathlib import Path

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
# Populated by ml.data.preprocess after fitting its train-only encoder.
FEATURE_COLUMNS: list[str] = []
