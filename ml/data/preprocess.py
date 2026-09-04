from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ml.config import (
    ALPHA,
    CLEAN_DATA_PATH,
    DATASET_PATH,
    FEATURE_COLUMNS,
    NUM_CLIENTS,
    SPLITS_DIR,
    STATISTICS_PATH,
    STATISTICS_TABLE_PATH,
)

DEATH_OR_HOSPICE_DISPOSITIONS = {11, 13, 14, 19, 20, 21}
DROP_COLUMNS = {"encounter_id", "patient_nbr", "weight", "payer_code", "medical_specialty"}
NUMERIC_COLUMNS = {
    "admission_type_id", "discharge_disposition_id", "admission_source_id",
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
}
ICD_GROUPS = {
    "circulatory": ((390, 459), (785, 785)),
    "respiratory": ((460, 519), (786, 786)),
    "digestive": ((520, 579), (787, 787)),
    "diabetes": ((250, 250),),
    "injury": ((800, 999),),
    "musculoskeletal": ((710, 739),),
    "genitourinary": ((580, 629), (788, 788)),
    "neoplasms": ((140, 239),),
}

_fitted_medians: dict[str, float] = {}
_fitted_categories: dict[str, list[str]] = {}
_scaler: StandardScaler | None = None


def _load_raw_dataframe() -> pd.DataFrame:
    return pd.read_csv(DATASET_PATH)


def _icd_group(value: object) -> str:
    if str(value).strip() in set(ICD_GROUPS) | {"other"}:
        return str(value).strip()
    try:
        code = float(str(value).strip())
    except (TypeError, ValueError):
        return "other"
    for group, ranges in ICD_GROUPS.items():
        if any(start <= code <= end for start, end in ranges):
            return group
    return "other"


def _clean_raw_dataframe() -> tuple[pd.DataFrame, dict[str, object]]:
    raw = _load_raw_dataframe()
    ordered = raw.sort_values("encounter_id", kind="mergesort")
    first = ordered.drop_duplicates("patient_nbr", keep="first").copy()
    first_count = len(first)
    dispositions = pd.to_numeric(first["discharge_disposition_id"], errors="coerce")
    first = first[~dispositions.isin(DEATH_OR_HOSPICE_DISPOSITIONS)].copy()
    for column in ("diag_1", "diag_2", "diag_3"):
        first[column] = first[column].map(_icd_group)
    first["readmitted"] = (first["readmitted"].astype(str) == "<30").astype(int)
    first = first.drop(columns={"weight", "payer_code", "medical_specialty"}, errors="ignore")
    missing = (raw.replace("?", pd.NA).isna().mean() * 100).round(2).to_dict()
    positive_rate = float(first["readmitted"].mean())
    stats = {
        "raw_rows": len(raw),
        "unique_patients_before_exclusions": first_count,
        "duplicate_rows_removed": len(raw) - raw["patient_nbr"].nunique(),
        "death_or_hospice_rows_removed": first_count - len(first),
        "clean_rows": len(first),
        "positive_rows": int(first["readmitted"].sum()),
        "positive_rate": round(positive_rate, 6),
        "baseline_auprc": round(positive_rate, 6),
        "missing_percent_raw": missing,
    }
    return first, stats


def _encode(features: pd.DataFrame) -> pd.DataFrame:
    numeric = list(_fitted_medians)
    result = pd.DataFrame(index=features.index)
    if numeric:
        values = pd.DataFrame({
            column: pd.to_numeric(features[column], errors="coerce").fillna(_fitted_medians[column])
            for column in numeric
        })
        result[numeric] = _scaler.transform(values) if _scaler is not None else values
    for column, categories in _fitted_categories.items():
        values = features[column].replace("?", pd.NA).fillna("missing").astype(str)
        for category in categories:
            result[f"{column}_{category}"] = (values == category).astype(float)
    return result.reindex(columns=FEATURE_COLUMNS, fill_value=0.0)


def _fit_transformer(train: pd.DataFrame) -> None:
    global _scaler
    features = train.drop(columns=DROP_COLUMNS | {"readmitted"}, errors="ignore")
    numeric = [column for column in features.columns if column in NUMERIC_COLUMNS]
    categorical = [column for column in features.columns if column not in numeric]
    _fitted_medians.clear()
    _fitted_categories.clear()
    for column in numeric:
        values = pd.to_numeric(features[column], errors="coerce")
        _fitted_medians[column] = float(values.median())
    for column in categorical:
        values = features[column].replace("?", pd.NA).fillna("missing").astype(str)
        _fitted_categories[column] = sorted(values.unique().tolist())
    numeric_values = pd.DataFrame({
        column: pd.to_numeric(features[column], errors="coerce").fillna(_fitted_medians[column])
        for column in numeric
    })
    _scaler = StandardScaler().fit(numeric_values)
    FEATURE_COLUMNS[:] = numeric + [
        f"{column}_{category}"
        for column, categories in _fitted_categories.items()
        for category in categories
    ]


def _prepare() -> pd.DataFrame:
    clean, stats = _clean_raw_dataframe()
    CLEAN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(CLEAN_DATA_PATH, index=False)
    train, remainder = train_test_split(clean, test_size=0.30, random_state=42, stratify=clean["readmitted"])
    validation, test = train_test_split(remainder, test_size=0.50, random_state=42, stratify=remainder["readmitted"])
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for name, frame in (("train", train), ("validation", validation), ("test", test)):
        frame.to_csv(SPLITS_DIR / f"{name}.csv", index=False)
        stats[f"{name}_rows"] = len(frame)
        stats[f"{name}_positive_rate"] = round(float(frame["readmitted"].mean()), 6)
    simple_stats = [{"statistic": key, "value": value} for key, value in stats.items() if not isinstance(value, dict)]
    pd.DataFrame(simple_stats).to_csv(STATISTICS_TABLE_PATH, index=False)
    STATISTICS_PATH.write_text(json.dumps(stats, indent=2))
    return clean


def preprocess_dataset() -> tuple[pd.DataFrame, pd.Series]:
    clean = _prepare()
    train = pd.read_csv(SPLITS_DIR / "train.csv")
    _fit_transformer(train)
    return _encode(clean.drop(columns="readmitted")), clean["readmitted"].astype(int)


def split_clients() -> list[pd.DataFrame]:
    preprocess_dataset()
    train = pd.read_csv(SPLITS_DIR / "train.csv")
    X = _encode(train.drop(columns="readmitted"))
    df = X.copy()
    df["__target__"] = train["readmitted"].astype(int).to_numpy()
    rng = np.random.default_rng(42)
    partitions: list[list[int]] = [[] for _ in range(NUM_CLIENTS)]
    for label in np.unique(df["__target__"]):
        candidate_idx = np.where(df["__target__"].to_numpy() == label)[0]
        permuted = rng.permutation(candidate_idx)
        proportions = rng.dirichlet(np.full(NUM_CLIENTS, ALPHA), size=1)[0]
        counts = np.floor(proportions * len(permuted)).astype(int)
        counts[-1] = len(permuted) - counts[:-1].sum()
        start = 0
        for client_id, count in enumerate(counts):
            end = start + count
            partitions[client_id].extend(permuted[start:end].tolist())
            start = end
    return [df.iloc[np.array(sorted(indices))].copy() for indices in partitions]


def split_train_val_by_client(client_id: int, client_shard: pd.DataFrame | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if client_shard is None:
        client_shard = split_clients()[client_id]
    X = client_shard.drop(columns=["__target__"]).copy()
    y = client_shard["__target__"].astype(int).to_numpy()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_train.to_numpy(dtype=np.float64), X_val.to_numpy(dtype=np.float64), y_train, y_val


def load_all_data() -> tuple[np.ndarray, np.ndarray]:
    X, y = preprocess_dataset()
    return X.to_numpy(dtype=np.float64), y.to_numpy(dtype=int)


def transform_raw_records(raw_df: pd.DataFrame) -> np.ndarray:
    preprocess_dataset()
    raw = raw_df.drop(columns=DROP_COLUMNS | {"readmitted"}, errors="ignore").copy()
    for column in ("diag_1", "diag_2", "diag_3"):
        if column in raw:
            raw[column] = raw[column].map(_icd_group)
    return _encode(raw).to_numpy(dtype=np.float32)
