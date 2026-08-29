from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from ml.config import FEATURE_COLUMNS


def _assert_feature_order(df: pd.DataFrame) -> pd.DataFrame:
    if list(df.columns) != FEATURE_COLUMNS:
        missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
        extra = [c for c in df.columns if c not in FEATURE_COLUMNS]
        raise ValueError(
            f"Feature order mismatch for SHAP. Expected {FEATURE_COLUMNS[:5]}... got {list(df.columns)[:5]}... "
            f"Missing={missing[:5]}, Extra={extra[:5]}"
        )
    return df[FEATURE_COLUMNS]


def explain_model(model, X: pd.DataFrame | np.ndarray, model_name: str):
    frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=FEATURE_COLUMNS)
    frame = _assert_feature_order(frame)
    X_array = frame.to_numpy(dtype=np.float64)

    if model_name in {"lr", "meta"}:
        explainer = shap.LinearExplainer(model.model if hasattr(model, "model") else model, X_array)
    else:
        explainer = shap.TreeExplainer(model.model if hasattr(model, "model") else model)

    values = explainer.shap_values(X_array)
    if isinstance(values, list):
        serializable = [np.asarray(v).tolist() for v in values]
    else:
        serializable = np.asarray(values).tolist()

    return {"feature_names": FEATURE_COLUMNS, "values": serializable}


def explain_models(models: dict, X: pd.DataFrame | np.ndarray) -> dict:
    result = {}
    for key, model in models.items():
        result[key] = explain_model(model, X, key)["values"]
    result["feature_names"] = FEATURE_COLUMNS
    return result
