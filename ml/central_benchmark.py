from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from ml.config import OUTPUTS_DIR
from ml.data.preprocess import get_test_data, get_train_data

RANDOM_STATE = 42
N_SPLITS = 5
RESULTS_PATH = OUTPUTS_DIR / "central_benchmark.json"


def _models() -> dict[str, Any]:
    return {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, solver="liblinear", random_state=RANDOM_STATE),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "xgboost": XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=1,
            random_state=RANDOM_STATE,
            verbosity=0,
        ),
        "mlp": make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(64,),
                early_stopping=True,
                validation_fraction=0.1,
                max_iter=100,
                random_state=RANDOM_STATE,
            ),
        ),
    }


def _probabilities(model: Any, X: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict_proba(X)[:, 1], dtype=float)


def _metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "auroc": float(roc_auc_score(y_true, probabilities)),
        "auprc": float(average_precision_score(y_true, probabilities)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
    }


def _fit_with_optional_smote(model: Any, X: np.ndarray, y: np.ndarray, use_smote: bool) -> Any:
    if use_smote:
        X, y = SMOTE(random_state=RANDOM_STATE).fit_resample(X, y)
    fitted = clone(model)
    fitted.fit(X, y)
    return fitted


def _stacked_test_probabilities(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    use_smote: bool,
) -> np.ndarray:
    folds = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    model_templates = _models()
    oof = np.zeros((len(X_train), len(model_templates)), dtype=float)
    test_predictions: list[np.ndarray] = []

    for train_idx, fold_idx in folds.split(X_train, y_train):
        fold_test_predictions: list[np.ndarray] = []
        for column, model in enumerate(model_templates.values()):
            fitted = _fit_with_optional_smote(model, X_train[train_idx], y_train[train_idx], use_smote)
            oof[fold_idx, column] = _probabilities(fitted, X_train[fold_idx])
            fold_test_predictions.append(_probabilities(fitted, X_test))
        test_predictions.append(np.column_stack(fold_test_predictions))

    meta_model = LogisticRegression(max_iter=1000, solver="liblinear", random_state=RANDOM_STATE)
    meta_model.fit(oof, y_train)
    averaged_test_predictions = np.mean(np.stack(test_predictions), axis=0)
    return _probabilities(meta_model, averaged_test_predictions)


def _evaluate_condition(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    use_smote: bool,
) -> dict[str, Any]:
    models = {}
    for name, template in _models().items():
        fitted = _fit_with_optional_smote(template, X_train, y_train, use_smote)
        models[name] = _metrics(y_test, _probabilities(fitted, X_test))

    stack_probabilities = _stacked_test_probabilities(X_train, y_train, X_test, use_smote)
    models["stacked_ensemble"] = _metrics(y_test, stack_probabilities)
    best_single_name = max(models, key=lambda name: models[name]["auroc"] if name != "stacked_ensemble" else -1)
    return {
        "smote": use_smote,
        "models": models,
        "best_single_model": best_single_name,
        "best_single_auroc": models[best_single_name]["auroc"],
        "stacked_ensemble_auroc": models["stacked_ensemble"]["auroc"],
    }


def run_benchmark() -> dict[str, Any]:
    X_train_frame, y_train_series = get_train_data()
    X_test_frame, y_test_series = get_test_data()
    X_train = X_train_frame.to_numpy(dtype=np.float64)
    y_train = y_train_series.to_numpy(dtype=int)
    X_test = X_test_frame.to_numpy(dtype=np.float64)
    y_test = y_test_series.to_numpy(dtype=int)

    results = {
        "protocol": {
            "dataset": "UCI diabetic readmission, one row per patient",
            "train_rows": int(len(y_train)),
            "test_rows": int(len(y_test)),
            "positive_rate_train": float(y_train.mean()),
            "positive_rate_test": float(y_test.mean()),
            "random_state": RANDOM_STATE,
            "cv_folds": N_SPLITS,
            "smote_rule": "SMOTE is fit only on each fold's training rows; validation and test rows are never resampled.",
        },
        "conditions": [
            _evaluate_condition(X_train, y_train, X_test, y_test, use_smote=False),
            _evaluate_condition(X_train, y_train, X_test, y_test, use_smote=True),
        ],
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    output = run_benchmark()
    for condition in output["conditions"]:
        print(
            "smote={} best_single={} auroc={:.4f} stack_auroc={:.4f}".format(
                condition["smote"],
                condition["best_single_model"],
                condition["best_single_auroc"],
                condition["stacked_ensemble_auroc"],
            )
        )
    print(f"saved={Path(RESULTS_PATH)}")