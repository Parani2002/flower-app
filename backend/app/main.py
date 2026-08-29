from __future__ import annotations

import json
import sys
from typing import Annotated, Any

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.auth import HOSPITALS, USERS, hospital_by_id, issue_token, parse_token
from app.paths import FL_ROOT, METRICS_PATH
from app.serving import build_serving_bundle, score_features
from app.train_job import runner

if str(FL_ROOT) not in sys.path:
    sys.path.insert(0, str(FL_ROOT))

from pytorchexample.task import load_raw_dataframe, raw_records_to_features  # noqa: E402

app = FastAPI(title="Aegis Ward", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LoginBody(BaseModel):
    username: str
    password: str


class PredictBody(BaseModel):
    encounter_id: int | None = None
    fields: dict[str, Any] | None = None


class PublishBody(BaseModel):
    max_rows: int = Field(default=8000, ge=2000, le=40000)


def current_user(authorization: Annotated[str | None, Header()] = None) -> dict:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Sign in required.")
    user = parse_token(authorization.split(" ", 1)[1])
    if not user:
        raise HTTPException(status_code=401, detail="Session expired. Sign in again.")
    return user


def _metrics() -> dict[str, Any] | None:
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text())
    return None


def _partition_frame(hospital_id: int, num_partitions: int = 3) -> pd.DataFrame:
    raw = load_raw_dataframe()
    idx = np.array_split(np.arange(len(raw)), num_partitions)[hospital_id]
    return raw.iloc[idx]


def _patient_card(row: pd.Series) -> dict[str, Any]:
    readmitted = str(row["readmitted"]) if "readmitted" in row.index else "unknown"
    return {
        "encounter_id": int(row.get("encounter_id", 0)),
        "age": str(row.get("age", "")),
        "gender": str(row.get("gender", "")),
        "race": str(row.get("race", "")),
        "time_in_hospital": int(row.get("time_in_hospital", 0) or 0),
        "num_medications": int(row.get("num_medications", 0) or 0),
        "number_inpatient": int(row.get("number_inpatient", 0) or 0),
        "number_emergency": int(row.get("number_emergency", 0) or 0),
        "A1Cresult": str(row.get("A1Cresult", "")),
        "insulin": str(row.get("insulin", "")),
        "diabetesMed": str(row.get("diabetesMed", "")),
        "actual_early_readmit": readmitted == "<30",
        "readmitted_label": readmitted,
    }


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "product": "Aegis Ward"}


@app.post("/api/login")
def login(body: LoginBody) -> dict:
    user = USERS.get(body.username)
    if not user or user["password"] != body.password:
        raise HTTPException(status_code=401, detail="Unknown hospital credentials.")
    token = issue_token(body.username)
    profile = {k: v for k, v in user.items() if k != "password"}
    hospital = hospital_by_id(profile["hospital_id"]) if profile["hospital_id"] is not None else None
    return {"token": token, "user": {**profile, "username": body.username}, "hospital": hospital}


@app.get("/api/me")
def me(user: dict = Depends(current_user)) -> dict:
    hospital = hospital_by_id(user["hospital_id"]) if user["hospital_id"] is not None else None
    return {"user": user, "hospital": hospital}


@app.get("/api/overview")
def overview(user: dict = Depends(current_user)) -> dict:
    raw = load_raw_dataframe()
    early = int((raw["readmitted"] == "<30").sum())
    stats = {
        "hospitals": HOSPITALS,
        "patients_in_network": int(len(raw)),
        "early_readmits": early,
        "early_rate": round(early / len(raw), 4),
        "metrics": _metrics(),
        "training": runner.status(),
        "serving_ready": METRICS_PATH.exists(),
    }
    if user["role"] == "hospital":
        part = _partition_frame(user["hospital_id"])
        stats["local"] = {
            "patients": int(len(part)),
            "early_readmits": int((part["readmitted"] == "<30").sum()),
            "hospital": hospital_by_id(user["hospital_id"]),
        }
    return stats


@app.get("/api/patients")
def patients(limit: int = 12, user: dict = Depends(current_user)) -> dict:
    hid = 0 if user["role"] == "admin" else user["hospital_id"]
    part = _partition_frame(hid)
    sample = part.sample(n=min(limit, len(part)), random_state=21)
    return {
        "hospital": hospital_by_id(hid),
        "patients": [_patient_card(row) for _, row in sample.iterrows()],
    }


@app.post("/api/predict")
def predict(body: PredictBody, user: dict = Depends(current_user)) -> dict:
    raw = load_raw_dataframe()
    if body.encounter_id is not None:
        match = raw[raw["encounter_id"] == body.encounter_id]
        if match.empty:
            raise HTTPException(status_code=404, detail="Encounter not found.")
        row = match.iloc[0]
        hid = user["hospital_id"]
        if user["role"] == "hospital":
            part = _partition_frame(hid)
            if int(row["encounter_id"]) not in set(part["encounter_id"].astype(int)):
                raise HTTPException(status_code=403, detail="This encounter is not on your ward.")
    elif body.fields:
        row = pd.Series(body.fields)
        if "encounter_id" not in row:
            row["encounter_id"] = 0
        if "patient_nbr" not in row:
            row["patient_nbr"] = 0
    else:
        raise HTTPException(status_code=400, detail="Provide encounter_id or fields.")

    try:
        X = raw_records_to_features(pd.DataFrame([row]))
        prob = float(score_features(X)[0])
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    risk = "high" if prob >= 0.5 else "moderate" if prob >= 0.25 else "lower"
    return {
        "probability": round(prob, 4),
        "risk": risk,
        "patient": _patient_card(row) if "readmitted" in row.index else None,
        "explanation": (
            "Stacked ensemble: local LR / Random Forest / XGBoost probabilities "
            "are combined by the federated meta-learner. Raw notes never leave the hospital in a live federation."
        ),
    }


@app.post("/api/training/start")
def start_training(user: dict = Depends(current_user)) -> dict:
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Only the network coordinator can start Flower.")
    result = runner.start()
    if not result["ok"]:
        raise HTTPException(status_code=409, detail=result["detail"])
    return result


@app.get("/api/training/status")
def training_status(user: dict = Depends(current_user)) -> dict:
    return runner.status()


@app.post("/api/serving/publish")
def publish(body: PublishBody, user: dict = Depends(current_user)) -> dict:
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Only the network coordinator can publish the model.")
    return build_serving_bundle(max_rows=body.max_rows)
