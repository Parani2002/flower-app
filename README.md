# Aegis Ward

One product for the final-year project: a hospital web app, a FastAPI backend, and the existing **Flower** federated stacked ensemble for diabetic **&lt;30-day** readmission.

```
frontend/                 React ward console (port 5173)
backend/                  FastAPI (port 8000)
quickstart-pytorch/        Flower clients, server, task.py, diabetic_data.csv
```

Patient rows are **partitioned** across three demo hospitals. Flower still trains **one** global meta-learner (`Net`). Base models (logistic regression, random forest, XGBoost) stay local to each client during federation. The API publishes a **serving pack** so the UI can score encounters.

## Demo logins

| Role | Username | Password |
| --- | --- | --- |
| St. Helen's Infirmary | `sthelens` | `ward-demo` |
| Riverside General | `riverside` | `ward-demo` |
| Oakridge Medical Centre | `oakridge` | `ward-demo` |
| Network coordinator | `admin` | `aegis-admin` |

## Run

Python **3.11**. This repo already has `flwr_env`.

```bash
source flwr_env/bin/activate
cd backend
uvicorn app.main:app --reload --port 8000
```

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173).

1. Sign in as a hospital to see **your** partition and score patients (after publish).
2. Sign in as **admin** → **Federation**: start `flwr run`, then **Publish serving model**.
3. First-time demo: **Publish** works even before Flower finishes; if `final_model.pt` exists from a Flower run, that federated meta-learner is used.

Flower simulation from the UI needs the `flwr` CLI on `PATH` (same environment as `pip install -e .` in `quickstart-pytorch`). A full federated run is slow (SMOTE + three models + DP-SGD × 3 clients × 5 rounds). Publish is the path that makes the ward board usable for a viva.

## What to say in the viva

- **Frontend:** hospital vs coordinator views; records never shown across hospitals.
- **Backend:** sessions, partitions, train job, stacked-ensemble inference.
- **Flower:** FedAvg on the meta-MLP only; Opacus DP-SGD on clients; UCI diabetic encounters.
