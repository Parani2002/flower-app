from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FL_ROOT = REPO_ROOT / "quickstart-pytorch"
ARTIFACTS = FL_ROOT / "pytorchexample" / "artifacts"
METRICS_PATH = ARTIFACTS / "metrics.json"
BUNDLE_PATH = ARTIFACTS / "serving_bundle.joblib"
NET_PATH = ARTIFACTS / "final_model.pt"
TRAIN_LOG = ARTIFACTS / "train.log"
