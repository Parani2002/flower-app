"""server.py — Flower ServerApp for diabetic readmission FYP."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from pytorchexample.task import (
    Net,
    generate_meta_features,
    load_centralized_dataset,
    test,
    train_base_learners,
    apply_smote,
)

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:

    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int          = context.run_config["num-server-rounds"]
    lr: float                = context.run_config["learning-rate"]

    # Initialise global meta-learner
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    strategy = FedAvg(fraction_evaluate=fraction_evaluate)

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    if context.run_config["save-model"]:
        from pytorchexample.task import ARTIFACTS_DIR

        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        path = ARTIFACTS_DIR / "final_model.pt"
        print(f"\nSaving final model to {path}...")
        torch.save(result.arrays.to_torch_state_dict(), path)


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate global meta-learner on central held-out set."""

    # Load global model
    net = Net()
    net.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    # Central test set — need base learners trained on it to get meta-features
    X_test, y_test = load_centralized_dataset()
    X_bal, y_bal   = apply_smote(X_test, y_test)
    lr_m, rf_m, xgb_m, scaler = train_base_learners(X_bal, y_bal)
    meta_test = generate_meta_features(X_test, lr_m, rf_m, xgb_m, scaler)

    auc, f1, recall = test(net, meta_test, y_test, device)
    print(f"[Round {server_round}] Global AUC: {auc:.4f}  F1: {f1:.4f}  Recall: {recall:.4f}")

    return MetricRecord({"auc": auc, "f1": f1, "recall": recall})