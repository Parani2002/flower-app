"""client.py — Flower ClientApp for diabetic readmission FYP."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from pytorchexample.task import (
    Net,
    apply_smote,
    generate_meta_features,
    load_data,
    test,
    train,
    train_base_learners,
)

app = ClientApp()


@app.train()
def train_client(msg: Message, context: Context):
    """Full local pipeline: load → SMOTE → base learners → meta-features → train MLP."""

    # ── Load partition ──────────────────────────────────────────────
    partition_id   = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    X_train, X_val, y_train, y_val = load_data(partition_id, num_partitions)

    # ── SMOTE on training split only ────────────────────────────────
    X_train_bal, y_train_bal = apply_smote(X_train, y_train)

    # ── Base learners ────────────────────────────────────────────────
# In train_client()
    lr_model, rf_model, xgb_model, scaler = train_base_learners(X_train_bal, y_train_bal)

    meta_train = generate_meta_features(X_train_bal, lr_model, rf_model, xgb_model, scaler)
    meta_val   = generate_meta_features(X_val,       lr_model, rf_model, xgb_model, scaler)

    # ── Load global meta-learner weights from server ─────────────────
    net = Net()
    net.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    # ── Train meta-learner with DP-SGD ───────────────────────────────
    epochs = context.run_config["local-epochs"]
    lr     = msg.content["config"]["lr"]
    train_loss = train(net, meta_train, y_train_bal, epochs, lr, device)

    # ── Quick local eval on val set ──────────────────────────────────
    auc, f1, recall = test(net, meta_val, y_val, device)

    # ── Return updated weights + metrics ─────────────────────────────
    content = RecordDict({
        "arrays":  ArrayRecord(net.state_dict()),
        "metrics": MetricRecord({
            "train_loss":   train_loss,
            "val_auc":      auc,
            "val_f1":       f1,
            "val_recall":   recall,
            "num-examples": len(y_train_bal),
        }),
    })
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate_client(msg: Message, context: Context):
    """Evaluate global model on this client's local val set."""

    partition_id   = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    _, X_val, _, y_val = load_data(partition_id, num_partitions)

    net = Net()
    net.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    # Need base learners to generate meta-features for val set
    # Re-train base learners on this client's training data
    X_train, _, y_train, _ = load_data(partition_id, num_partitions)
    X_train_bal, y_train_bal = apply_smote(X_train, y_train)
    # In evaluate_client()
    lr_model, rf_model, xgb_model, scaler = train_base_learners(X_train_bal, y_train_bal)
    meta_val = generate_meta_features(X_val, lr_model, rf_model, xgb_model, scaler)

    auc, f1, recall = test(net, meta_val, y_val, device)

    content = RecordDict({
        "metrics": MetricRecord({
            "eval_auc":     auc,
            "eval_f1":      f1,
            "eval_recall":  recall,
            "num-examples": len(y_val),
        }),
    })
    return Message(content=content, reply_to=msg)