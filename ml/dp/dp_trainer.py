from __future__ import annotations

import numpy as np
import torch
from opacus import PrivacyEngine


def train_logistic_with_dp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    epsilon: float,
    delta: float,
    max_grad_norm: float,
    noise_multiplier: float,
    learning_rate: float = 0.01,
    epochs: int = 3,
    batch_size: int = 64,
):
    """Train a tiny PyTorch logistic model with DP-SGD.

    The function intentionally avoids make_private_with_epsilon because of the macOS
    hang issue described in the project requirements. We supply noise_multiplier
    directly and compute the actual privacy budget after training completes.
    """
    device = torch.device("cpu")
    X_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_train, dtype=torch.float32, device=device).reshape(-1, 1)

    model = torch.nn.Sequential(
        torch.nn.Linear(X_train.shape[1], 1),
        torch.nn.Sigmoid(),
    )
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = torch.nn.BCELoss()

    dataset = torch.utils.data.TensorDataset(X_t, y_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    privacy_engine = PrivacyEngine()
    model, optimizer, loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )

    for _ in range(epochs):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

    actual_epsilon = privacy_engine.get_epsilon(delta)
    return model, actual_epsilon
