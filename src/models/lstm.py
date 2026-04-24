"""LSTM PyTorch — modèle séquentiel pour prédiction BTC up/down.

Architecture : 2 couches LSTM + dropout + MLP tête classification + sigmoid.
Entrée : fenêtre glissante de N ticks × features.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.features import FEATURE_COLS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class LSTMClassifier(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, layers: int = 2, dropout: float = 0.25) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            n_features, hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


@dataclass
class LSTMModel:
    net: LSTMClassifier
    window: int
    feature_cols: list[str]

    def predict_proba_window(self, X_windows: np.ndarray) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            t = torch.from_numpy(X_windows).float().to(DEVICE)
            logits = self.net(t)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs


def make_windows(X: np.ndarray, y: np.ndarray, window: int = 30):
    """Crée des fenêtres glissantes (batch, window, features)."""
    Xs, ys = [], []
    for i in range(window, len(X)):
        Xs.append(X[i - window:i])
        ys.append(y[i])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def train_lstm(
    df_features,
    window: int = 30,
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 64,
) -> tuple[LSTMModel, dict]:
    from src.models.features import get_xy
    X, y = get_xy(df_features)
    X_arr = X.to_numpy(dtype=np.float32)
    y_arr = y.to_numpy(dtype=np.float32)

    if len(X_arr) < window + 50:
        raise ValueError(f"Pas assez de data pour LSTM ({len(X_arr)} < {window + 50})")

    Xw, yw = make_windows(X_arr, y_arr, window=window)
    split = int(len(Xw) * 0.8)
    X_tr, X_te = Xw[:split], Xw[split:]
    y_tr, y_te = yw[:split], yw[split:]

    train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    net = LSTMClassifier(n_features=X_arr.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss()

    for ep in range(epochs):
        net.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            logits = net(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total += float(loss) * len(xb)
        if (ep + 1) % 5 == 0:
            print(f"  [LSTM] epoch {ep+1}/{epochs} loss={total/len(train_ds):.4f}")

    # Eval
    net.eval()
    with torch.no_grad():
        t = torch.from_numpy(X_te).float().to(DEVICE)
        probs = torch.sigmoid(net(t)).cpu().numpy()
    preds = (probs > 0.5).astype(int)
    acc = float((preds == y_te).mean())

    model = LSTMModel(net=net, window=window, feature_cols=FEATURE_COLS)
    metrics = {"accuracy": acc, "n_train": len(X_tr), "n_test": len(X_te)}
    return model, metrics
