"""Dev B — LSTM pour la prévision de production solaire."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src import config
from src.data.features import add_time_features, make_sequences


class SolarLSTM(nn.Module):
    def __init__(self, n_features: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)


class SolarForecaster:
    """Entraînement / prédiction du LSTM avec normalisation min-max interne."""

    def __init__(self, params: dict = config.LSTM_SOLAR):
        self.params = params
        self.features = params["features"]
        self.target = params["target"]
        self.net = SolarLSTM(len(self.features), params["hidden_size"],
                             params["num_layers"], params["dropout"])
        self.x_min = self.x_max = None
        self.y_max = 1.0

    def _prepare(self, df: pd.DataFrame):
        df = add_time_features(df)
        X = df[self.features].to_numpy(dtype=np.float32)
        y = df[self.target].to_numpy(dtype=np.float32)
        X = (X - self.x_min) / (self.x_max - self.x_min + 1e-8)
        return make_sequences(X, y / self.y_max, self.params["seq_len"])

    def fit(self, df: pd.DataFrame, verbose: bool = True) -> list[float]:
        torch.manual_seed(config.RANDOM_SEED)
        X_raw = add_time_features(df)[self.features].to_numpy(dtype=np.float32)
        self.x_min, self.x_max = X_raw.min(axis=0), X_raw.max(axis=0)
        self.y_max = float(df[self.target].max()) or 1.0

        X, y = self._prepare(df)
        loader = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
                            batch_size=self.params["batch_size"], shuffle=True)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.params["lr"])
        loss_fn = nn.MSELoss()

        history = []
        self.net.train()
        for epoch in range(self.params["epochs"]):
            total = 0.0
            for xb, yb in loader:
                opt.zero_grad()
                loss = loss_fn(self.net(xb), yb)
                loss.backward()
                opt.step()
                total += loss.item() * len(xb)
            history.append(total / len(loader.dataset))
            if verbose:
                print(f"[LSTM] epoch {epoch + 1}/{self.params['epochs']} loss={history[-1]:.5f}")
        return history

    @torch.no_grad()
    def predict_with_target(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X, y = self._prepare(df)
        self.net.eval()
        pred = self.net(torch.from_numpy(X)).numpy().clip(min=0)
        return y * self.y_max, pred * self.y_max

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.predict_with_target(df)[1]

    def save(self, path: Path = config.MODELS_DIR / "lstm_solar.pt") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"params": self.params, "state_dict": self.net.state_dict(),
                    "x_min": self.x_min, "x_max": self.x_max, "y_max": self.y_max}, path)

    @staticmethod
    def load(path: Path = config.MODELS_DIR / "lstm_solar.pt") -> "SolarForecaster":
        ckpt = torch.load(path, weights_only=False)
        model = SolarForecaster(ckpt["params"])
        model.net.load_state_dict(ckpt["state_dict"])
        model.x_min, model.x_max, model.y_max = ckpt["x_min"], ckpt["x_max"], ckpt["y_max"]
        return model
