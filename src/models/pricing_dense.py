"""Dev C — Agent de tarification dynamique (réseau dense).

L'agent reçoit l'état du micro-réseau (consommation prévue, production solaire
prévue, état de charge de la batterie, heure) et propose un prix de vente
(€/kWh) borné dans [price_min, price_max].

Entraînement : apprentissage supervisé sur une politique cible qui fixe un prix
élevé quand le déficit net (demande - solaire) est fort et la batterie vide, et
un prix bas en cas de surplus. Cette cible peut être remplacée par une
politique optimisée dans l'environnement (``src/simulation``).
"""
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src import config

STATE_FEATURES = ["demand_kwh", "solar_kwh", "soc", "hour_sin", "hour_cos"]


class PricingNet(nn.Module):
    def __init__(self, n_inputs: int, hidden_sizes: list[int]):
        super().__init__()
        layers, prev = [], n_inputs
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def target_price(states: np.ndarray, params: dict = config.PRICING_DENSE) -> np.ndarray:
    """Politique cible heuristique, renvoie un prix dans [price_min, price_max]."""
    demand, solar, soc = states[:, 0], states[:, 1], states[:, 2]
    net = (demand - solar) / (np.abs(demand) + np.abs(solar) + 1e-6)  # dans [-1, 1]
    score = 0.5 + 0.35 * net + 0.15 * (0.5 - soc)
    score = np.clip(score, 0, 1)
    return params["price_min"] + score * (params["price_max"] - params["price_min"])


class PricingAgent:
    def __init__(self, params: dict = config.PRICING_DENSE):
        self.params = params
        self.net = PricingNet(len(STATE_FEATURES), params["hidden_sizes"])
        self.scale = np.ones(len(STATE_FEATURES), dtype=np.float32)

    def _normalize(self, states: np.ndarray) -> torch.Tensor:
        return torch.from_numpy((np.asarray(states, dtype=np.float32) / self.scale))

    def fit(self, states: np.ndarray, prices: np.ndarray | None = None, verbose: bool = True) -> list[float]:
        torch.manual_seed(config.RANDOM_SEED)
        states = np.asarray(states, dtype=np.float32)
        if prices is None:
            prices = target_price(states, self.params)
        self.scale = np.maximum(np.abs(states).max(axis=0), 1e-6).astype(np.float32)

        lo, hi = self.params["price_min"], self.params["price_max"]
        y = ((prices - lo) / (hi - lo)).astype(np.float32)
        loader = DataLoader(TensorDataset(self._normalize(states), torch.from_numpy(y)),
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
            if verbose and (epoch + 1) % 10 == 0:
                print(f"[Pricing] epoch {epoch + 1}/{self.params['epochs']} loss={history[-1]:.5f}")
        return history

    @torch.no_grad()
    def predict(self, states: np.ndarray) -> np.ndarray:
        self.net.eval()
        states = np.atleast_2d(states)
        out = self.net(self._normalize(states)).numpy()
        lo, hi = self.params["price_min"], self.params["price_max"]
        return lo + out * (hi - lo)

    def act(self, state: np.ndarray) -> float:
        """Prix pour un seul état (interface utilisée par la simulation)."""
        return float(self.predict(state)[0])

    def save(self, path: Path = config.MODELS_DIR / "pricing_dense.pt") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"params": self.params, "state_dict": self.net.state_dict(), "scale": self.scale}, path)

    @staticmethod
    def load(path: Path = config.MODELS_DIR / "pricing_dense.pt") -> "PricingAgent":
        ckpt = torch.load(path, weights_only=False)
        agent = PricingAgent(ckpt["params"])
        agent.net.load_state_dict(ckpt["state_dict"])
        agent.scale = ckpt["scale"]
        return agent
