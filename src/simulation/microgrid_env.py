"""Environnement de simulation d'un micro-réseau (foyers + PV + batterie + réseau)."""
from dataclasses import dataclass, field

import numpy as np

from src import config


@dataclass
class StepResult:
    state: np.ndarray
    reward: float
    done: bool
    info: dict = field(default_factory=dict)


class MicrogridEnv:
    """Environnement pas-à-pas inspiré de l'interface Gym.

    À chaque pas, l'agent fixe un prix ; la demande réagit selon une élasticité
    prix, le solaire couvre d'abord la demande, la batterie absorbe le surplus ou
    couvre le déficit, et le reste est échangé avec le réseau principal.

    Récompense = revenus des ventes aux foyers + export - coût d'import.
    """

    def __init__(self, demand: np.ndarray, solar: np.ndarray, hours: np.ndarray | None = None,
                 params: dict = config.MICROGRID):
        if len(demand) != len(solar):
            raise ValueError("demand et solar doivent avoir la même longueur")
        self.base_demand = np.asarray(demand, dtype=float)
        self.solar = np.asarray(solar, dtype=float)
        self.hours = np.arange(len(demand)) % 24 if hours is None else np.asarray(hours)
        self.p = params
        self.reset()

    def reset(self) -> np.ndarray:
        self.t = 0
        self.soc = self.p["initial_soc"]
        self.history: list[dict] = []
        return self._state()

    def _state(self) -> np.ndarray:
        t = min(self.t, len(self.base_demand) - 1)
        h = self.hours[t]
        return np.array([self.base_demand[t], self.solar[t], self.soc,
                         np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)], dtype=np.float32)

    def step(self, price: float) -> StepResult:
        p = self.p
        demand = self.base_demand[self.t] * (price / p["reference_price"]) ** p["demand_elasticity"]
        net = demand - self.solar[self.t]   # >0 : déficit, <0 : surplus

        cap, pmax, eff = p["battery_capacity_kwh"], p["battery_power_kw"], p["battery_efficiency"]
        energy = self.soc * cap
        if net > 0:   # décharge
            discharge = min(net, pmax, energy * eff)
            energy -= discharge / eff
            grid = net - discharge
            battery_flow = -discharge
        else:         # charge
            charge = min(-net, pmax, (cap - energy) / eff)
            energy += charge * eff
            grid = net + charge          # négatif = export
            battery_flow = charge
        self.soc = energy / cap

        grid_import, grid_export = max(grid, 0.0), max(-grid, 0.0)
        revenue = demand * price
        cost = grid_import * p["grid_import_price"] - grid_export * p["grid_export_price"]
        reward = revenue - cost

        info = {"t": self.t, "price": price, "demand": demand, "solar": self.solar[self.t],
                "battery_flow": battery_flow, "soc": self.soc, "grid_import": grid_import,
                "grid_export": grid_export, "revenue": revenue, "cost": cost, "reward": reward}
        self.history.append(info)

        self.t += 1
        done = self.t >= len(self.base_demand)
        return StepResult(self._state(), reward, done, info)

    def run(self, policy) -> list[dict]:
        """Exécute un épisode complet ; ``policy(state) -> price``."""
        state, done = self.reset(), False
        while not done:
            res = self.step(policy(state))
            state, done = res.state, res.done
        return self.history


def fixed_price_policy(price: float = config.MICROGRID["reference_price"]):
    """Politique de référence : prix constant."""
    return lambda state: price
