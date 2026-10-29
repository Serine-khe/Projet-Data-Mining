"""Environnement de simulation d'un micro-réseau (foyers + PV + batterie + réseau).

==============================================================================
PROMPT POUR L'ÉQUIPE (fichier partagé, une méthode par personne)
==============================================================================
Objectif : simuler heure par heure l'équilibre offre / demande du micro-réseau,
pour évaluer l'agent de tarification d'Asma.

Déroulement d'un pas ``step(price)`` (déjà écrit, à ne pas changer sans accord) :
  1. demande réelle = _demand_response(demande de base, prix)      -> Serine
  2. déficit net = demande - solaire
  3. flux batterie, import, export = _dispatch_battery(déficit net) -> Hadjer
  4. récompense = _reward(demande, prix, import, export)           -> Asma

Serine — ``_demand_response`` : comment les foyers réagissent au prix.
  Piste : élasticité constante, D = D_base * (prix / prix_ref) ** élasticité
  (paramètres dans config.MICROGRID). Justifier la valeur de l'élasticité.

Hadjer — ``_dispatch_battery`` : le solaire couvre d'abord la demande ; la
  batterie se décharge en cas de déficit et se charge en cas de surplus, dans la
  limite de sa puissance (kW), de sa capacité (kWh) et de son rendement. Le reste
  est importé du / exporté vers le réseau principal. Mettre à jour self.soc
  (toujours entre 0 et 1).

Asma — ``_reward`` : ce que l'agent cherche à maximiser. Piste : revenus des
  ventes aux foyers + export * prix_export - import * prix_import. On peut
  ajouter une pénalité sur l'import pour favoriser l'autoconsommation.

Validation : ``pytest tests/test_simulation.py`` (bilan énergétique respecté,
SoC dans [0, 1], la demande baisse quand le prix monte).
==============================================================================
"""
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
    """Environnement pas-à-pas inspiré de l'interface Gym."""

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

    def _demand_response(self, base_demand: float, price: float) -> float:
        """Demande effective des foyers au prix donné (kWh)."""
        # TODO (Serine) : modèle de réaction de la demande au prix
        raise NotImplementedError

    def _dispatch_battery(self, net: float) -> tuple[float, float, float]:
        """Répartit le déficit (net > 0) ou le surplus (net < 0) entre batterie et réseau.

        Met à jour ``self.soc`` et renvoie ``(battery_flow, grid_import, grid_export)``
        avec battery_flow > 0 en charge, < 0 en décharge, import et export >= 0.
        Bilan à respecter : net = grid_import - grid_export - battery_flow.
        """
        # TODO (Hadjer) : limites de puissance, de capacité et rendement
        raise NotImplementedError

    def _reward(self, demand: float, price: float, grid_import: float, grid_export: float) -> tuple[float, float, float]:
        """Renvoie ``(reward, revenue, cost)`` pour le pas courant (€)."""
        # TODO (Asma) : revenus des ventes, coût d'import, gain d'export
        raise NotImplementedError

    def step(self, price: float) -> StepResult:
        demand = self._demand_response(self.base_demand[self.t], price)
        net = demand - self.solar[self.t]   # >0 : déficit, <0 : surplus
        battery_flow, grid_import, grid_export = self._dispatch_battery(net)
        reward, revenue, cost = self._reward(demand, price, grid_import, grid_export)

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
