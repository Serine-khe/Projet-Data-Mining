"""Dev C (Asma) — Agent de tarification dynamique (réseau de neurones dense).

==============================================================================
PROMPT POUR ASMA
==============================================================================
Objectif : à chaque heure, fixer un prix de vente (€/kWh) qui équilibre l'offre
solaire et la demande des foyers : prix élevé en cas de déficit (inciter à
consommer moins), prix bas en cas de surplus (inciter à consommer le solaire).

Entrée de l'agent (état) : STATE_FEATURES =
  [demande prévue (Serine), production solaire prévue (Hadjer),
   état de charge de la batterie, sin(heure), cos(heure)]
Sortie : un prix dans [price_min, price_max] (config.PRICING_DENSE).

À écrire dans ce fichier :
  1. ``PricingNet`` (nn.Module) : couches Linear + ReLU (tailles dans
     config.PRICING_DENSE["hidden_sizes"]), sortie Sigmoid pour rester dans [0, 1],
     ensuite ramenée à [price_min, price_max].
  2. ``target_price`` : DÉFINIR comment le prix « idéal » est obtenu pour
     l'entraînement. C'est le cœur de ta contribution, à justifier dans le rapport.
     Pistes :
       a) règle experte (prix fonction du déficit net et de la batterie) ;
       b) recherche du meilleur prix par état dans la simulation (grille de prix,
          on garde celui qui maximise la récompense) ;
       c) optimisation directe du réseau sur la récompense de l'environnement.
  3. ``PricingAgent.fit`` : normaliser les états, entraîner le réseau (Adam, MSE),
     renvoyer la perte par époque.
  4. ``predict`` (lot d'états) et ``act`` (un état -> float).

Questions à traiter dans le notebook 05 et le rapport :
  - Comparaison avec un prix fixe : profit, import réseau, autoconsommation.
  - Le prix réagit-il correctement au déficit / surplus et à la batterie ?
  - Sensibilité à l'élasticité de la demande (config.MICROGRID).

Interface à respecter (utilisée par src/pipeline.py, la simulation et les tests) :
  - PricingAgent(params).fit(states, prices=None, verbose) -> list[float]
  - predict(states) -> np.ndarray, valeurs dans [price_min, price_max]
  - act(state) -> float
  - save(path) / load(path)
Validation : ``pytest tests/test_models.py -k pricing`` doit passer (prix plus
élevé en déficit qu'en surplus, bornes respectées).
==============================================================================
"""
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src import config

STATE_FEATURES = ["demand_kwh", "solar_kwh", "soc", "hour_sin", "hour_cos"]


class PricingNet(nn.Module):
    def __init__(self, n_inputs: int, hidden_sizes: list[int]):
        super().__init__()
        # TODO (Asma) : self.net = nn.Sequential(Linear, ReLU, ..., Linear(.., 1), Sigmoid)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def target_price(states: np.ndarray, params: dict = config.PRICING_DENSE) -> np.ndarray:
    """Prix cible pour chaque état, dans [price_min, price_max]."""
    # TODO (Asma) : définir la politique cible (voir pistes a, b, c ci-dessus)
    raise NotImplementedError


class PricingAgent:
    def __init__(self, params: dict = config.PRICING_DENSE):
        self.params = params
        self.net = PricingNet(len(STATE_FEATURES), params["hidden_sizes"])
        # TODO (Asma) : attributs de normalisation des états

    def fit(self, states: np.ndarray, prices: np.ndarray | None = None, verbose: bool = True) -> list[float]:
        # TODO (Asma) : si prices est None, utiliser target_price(states) ; entraîner le réseau
        raise NotImplementedError

    @torch.no_grad()
    def predict(self, states: np.ndarray) -> np.ndarray:
        # TODO (Asma) : normaliser, passer dans le réseau, ramener à [price_min, price_max]
        raise NotImplementedError

    def act(self, state: np.ndarray) -> float:
        """Prix pour un seul état (interface utilisée par la simulation)."""
        return float(self.predict(np.atleast_2d(state))[0])

    def save(self, path: Path = config.MODELS_DIR / "pricing_dense.pt") -> None:
        # TODO (Asma) : torch.save des params, du state_dict et des stats de normalisation
        raise NotImplementedError

    @staticmethod
    def load(path: Path = config.MODELS_DIR / "pricing_dense.pt") -> "PricingAgent":
        # TODO (Asma) : reconstruire l'agent à partir du fichier
        raise NotImplementedError
