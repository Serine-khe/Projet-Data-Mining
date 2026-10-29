"""Dev B (Hadjer) — LSTM pour la prévision de production photovoltaïque.

==============================================================================
PROMPT POUR HADJER
==============================================================================
Objectif : prévoir la production solaire de l'heure suivante (kW) à partir des
``seq_len`` dernières heures d'irradiance, de température et du calendrier.

À écrire dans ce fichier :
  1. ``SolarLSTM`` (nn.Module, PyTorch) : couche(s) nn.LSTM (batch_first=True)
     puis une couche linéaire qui sort une valeur. Entrée (batch, seq_len,
     n_features) -> sortie (batch,).
  2. ``SolarForecaster.fit`` :
       - ajouter les variables calendaires (src/data/features.add_time_features) ;
       - normaliser les entrées (min-max ou standardisation) et la cible, en
         calculant les statistiques UNIQUEMENT sur le train ;
       - découper en fenêtres glissantes (src/data/features.make_sequences) ;
       - boucle d'entraînement : DataLoader, Adam, MSELoss, nombre d'époques
         dans config.LSTM_SOLAR ; renvoyer la liste des pertes par époque.
  3. ``predict_with_target`` : mêmes transformations, prédiction en mode eval,
     dé-normalisation, valeurs négatives ramenées à 0.
  4. ``save`` / ``load`` : sauvegarder aussi les statistiques de normalisation.

Questions à traiter dans le notebook 04 et le rapport :
  - Influence de seq_len, hidden_size, num_layers, dropout.
  - Comparaison avec une référence simple (persistance : « même valeur qu'il y a 24 h »).
  - Comportement les jours nuageux vs ensoleillés ; la nuit (production nulle).

Interface à respecter (utilisée par src/pipeline.py et tests/test_models.py) :
  - SolarForecaster(params).fit(df, verbose) -> list[float] (perte par époque)
  - predict(df) -> np.ndarray, longueur len(df) - seq_len, valeurs >= 0
  - predict_with_target(df) -> (y_true, y_pred)
  - save(path) / load(path) -> mêmes prédictions après rechargement
Validation : ``pytest tests/test_models.py -k lstm`` doit passer.
==============================================================================
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from src import config


class SolarLSTM(nn.Module):
    def __init__(self, n_features: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        # TODO (Hadjer) : self.lstm = nn.LSTM(...) ; self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO (Hadjer) : passer x dans le LSTM, garder la dernière sortie, appliquer la tête
        raise NotImplementedError


class SolarForecaster:
    def __init__(self, params: dict = config.LSTM_SOLAR):
        self.params = params
        self.features = params["features"]
        self.target = params["target"]
        self.net = SolarLSTM(len(self.features), params["hidden_size"],
                             params["num_layers"], params["dropout"])
        # TODO (Hadjer) : attributs de normalisation (ex. self.x_min, self.x_max, self.y_max)

    def fit(self, df: pd.DataFrame, verbose: bool = True) -> list[float]:
        # TODO (Hadjer) : normalisation, séquences, boucle d'entraînement
        raise NotImplementedError

    @torch.no_grad()
    def predict_with_target(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        # TODO (Hadjer) : séquences, prédiction en mode eval, dé-normalisation
        raise NotImplementedError

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.predict_with_target(df)[1]

    def save(self, path: Path = config.MODELS_DIR / "lstm_solar.pt") -> None:
        # TODO (Hadjer) : torch.save des params, du state_dict et des stats de normalisation
        raise NotImplementedError

    @staticmethod
    def load(path: Path = config.MODELS_DIR / "lstm_solar.pt") -> "SolarForecaster":
        # TODO (Hadjer) : reconstruire le modèle à partir du fichier
        raise NotImplementedError
