"""Dev A (Serine) — Régression linéaire pour la prévision de consommation de base.

==============================================================================
PROMPT POUR SERINE
==============================================================================
Objectif : prévoir la consommation horaire agrégée (kWh) des foyers londoniens
à partir du calendrier et de l'historique récent.

À écrire dans ce fichier :
  1. ``build_features`` : à partir d'un DataFrame (``timestamp``, ``consumption_kwh``),
     construire les variables explicatives :
       - variables calendaires (heure, jour de semaine, week-end, saison),
         de préférence en encodage cyclique (sin/cos) ;
       - retards de la consommation (ex. t-1, t-24, t-168) et moyennes glissantes.
     Utiliser ``src/data/features.py`` (add_time_features, add_lag_features).
     Attention à la fuite de données : aucune variable ne doit utiliser la valeur
     de l'heure à prédire.
  2. ``ConsumptionForecaster.fit`` : entraîner une régression linéaire
     (LinearRegression ou Ridge de scikit-learn, avec standardisation).
  3. ``predict`` / ``predict_with_target`` : renvoyer les prévisions (et les
     vraies valeurs alignées, car les retards suppriment les premières lignes).

Questions à traiter dans le notebook 03 et le rapport :
  - Quels retards apportent le plus ? (coefficients, sélection de variables)
  - OLS vs Ridge : la régularisation aide-t-elle ?
  - Erreurs par heure / jour de semaine : où le modèle se trompe-t-il ?

Interface à respecter (utilisée par src/pipeline.py et tests/test_models.py) :
  - ConsumptionForecaster(params).fit(df) -> self
  - predict(df) -> np.ndarray
  - predict_with_target(df) -> (y_true, y_pred)
  - save(path) / load(path)
Validation : ``pytest tests/test_models.py -k linear`` doit passer (R² > 0.7).
==============================================================================
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import config

TARGET = "consumption_kwh"


def build_features(df: pd.DataFrame, params: dict = config.LINEAR_REGRESSION) -> pd.DataFrame:
    """Retourne ``df`` enrichi des variables explicatives (lignes incomplètes supprimées)."""
    # TODO (Serine) : variables calendaires + retards + moyennes glissantes
    raise NotImplementedError


class ConsumptionForecaster:
    def __init__(self, params: dict = config.LINEAR_REGRESSION):
        self.params = params
        self.model = None                 # TODO (Serine) : Pipeline(StandardScaler, Ridge/LinearRegression)
        self.features: list[str] = []     # noms des colonnes utilisées

    def fit(self, df: pd.DataFrame) -> "ConsumptionForecaster":
        # TODO (Serine) : build_features, choisir les colonnes, entraîner self.model
        raise NotImplementedError

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.predict_with_target(df)[1]

    def predict_with_target(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        # TODO (Serine) : renvoyer (vraies valeurs, prévisions) alignées
        raise NotImplementedError

    def save(self, path: Path = config.MODELS_DIR / "linear_regression.pkl") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path = config.MODELS_DIR / "linear_regression.pkl") -> "ConsumptionForecaster":
        return joblib.load(path)
