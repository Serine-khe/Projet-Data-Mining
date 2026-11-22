"""
linear_regression.py
====================
Prévision de la consommation électrique par foyer (Dev A).

Modèles : Régression Linéaire, Ridge, Lasso (scikit-learn).
Interface commune : fit() / train(), predict(), evaluate(), save(), load().

Exemple :
    from src.models.linear_regression import (build_features, split_chronological,
                                              ConsumptionModel)
    df = pd.read_csv("data/processed/consumption_hourly.csv", index_col=0, parse_dates=True)
    X, y = build_features(df)
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = split_chronological(X, y)
    model = ConsumptionModel("ridge", alpha=1.0).fit(X_tr, y_tr)
    print(model.evaluate(X_te, y_te))
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET_COL = "mean_kwh"
WEATHER_COLS = ["temperature", "humidity", "windSpeed"]
DEFAULT_LAGS = (1, 2, 24, 48, 168)   # heures


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------
def build_features(df: pd.DataFrame, lags=DEFAULT_LAGS, target_col: str = TARGET_COL,
                   use_weather: bool = True, use_lags: bool = True):
    """Construit la matrice de features X et la cible y.

    Le DataFrame doit avoir un DatetimeIndex horaire.
    - lags : consommation passée (par défaut 1 h, 2 h, 24 h, 48 h, 168 h)
      -> avec lag_1 on fait de la prévision à 1 heure.
      -> pour prévoir plus loin, n'utiliser que des lags >= horizon.
    """
    X = pd.DataFrame(index=df.index)
    idx = df.index

    # Calendrier (encodage cyclique)
    X["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    X["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    X["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
    X["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)
    X["month_sin"] = np.sin(2 * np.pi * idx.month / 12)
    X["month_cos"] = np.cos(2 * np.pi * idx.month / 12)
    X["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    if "is_holiday" in df.columns:
        X["is_holiday"] = df["is_holiday"].astype(int)

    # Météo
    if use_weather:
        for c in WEATHER_COLS:
            if c in df.columns:
                X[c] = df[c]

    # Retards de la consommation
    if use_lags:
        for l in lags:
            X[f"lag_{l}"] = df[target_col].shift(l)
        X["roll_mean_24"] = df[target_col].shift(1).rolling(24).mean()

    y = df[target_col]
    mask = X.notna().all(axis=1)
    return X[mask], y[mask]


def split_chronological(X, y, train_size=0.70, val_size=0.15):
    """Découpage temporel (sans mélange). Retourne (train), (val), (test)."""
    n = len(X)
    i1, i2 = int(n * train_size), int(n * (train_size + val_size))
    return (X.iloc[:i1], y.iloc[:i1]), (X.iloc[i1:i2], y.iloc[i1:i2]), (X.iloc[i2:], y.iloc[i2:])


def compute_metrics(y_true, y_pred) -> dict:
    """MAE, RMSE, R² et MAPE (%)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-6, None))) * 100)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE_%": mape,
    }


def baseline_metrics(X, y) -> dict:
    """Baselines naïves : persistance (lag_1), saisonnière 24 h et 168 h (si présentes)."""
    out = {}
    for name, col in [("Persistance (t-1)", "lag_1"),
                      ("Saisonnière (t-24)", "lag_24"),
                      ("Saisonnière (t-168)", "lag_168")]:
        if col in X.columns:
            out[name] = compute_metrics(y, X[col])
    return out


# --------------------------------------------------------------------------
# Modèle
# --------------------------------------------------------------------------
class ConsumptionModel:
    """Régression (linear / ridge / lasso) avec standardisation des features."""

    def __init__(self, kind: str = "linear", alpha: float = 1.0):
        self.kind = kind
        self.alpha = alpha
        self.pipeline = Pipeline([("scaler", StandardScaler()), ("reg", self._make_reg())])
        self.feature_names = None

    def _make_reg(self):
        if self.kind == "linear":
            return LinearRegression()
        if self.kind == "ridge":
            return Ridge(alpha=self.alpha)
        if self.kind == "lasso":
            return Lasso(alpha=self.alpha, max_iter=10000)
        raise ValueError("kind doit être 'linear', 'ridge' ou 'lasso'")

    def fit(self, X, y):
        self.feature_names = list(X.columns)
        self.pipeline.fit(X, y)
        return self

    train = fit  # alias pour l'interface commune

    def predict(self, X) -> np.ndarray:
        """Prévisions en kWh par foyer et par heure (jamais négatives)."""
        return np.clip(self.pipeline.predict(X[self.feature_names]), 0, None)

    def evaluate(self, X, y) -> dict:
        return compute_metrics(y, self.predict(X))

    def coefficients(self) -> pd.Series:
        """Coefficients (sur features standardisées) triés par importance."""
        coefs = pd.Series(self.pipeline.named_steps["reg"].coef_, index=self.feature_names)
        return coefs.reindex(coefs.abs().sort_values(ascending=False).index)

    def save(self, directory="models_saved", name="linear_regression"):
        os.makedirs(directory, exist_ok=True)
        joblib.dump({"kind": self.kind, "alpha": self.alpha,
                     "pipeline": self.pipeline, "features": self.feature_names},
                    os.path.join(directory, f"{name}.pkl"))

    @classmethod
    def load(cls, directory="models_saved", name="linear_regression"):
        d = joblib.load(os.path.join(directory, f"{name}.pkl"))
        obj = cls(d["kind"], d["alpha"])
        obj.pipeline, obj.feature_names = d["pipeline"], d["features"]
        return obj


def time_series_cv(X, y, kind="ridge", alpha=1.0, n_splits=5) -> pd.DataFrame:
    """Validation croisée temporelle (TimeSeriesSplit) : une ligne de métriques par pli."""
    from sklearn.model_selection import TimeSeriesSplit

    rows = []
    for k, (tr, te) in enumerate(TimeSeriesSplit(n_splits=n_splits).split(X), 1):
        m = ConsumptionModel(kind, alpha).fit(X.iloc[tr], y.iloc[tr])
        rows.append({"fold": k, **m.evaluate(X.iloc[te], y.iloc[te])})
    return pd.DataFrame(rows).set_index("fold")
