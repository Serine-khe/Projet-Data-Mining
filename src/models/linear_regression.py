"""Dev A — Régression linéaire pour la prévision de consommation."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.data.features import add_lag_features, add_time_features

TARGET = "consumption_kwh"
TIME_FEATURES = ["hour_sin", "hour_cos", "doy_sin", "doy_cos", "dayofweek", "is_weekend"]


def build_features(df: pd.DataFrame, params: dict = config.LINEAR_REGRESSION) -> pd.DataFrame:
    df = add_time_features(df)
    return add_lag_features(df, TARGET, params["lags"], params["rolling_windows"])


def feature_columns(df: pd.DataFrame) -> list[str]:
    lag_cols = [c for c in df.columns if c.startswith(f"{TARGET}_")]
    return TIME_FEATURES + lag_cols


class ConsumptionForecaster:
    """Régression linéaire (Ridge si ``alpha > 0``) sur variables temporelles et retards."""

    def __init__(self, params: dict = config.LINEAR_REGRESSION):
        self.params = params
        reg = Ridge(alpha=params["alpha"]) if params["alpha"] > 0 else LinearRegression()
        self.model = Pipeline([("scaler", StandardScaler()), ("reg", reg)])
        self.features: list[str] = []

    def fit(self, df: pd.DataFrame) -> "ConsumptionForecaster":
        feats = build_features(df, self.params)
        self.features = feature_columns(feats)
        self.model.fit(feats[self.features], feats[TARGET])
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        feats = build_features(df, self.params)
        return self.model.predict(feats[self.features])

    def predict_with_target(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        feats = build_features(df, self.params)
        return feats[TARGET].to_numpy(), self.model.predict(feats[self.features])

    def save(self, path: Path = config.MODELS_DIR / "linear_regression.pkl") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path = config.MODELS_DIR / "linear_regression.pkl") -> "ConsumptionForecaster":
        return joblib.load(path)
