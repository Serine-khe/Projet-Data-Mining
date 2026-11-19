import json
import os
import random

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

SEED = 42
TARGET_COL = "Radiation"
BASE_FEATURES = ["Radiation", "Temperature", "Humidity", "Pressure", "Speed"]
TIME_FEATURES = ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]
FEATURE_COLS = BASE_FEATURES + TIME_FEATURES


def set_seed(seed: int = SEED) -> None:
    """Fixe les graines aléatoires pour la reproductibilité."""
    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# --------------------------------------------------------------------------
# Fonctions utilitaires
# --------------------------------------------------------------------------
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute l'heure et le jour de l'année sous forme cyclique (sin/cos).
    Le DataFrame doit avoir un DatetimeIndex."""
    df = df.copy()
    hour = df.index.hour
    doy = df.index.dayofyear
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return df


def split_chronological(df: pd.DataFrame, train_size: float = 0.70, val_size: float = 0.15):
    """Découpe temporelle (sans mélange) en train / validation / test."""
    n = len(df)
    i_train = int(n * train_size)
    i_val = int(n * (train_size + val_size))
    return df.iloc[:i_train], df.iloc[i_train:i_val], df.iloc[i_val:]


def make_windows(features: np.ndarray, target: np.ndarray, window: int, horizon: int):
    """Construit les fenêtres glissantes.
    X : (n, window, n_features)   y : (n, horizon)"""
    X, y = [], []
    for i in range(len(features) - window - horizon + 1):
        X.append(features[i : i + window])
        y.append(target[i + window : i + window + horizon])
    return np.array(X, dtype="float32"), np.array(y, dtype="float32")


def persistence_baseline(target: np.ndarray, window: int, horizon: int) -> np.ndarray:
    """Baseline naïve : la valeur future = dernière valeur observée."""
    preds = []
    for i in range(len(target) - window - horizon + 1):
        preds.append(np.repeat(target[i + window - 1], horizon))
    return np.array(preds, dtype="float32")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """MAE, RMSE et R² (sur les valeurs aplaties)."""
    y_true, y_pred = y_true.ravel(), y_pred.ravel()
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


# --------------------------------------------------------------------------
# Modèle
# --------------------------------------------------------------------------
class SolarLSTM:
    """LSTM empilé pour prévoir l'irradiance solaire sur `horizon` heures."""

    def __init__(self, window=24, horizon=1, feature_cols=None, target_col=TARGET_COL,
                 units=(64, 32), dropout=0.2, learning_rate=1e-3):
        self.window = window
        self.horizon = horizon
        self.feature_cols = list(feature_cols or FEATURE_COLS)
        self.target_col = target_col
        self.units = tuple(units)
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        self.model = None
        self.history = None

    # -- construction ------------------------------------------------------
    def build(self):
        """Construit le réseau Keras."""
        from tensorflow import keras
        from tensorflow.keras import layers

        set_seed()
        model = keras.Sequential(name="solar_lstm")
        model.add(layers.Input(shape=(self.window, len(self.feature_cols))))
        for i, u in enumerate(self.units):
            last = i == len(self.units) - 1
            model.add(layers.LSTM(u, return_sequences=not last))
            model.add(layers.Dropout(self.dropout))
        model.add(layers.Dense(self.horizon))
        model.compile(optimizer=keras.optimizers.Adam(self.learning_rate),
                      loss="mse", metrics=["mae"])
        self.model = model
        return model

    # -- préparation des données -------------------------------------------
    def _windows(self, df: pd.DataFrame, fit_scalers: bool = False):
        if fit_scalers:
            X = self.feature_scaler.fit_transform(df[self.feature_cols])
            y = self.target_scaler.fit_transform(df[[self.target_col]]).ravel()
        else:
            X = self.feature_scaler.transform(df[self.feature_cols])
            y = self.target_scaler.transform(df[[self.target_col]]).ravel()
        return make_windows(X, y, self.window, self.horizon)

    # -- entraînement ------------------------------------------------------
    def fit(self, train_df, val_df, epochs=50, batch_size=64, patience=8, verbose=1):
        """Entraîne le modèle (les scalers sont ajustés sur train uniquement)."""
        from tensorflow import keras

        X_tr, y_tr = self._windows(train_df, fit_scalers=True)
        X_va, y_va = self._windows(val_df)
        if self.model is None:
            self.build()
        callbacks = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                          restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
        ]
        self.history = self.model.fit(X_tr, y_tr, validation_data=(X_va, y_va),
                                      epochs=epochs, batch_size=batch_size,
                                      callbacks=callbacks, shuffle=False, verbose=verbose)
        return self.history

    train = fit  # alias pour l'interface commune

    # -- prédiction --------------------------------------------------------
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Retourne les prévisions en unités réelles (W/m²), shape (n, horizon)."""
        X, _ = self._windows(df)
        y_scaled = self.model.predict(X, verbose=0)
        y = self.target_scaler.inverse_transform(y_scaled.reshape(-1, 1)).reshape(-1, self.horizon)
        return np.clip(y, 0, None)  # pas d'irradiance négative

    def true_values(self, df: pd.DataFrame) -> np.ndarray:
        """Valeurs réelles alignées avec predict(df)."""
        _, y = make_windows(df[self.feature_cols].values, df[self.target_col].values,
                            self.window, self.horizon)
        return y

    def target_index(self, df: pd.DataFrame):
        """Horodatages correspondant au 1er pas prévu de chaque fenêtre."""
        return df.index[self.window : len(df) - self.horizon + 1]

    # -- évaluation --------------------------------------------------------
    def evaluate(self, df: pd.DataFrame) -> dict:
        """Métriques du LSTM et de la baseline de persistance."""
        y_true = self.true_values(df)
        y_pred = self.predict(df)
        y_base = persistence_baseline(df[self.target_col].values, self.window, self.horizon)
        return {"LSTM": compute_metrics(y_true, y_pred),
                "Persistance": compute_metrics(y_true, y_base)}

    # -- sauvegarde / chargement ------------------------------------------
    def save(self, directory: str = "models_saved", name: str = "lstm_solar"):
        os.makedirs(directory, exist_ok=True)
        self.model.save(os.path.join(directory, f"{name}.keras"))
        joblib.dump({"feature_scaler": self.feature_scaler,
                     "target_scaler": self.target_scaler},
                    os.path.join(directory, f"{name}_scalers.pkl"))
        meta = {"window": self.window, "horizon": self.horizon,
                "feature_cols": self.feature_cols, "target_col": self.target_col,
                "units": list(self.units), "dropout": self.dropout,
                "learning_rate": self.learning_rate}
        with open(os.path.join(directory, f"{name}_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, directory: str = "models_saved", name: str = "lstm_solar"):
        from tensorflow import keras

        with open(os.path.join(directory, f"{name}_meta.json")) as f:
            meta = json.load(f)
        obj = cls(**meta)
        scalers = joblib.load(os.path.join(directory, f"{name}_scalers.pkl"))
        obj.feature_scaler = scalers["feature_scaler"]
        obj.target_scaler = scalers["target_scaler"]
        obj.model = keras.models.load_model(os.path.join(directory, f"{name}.keras"))
        return obj