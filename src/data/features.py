"""Construction des variables explicatives."""
import numpy as np
import pandas as pd


def add_time_features(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    """Ajoute heure, jour de semaine et encodages cycliques."""
    df = df.copy()
    ts = pd.to_datetime(df[ts_col])
    df["hour"] = ts.dt.hour
    df["dayofweek"] = ts.dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    doy = ts.dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return df


def add_lag_features(df: pd.DataFrame, col: str, lags: list[int],
                     rolling_windows: list[int] | None = None) -> pd.DataFrame:
    """Ajoute des retards et moyennes glissantes (décalées pour éviter la fuite)."""
    df = df.copy()
    for lag in lags:
        df[f"{col}_lag{lag}"] = df[col].shift(lag)
    for w in rolling_windows or []:
        df[f"{col}_roll{w}"] = df[col].shift(1).rolling(w).mean()
    return df.dropna().reset_index(drop=True)


def make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    """Transforme des séries en fenêtres glissantes pour le LSTM.

    Retourne ``X_seq`` de forme (n, seq_len, n_features) et ``y_seq`` de forme (n,),
    où ``y_seq[i]`` est la cible au pas suivant la fenêtre ``i``.
    """
    n = len(X) - seq_len
    if n <= 0:
        raise ValueError("Série trop courte pour la longueur de séquence demandée")
    X_seq = np.stack([X[i:i + seq_len] for i in range(n)])
    y_seq = y[seq_len:]
    return X_seq.astype(np.float32), y_seq.astype(np.float32)
