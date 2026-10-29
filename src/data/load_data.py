"""Chargement des données brutes (London Smart Meters, irradiance solaire)."""
from pathlib import Path

import pandas as pd

from src import config


def load_london_smart_meters(path: Path = config.LONDON_DIR, nrows: int | None = None) -> pd.DataFrame:
    """Charge les CSV London Smart Meters (relevés demi-horaires).

    Retourne un DataFrame avec les colonnes ``household_id``, ``timestamp``, ``kwh``.
    """
    files = sorted(Path(path).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"Aucun fichier CSV trouvé dans {path}")

    frames = [pd.read_csv(f, nrows=nrows) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={
        "LCLid": "household_id",
        "DateTime": "timestamp",
        "KWH/hh (per half hour) ": "kwh",
        "KWH/hh (per half hour)": "kwh",
    })
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["kwh"] = pd.to_numeric(df["kwh"], errors="coerce")
    return df[["household_id", "timestamp", "kwh"]]


def load_solar_irradiance(path: Path = config.SOLAR_DIR) -> pd.DataFrame:
    """Charge les CSV d'irradiance solaire.

    Colonnes attendues : ``timestamp``, ``ghi`` (W/m²), ``temperature`` (°C),
    et optionnellement ``pv_power`` (kW).
    """
    files = sorted(Path(path).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"Aucun fichier CSV trouvé dans {path}")

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def load_processed(name: str) -> pd.DataFrame:
    """Charge un fichier nettoyé depuis ``data/processed``."""
    return pd.read_csv(config.PROCESSED_DIR / f"{name}.csv", parse_dates=["timestamp"])
