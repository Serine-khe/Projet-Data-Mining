"""Nettoyage et rééchantillonnage des données.

==============================================================================
PROMPT — code fonctionnel, à AJUSTER selon les résultats de l'EDA
==============================================================================
Serine — ``clean_consumption`` / ``aggregate_consumption`` : décider, d'après le
  notebook 01, comment traiter les valeurs manquantes, aberrantes et les foyers
  incomplets (seuils, interpolation, exclusion). Documenter les choix.
Hadjer — ``clean_solar`` : idem pour l'irradiance (notebook 02) : valeurs
  négatives, nuits, trous de mesure, changement d'heure.
``train_test_split_time`` : partagé, ne pas remplacer par un découpage aléatoire.
==============================================================================
"""
import numpy as np
import pandas as pd

from src import config


def clean_consumption(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime doublons et valeurs aberrantes, interpole les trous courts."""
    df = df.drop_duplicates(subset=["household_id", "timestamp"])
    df = df[df["kwh"].isna() | (df["kwh"] >= 0)]

    def _clean(group: pd.DataFrame) -> pd.DataFrame:
        group = group.set_index("timestamp").sort_index()
        upper = group["kwh"].quantile(0.999)
        group.loc[group["kwh"] > upper, "kwh"] = np.nan
        group["kwh"] = group["kwh"].interpolate(limit=4)
        return group

    df = df.groupby("household_id", group_keys=True).apply(_clean, include_groups=False)
    return df.reset_index().dropna(subset=["kwh"])


def aggregate_consumption(df: pd.DataFrame, freq: str = config.FREQ) -> pd.DataFrame:
    """Agrège la consommation de tous les foyers au pas ``freq`` (kWh)."""
    out = (
        df.set_index("timestamp")["kwh"]
        .resample(freq).sum(min_count=1)
        .interpolate(limit=3)
        .dropna()
        .rename("consumption_kwh")
        .reset_index()
    )
    return out


def clean_solar(df: pd.DataFrame, freq: str = config.FREQ) -> pd.DataFrame:
    """Rééchantillonne l'irradiance, borne les valeurs physiques."""
    df = df.set_index("timestamp").sort_index().resample(freq).mean()
    df["ghi"] = df["ghi"].clip(lower=0)
    if "pv_power" in df:
        df["pv_power"] = df["pv_power"].clip(lower=0)
    df = df.interpolate(limit=3).dropna()
    return df.reset_index()


def train_test_split_time(df: pd.DataFrame, test_size: float = config.TEST_SIZE):
    """Découpage chronologique (pas de mélange pour les séries temporelles)."""
    split = int(len(df) * (1 - test_size))
    return df.iloc[:split].copy(), df.iloc[split:].copy()
