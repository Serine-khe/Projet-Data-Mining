"""Chargement des fichiers bruts et des fichiers traités.

Fichiers bruts attendus (non versionnés, voir README) :
  data/raw/london_smart_meters/elec_block.csv             consommation demi-horaire par bloc de foyers
  data/raw/london_smart_meters/weather_hourly_darksky.csv météo horaire de Londres
  data/raw/solar_irradiance/SolarPrediction.csv           irradiance HI-SEAS (Hawaï), mesures toutes les 5 min
"""
import pandas as pd

from src import config


def load_elec_blocks(path=config.LONDON_DIR / "elec_block.csv") -> pd.DataFrame:
    """Format large : index tstp (30 min), une colonne par bloc (kWh moyen par foyer)."""
    return pd.read_csv(path, parse_dates=["tstp"], index_col="tstp").sort_index().astype("float32")


def load_weather(path=config.LONDON_DIR / "weather_hourly_darksky.csv") -> pd.DataFrame:
    weather = pd.read_csv(path, parse_dates=["time"]).set_index("time").sort_index()
    return weather[~weather.index.duplicated()]


def load_solar_raw(path=config.SOLAR_DIR / "SolarPrediction.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def load_processed():
    """Fichiers horaires produits par les notebooks 01 et 02 (ou preprocess.py)."""
    consumption = pd.read_csv(config.PROCESSED_DIR / "consumption_hourly.csv", index_col=0, parse_dates=True)
    solar = pd.read_csv(config.PROCESSED_DIR / "solar_hourly.csv", index_col=0, parse_dates=True)
    return consumption, solar
