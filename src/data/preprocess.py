"""Prétraitement : mêmes étapes que les notebooks 01 et 02, regroupées en fonctions.

    python -m src.data.preprocess     # recrée data/processed/*.csv à partir de data/raw
"""
import pandas as pd

from src import config
from src.data import load_data

WEATHER_COLS = ["temperature", "humidity", "windSpeed", "pressure"]


def consumption_to_hourly(blocks: pd.DataFrame, min_active_share: float = 0.8) -> pd.DataFrame:
    """Blocs demi-horaires -> consommation horaire moyenne par foyer (kWh)."""
    # On garde les instants où au moins 80 % des blocs ont des données (série stable)
    n_active = blocks.notna().sum(axis=1)
    stable = blocks[n_active >= min_active_share * blocks.shape[1]]

    half_hour = pd.DataFrame({"mean": stable.mean(axis=1), "std_blocks": stable.std(axis=1)})
    hourly = pd.DataFrame({
        "mean_kwh": half_hour["mean"].resample("h").sum(min_count=2),   # NaN si une demi-heure manque
        "std_blocks": half_hour["std_blocks"].resample("h").mean(),
    })
    hourly = hourly.interpolate(limit=3).dropna()
    hourly.index.name = "timestamp"
    return hourly


def add_holidays(hourly: pd.DataFrame) -> pd.DataFrame:
    import holidays

    uk = holidays.UK(years=sorted(set(hourly.index.year)))
    hourly = hourly.copy()
    hourly["is_holiday"] = [int(d in uk) for d in hourly.index.date]
    return hourly


def add_weather(hourly: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in WEATHER_COLS if c in weather.columns]
    out = hourly.join(weather[cols], how="left")
    out[cols] = out[cols].interpolate(limit=6)
    return out.dropna(subset=cols)


def solar_to_hourly(raw: pd.DataFrame) -> pd.DataFrame:
    """Mesures HI-SEAS (5 min, temps UNIX en UTC) -> moyennes horaires en heure locale d'Hawaï (UTC-10)."""
    df = raw.copy()
    df["timestamp"] = pd.to_datetime(df["UNIXTime"], unit="s") - pd.Timedelta(hours=10)
    df = df.sort_values("timestamp").set_index("timestamp")
    df = df.rename(columns={"WindDirection(Degrees)": "WindDirection"})
    df = df[["Radiation", "Temperature", "Pressure", "Humidity", "WindDirection", "Speed"]]
    df = df[~df.index.duplicated(keep="first")]
    df["Radiation"] = df["Radiation"].clip(lower=0)
    return df.resample("1h").mean().interpolate(limit=3).dropna()


def main():
    consumption = add_holidays(consumption_to_hourly(load_data.load_elec_blocks()))
    weather_path = config.LONDON_DIR / "weather_hourly_darksky.csv"
    if weather_path.exists():
        consumption = add_weather(consumption, load_data.load_weather(weather_path))
    solar = solar_to_hourly(load_data.load_solar_raw())

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    consumption.to_csv(config.PROCESSED_DIR / "consumption_hourly.csv")
    solar.to_csv(config.PROCESSED_DIR / "solar_hourly.csv")
    print("consumption_hourly.csv :", consumption.shape, "| solar_hourly.csv :", solar.shape)


if __name__ == "__main__":
    main()
