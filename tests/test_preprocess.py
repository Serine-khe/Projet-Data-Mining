import numpy as np
import pandas as pd
import pytest

from src.data import preprocess
from src.data.features import add_lag_features, add_time_features


def _blocks():
    index = pd.date_range("2013-01-01", periods=96, freq="30min")
    blocks = pd.DataFrame({f"block_{i}": 0.2 + 0.01 * i for i in range(5)}, index=index)
    blocks.iloc[:10, :2] = np.nan        # 2 blocs pas encore actifs au début
    return blocks


def test_consumption_to_hourly():
    hourly = preprocess.consumption_to_hourly(_blocks())
    assert list(hourly.columns) == ["mean_kwh", "std_blocks"]
    assert (hourly.index.to_series().diff().dropna() == pd.Timedelta("1h")).all()
    # 2 demi-heures de ~0.22 kWh -> ~0.44 kWh par heure et par foyer
    assert hourly["mean_kwh"].iloc[-1] == pytest.approx(0.44)
    # les 10 premières demi-heures (moins de 80 % des blocs) sont retirées
    assert hourly.index[0] == pd.Timestamp("2013-01-01 05:00")


def test_solar_to_hourly():
    unix = pd.date_range("2016-09-01 10:00", periods=48, freq="5min").astype("int64") // 10**9
    raw = pd.DataFrame({"UNIXTime": unix, "Radiation": np.linspace(-1, 500, 48), "Temperature": 50,
                        "Pressure": 30.4, "Humidity": 80, "WindDirection(Degrees)": 100, "Speed": 5})
    hourly = preprocess.solar_to_hourly(raw)
    assert len(hourly) == 4
    assert hourly.index[0] == pd.Timestamp("2016-09-01 00:00")      # UTC -> heure d'Hawaï (UTC-10)
    assert (hourly["Radiation"] >= 0).all()


def test_add_holidays():
    index = pd.date_range("2013-12-24", periods=72, freq="h")
    out = preprocess.add_holidays(pd.DataFrame({"mean_kwh": 0.4}, index=index))
    assert out.loc["2013-12-25", "is_holiday"].eq(1).all()
    assert out.loc["2013-12-24", "is_holiday"].eq(0).all()


def test_time_features_are_cyclic():
    df = add_time_features(pd.DataFrame({"timestamp": pd.date_range("2013-01-01", periods=24, freq="h")}))
    assert np.allclose(df["hour_sin"] ** 2 + df["hour_cos"] ** 2, 1)


def test_lag_features_no_leakage():
    df = pd.DataFrame({"y": np.arange(10, dtype=float)})
    out = add_lag_features(df, "y", lags=[1, 3], rolling_windows=[2])
    assert (out["y"] - out["y_lag1"] == 1).all()
    assert (out["y"] - out["y_lag3"] == 3).all()
