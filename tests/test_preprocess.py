import numpy as np
import pandas as pd
import pytest

from src.data import preprocess
from src.data.features import add_lag_features, add_time_features, make_sequences


def _raw_consumption():
    ts = pd.date_range("2013-01-01", periods=96, freq="30min")
    df = pd.DataFrame({"household_id": "MAC000001", "timestamp": ts, "kwh": 0.2})
    df.loc[5, "kwh"] = np.nan
    df.loc[10, "kwh"] = -1.0
    return pd.concat([df, df.iloc[:3]])   # doublons


def test_clean_consumption_removes_duplicates_and_negatives():
    out = preprocess.clean_consumption(_raw_consumption())
    assert not out.duplicated(subset=["household_id", "timestamp"]).any()
    assert (out["kwh"] >= 0).all()
    assert out["kwh"].notna().all()


def test_aggregate_consumption_is_hourly():
    out = preprocess.aggregate_consumption(preprocess.clean_consumption(_raw_consumption()))
    assert list(out.columns) == ["timestamp", "consumption_kwh"]
    assert (out["timestamp"].diff().dropna() == pd.Timedelta("1h")).all()


def test_clean_solar_clips_negative_irradiance():
    ts = pd.date_range("2013-06-01", periods=48, freq="h")
    df = pd.DataFrame({"timestamp": ts, "ghi": np.linspace(-10, 500, 48), "temperature": 20.0})
    assert (preprocess.clean_solar(df)["ghi"] >= 0).all()


def test_time_split_is_chronological():
    df = pd.DataFrame({"timestamp": pd.date_range("2013-01-01", periods=100, freq="h")})
    train, test = preprocess.train_test_split_time(df, 0.2)
    assert len(train) == 80 and len(test) == 20
    assert train["timestamp"].max() < test["timestamp"].min()


def test_time_features_are_cyclic():
    df = add_time_features(pd.DataFrame({"timestamp": pd.date_range("2013-01-01", periods=24, freq="h")}))
    assert np.allclose(df["hour_sin"] ** 2 + df["hour_cos"] ** 2, 1)


def test_lag_features_no_leakage():
    df = pd.DataFrame({"y": np.arange(10, dtype=float)})
    out = add_lag_features(df, "y", lags=[1, 3], rolling_windows=[2])
    assert (out["y"] - out["y_lag1"] == 1).all()
    assert (out["y"] - out["y_lag3"] == 3).all()
    assert (out["y_roll2"] == out["y"] - 1.5).all()


def test_make_sequences_shapes():
    X, y = np.random.rand(50, 3), np.arange(50)
    Xs, ys = make_sequences(X, y, seq_len=10)
    assert Xs.shape == (40, 10, 3) and ys.shape == (40,)
    assert ys[0] == 10
    with pytest.raises(ValueError):
        make_sequences(X, y, seq_len=60)
