import numpy as np
import pandas as pd
import pytest

from src import config
from src.evaluation.metrics import mae, mape, r2, rmse
from src.models import linear_regression as lr
from src.models.lstm_solar import estimate_pv_power
from src.models.pricing_dense import best_day_tariffs, best_tariff_index, make_features

tf = pytest.importorskip("tensorflow")
from src.models.lstm_solar import SolarLSTM, add_time_features  # noqa: E402
from src.models.pricing_dense import PricingAgent  # noqa: E402


def _consumption(n_days=40):
    rng = np.random.default_rng(0)
    index = pd.date_range("2013-01-01", periods=n_days * 24, freq="h")
    h = index.hour.to_numpy()
    kwh = 0.3 + 0.15 * np.exp(-((h - 8) ** 2) / 6) + 0.3 * np.exp(-((h - 19) ** 2) / 8)
    return pd.DataFrame({"mean_kwh": kwh * rng.lognormal(0, 0.05, len(index)),
                         "temperature": 10 + rng.normal(0, 2, len(index))}, index=index)


def _solar(n_days=20):
    index = pd.date_range("2016-09-01", periods=n_days * 24, freq="h")
    rad = 900 * np.clip(np.sin(np.pi * (index.hour - 6) / 12), 0, None)
    df = pd.DataFrame({"Radiation": rad, "Temperature": 50 + rad / 100, "Humidity": 80.0,
                       "Pressure": 30.4, "Speed": 5.0}, index=index)
    return add_time_features(df)


def test_metrics():
    y = np.array([1.0, 2.0, 3.0])
    assert mae(y, y) == 0 and rmse(y, y) == 0 and r2(y, y) == 1
    assert mape(np.array([0.0, 2.0]), np.array([5.0, 1.0])) == pytest.approx(50.0)


def test_linear_regression_learns_and_reloads(tmp_path):
    X, y = lr.build_features(_consumption())
    assert not X.isna().any().any()
    (X_tr, y_tr), _, (X_te, y_te) = lr.split_chronological(X, y)
    model = lr.ConsumptionModel("ridge", 1.0).fit(X_tr, y_tr)
    assert model.evaluate(X_te, y_te)["R2"] > 0.8

    model.save(tmp_path, "lr")
    assert np.allclose(lr.ConsumptionModel.load(tmp_path, "lr").predict(X_te), model.predict(X_te))


def test_lstm_shapes_and_positive_output():
    df = _solar()
    model = SolarLSTM(window=6, horizon=1, units=(8,))
    train, val, test = df.iloc[:300], df.iloc[300:400], df.iloc[400:]
    model.fit(train, val, epochs=2, verbose=0)
    pred = model.predict(test)
    assert pred.shape == (len(test) - 6, 1)
    assert (pred >= 0).all()
    assert len(model.target_index(test)) == len(pred) == len(model.true_values(test))


def test_estimate_pv_power():
    assert estimate_pv_power(1000, peak_kw=100, performance_ratio=0.8) == pytest.approx(80)
    assert estimate_pv_power(-5, peak_kw=100) == 0


def test_best_day_tariffs_are_price_neutral():
    tariffs = np.array(config.PRICING["tariffs"])
    hours = np.arange(24)
    demand = 40 + 25 * np.exp(-((hours - 19) ** 2) / 6)           # pointe le soir
    pv = 120 * np.clip(np.sin(np.pi * (hours - 6) / 12), 0, None)  # surplus à midi
    prices = tariffs[best_day_tariffs(demand, pv)]
    assert prices.mean() == pytest.approx(config.MICROGRID["reference_price"], abs=0.005)
    assert prices[19] > prices[12]                                  # cher à la pointe, bon marché au soleil

    labels = best_tariff_index(np.tile(demand, 2), np.tile(pv, 2), np.repeat([1, 2], 24))
    assert (labels[:24] == labels[24:]).all()


def test_features_use_past_only():
    X1 = make_features([10, 20, 30], [0, 0, 0], [0, 1, 2])
    X2 = make_features([10, 20, 30, 999], [0, 0, 0, 0], [0, 1, 2, 3])
    assert np.allclose(X1, X2[:3])      # ajouter une heure future ne change pas le passé


def test_pricing_agent_learns_labels(tmp_path):
    rng = np.random.default_rng(0)
    n_days = 60
    hours = np.tile(np.arange(24), n_days)
    days = np.repeat(np.arange(n_days), 24)
    demand = 40 + 25 * np.exp(-((hours - 19) ** 2) / 6) + rng.normal(0, 3, len(hours))
    pv = 120 * np.clip(np.sin(np.pi * (hours - 6) / 12), 0, None) * np.repeat(rng.uniform(0.2, 1, n_days), 24)
    X, y = make_features(demand, pv, hours), best_tariff_index(demand, pv, days)
    agent = PricingAgent()
    agent.fit(X, y, epochs=40, verbose=0)
    assert (np.abs(agent.predict_index(X) - y) <= 1).mean() > 0.8
    assert set(agent.predict_price(X)) <= set(config.PRICING["tariffs"])

    agent.save(tmp_path, "agent")
    reloaded = PricingAgent.load(tmp_path, "agent")
    assert (reloaded.predict_index(X) == agent.predict_index(X)).all()

    # la politique pas-à-pas donne les mêmes tarifs que le calcul en bloc
    policy = reloaded.make_policy()
    online = [policy({"demand_forecast": d, "pv_forecast": p, "soc": 0.5, "hour": h})
              for d, p, h in zip(demand[:48], pv[:48], hours[:48])]
    assert np.allclose(online, reloaded.predict_price(X[:48]))
