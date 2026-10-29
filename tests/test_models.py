import numpy as np
import pytest

from src import config
from src.evaluation.metrics import mae, mape, r2, rmse
from src.models.linear_regression import ConsumptionForecaster
from src.simulation.run_simulation import generate_synthetic_data

torch = pytest.importorskip("torch")
from src.models.lstm_solar import SolarForecaster  # noqa: E402
from src.models.pricing_dense import PricingAgent, target_price  # noqa: E402


@pytest.fixture(scope="module")
def data():
    return generate_synthetic_data(n_days=30)


def test_metrics():
    y = np.array([1.0, 2.0, 3.0])
    assert mae(y, y) == 0 and rmse(y, y) == 0 and r2(y, y) == 1
    assert mape(np.array([0.0, 2.0]), np.array([5.0, 1.0])) == pytest.approx(50.0)


def test_linear_regression_learns(data, tmp_path):
    consumption, _ = data
    model = ConsumptionForecaster().fit(consumption)
    y_true, y_pred = model.predict_with_target(consumption)
    assert r2(y_true, y_pred) > 0.7

    model.save(tmp_path / "lr.pkl")
    reloaded = ConsumptionForecaster.load(tmp_path / "lr.pkl")
    assert np.allclose(reloaded.predict(consumption), y_pred)


def test_lstm_solar_trains_and_reloads(data, tmp_path):
    _, solar = data
    params = {**config.LSTM_SOLAR, "epochs": 3, "hidden_size": 16, "num_layers": 1}
    model = SolarForecaster(params)
    history = model.fit(solar, verbose=False)
    assert history[-1] < history[0]

    y_true, y_pred = model.predict_with_target(solar)
    assert len(y_true) == len(y_pred) == len(solar) - params["seq_len"]
    assert (y_pred >= 0).all()

    model.save(tmp_path / "lstm.pt")
    assert np.allclose(SolarForecaster.load(tmp_path / "lstm.pt").predict(solar), y_pred)


def test_pricing_agent_respects_bounds(tmp_path):
    rng = np.random.default_rng(0)
    states = np.column_stack([rng.uniform(0, 50, 500), rng.uniform(0, 80, 500), rng.uniform(0, 1, 500),
                              rng.uniform(-1, 1, 500), rng.uniform(-1, 1, 500)])
    agent = PricingAgent({**config.PRICING_DENSE, "epochs": 30})
    agent.fit(states, verbose=False)

    prices = agent.predict(states)
    p = config.PRICING_DENSE
    assert prices.min() >= p["price_min"] and prices.max() <= p["price_max"]
    assert mae(target_price(states), prices) < 0.03

    # Déficit fort -> prix plus élevé qu'un surplus fort
    deficit = np.array([[50, 0, 0.2, 0, 1]], dtype=float)
    surplus = np.array([[5, 80, 0.8, 0, 1]], dtype=float)
    assert agent.act(deficit) > agent.act(surplus)

    agent.save(tmp_path / "pricing.pt")
    assert np.allclose(PricingAgent.load(tmp_path / "pricing.pt").predict(states), prices)
