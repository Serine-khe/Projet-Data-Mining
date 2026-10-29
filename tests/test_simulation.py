import numpy as np
import pytest

from src import config
from src.simulation.microgrid_env import MicrogridEnv, fixed_price_policy
from src.simulation.run_simulation import generate_synthetic_data, simulate, summarize


def test_synthetic_data_shapes():
    consumption, solar = generate_synthetic_data(n_days=5)
    assert len(consumption) == len(solar) == 5 * 24
    assert (consumption["consumption_kwh"] > 0).all()
    assert (solar["pv_power"] >= 0).all()
    assert solar.loc[solar["timestamp"].dt.hour == 0, "pv_power"].max() == pytest.approx(0)


def test_soc_stays_in_bounds():
    rng = np.random.default_rng(0)
    env = MicrogridEnv(rng.uniform(0, 100, 200), rng.uniform(0, 150, 200))
    history = env.run(fixed_price_policy())
    soc = np.array([h["soc"] for h in history])
    assert len(history) == 200
    assert soc.min() >= 0 and soc.max() <= 1


def test_energy_balance():
    env = MicrogridEnv(np.array([30.0, 10.0, 80.0]), np.array([0.0, 60.0, 20.0]))
    for h in env.run(fixed_price_policy()):
        supply = h["solar"] + h["grid_import"] - h["grid_export"] - h["battery_flow"]
        assert supply == pytest.approx(h["demand"])


def test_higher_price_lowers_demand():
    env = MicrogridEnv(np.array([50.0]), np.array([0.0]))
    low = env.step(0.10).info["demand"]
    env.reset()
    high = env.step(0.40).info["demand"]
    assert high < low


def test_summary_keys():
    consumption, solar = generate_synthetic_data(n_days=3)
    res = simulate(consumption["consumption_kwh"].to_numpy(), solar["pv_power"].to_numpy(), fixed_price_policy())
    summary = summarize(res)
    assert set(summary) == {"profit_total", "import_total_kwh", "export_total_kwh", "autoconsommation", "prix_moyen"}
    assert summary["prix_moyen"] == pytest.approx(config.MICROGRID["reference_price"])


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        MicrogridEnv(np.ones(3), np.ones(4))
