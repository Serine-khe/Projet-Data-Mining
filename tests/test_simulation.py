import numpy as np
import pandas as pd
import pytest

from src import config
from src.simulation.microgrid_env import MicrogridEnv, fixed_price_policy, rule_based_policy
from src.simulation.run_simulation import compare_policies, summarize


def _grid(n=72):
    rng = np.random.default_rng(0)
    index = pd.date_range("2013-09-01", periods=n, freq="h")
    pv = 100 * np.clip(np.sin(np.pi * (index.hour - 6) / 12), 0, None)
    demand = 40 + 20 * rng.random(n)
    return pd.DataFrame({"demand_true": demand, "pv_true": pv,
                         "demand_pred": demand * 1.05, "pv_pred": pv * 0.95}, index=index)


def test_soc_stays_in_bounds():
    rng = np.random.default_rng(0)
    env = MicrogridEnv(rng.uniform(0, 100, 200), rng.uniform(0, 150, 200))
    soc = env.run(fixed_price_policy())["soc"]
    assert len(soc) == 200
    assert soc.min() >= 0 and soc.max() <= 1


def test_energy_balance():
    env = MicrogridEnv(np.array([30.0, 10.0, 80.0, 5.0]), np.array([0.0, 60.0, 20.0, 90.0]))
    for _, h in env.run(fixed_price_policy()).iterrows():
        supply = h["pv"] + h["grid_import"] - h["grid_export"] - h["battery_flow"]
        assert supply == pytest.approx(h["demand"])


def test_higher_price_lowers_demand():
    env = MicrogridEnv(np.array([50.0]), np.array([0.0]))
    assert env.demand_response(50, 0.30) < 50 < env.demand_response(50, 0.10)
    assert env.demand_response(50, config.MICROGRID["reference_price"]) == pytest.approx(50)


def test_policy_sees_forecasts_not_true_values():
    seen = []
    env = MicrogridEnv([10.0, 20.0], [0.0, 0.0], demand_forecast=[11.0, 22.0])
    env.run(lambda s: seen.append(s["demand_forecast"]) or 0.2)
    assert seen == [11.0, 22.0]


def test_rule_based_policy_is_price_neutral():
    policy = rule_based_policy()
    tariffs = config.PRICING["tariffs"]
    prices = [policy({"demand_forecast": 50, "pv_forecast": 0, "soc": 0.5, "hour": h}) for h in range(24)]
    assert prices[19] == max(tariffs) and prices[13] == min(tariffs) and prices[3] == config.MICROGRID["reference_price"]
    assert np.mean(prices) == pytest.approx(config.MICROGRID["reference_price"])


def test_compare_policies_table():
    table, results = compare_policies(_grid(), {"fixe": fixed_price_policy(), "règles": rule_based_policy()})
    assert list(table.index) == ["fixe", "règles"]
    assert table.loc["fixe", "Prix moyen (€/kWh)"] == pytest.approx(config.MICROGRID["reference_price"])
    assert 0 <= table.loc["fixe", "Autoconsommation PV (%)"] <= 100
    assert len(results["fixe"]) == 72
    assert set(summarize(results["fixe"])) == set(table.columns)


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        MicrogridEnv(np.ones(3), np.ones(4))
