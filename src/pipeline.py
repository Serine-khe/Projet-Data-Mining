"""Enchaînement des 3 modèles : consommation (A) + solaire (B) -> tarification (C) -> simulation."""
import argparse

import numpy as np
import pandas as pd

from src import config
from src.data import load_data, preprocess
from src.evaluation.metrics import regression_report
from src.models.linear_regression import ConsumptionForecaster
from src.models.lstm_solar import SolarForecaster
from src.models.pricing_dense import PricingAgent
from src.simulation.microgrid_env import fixed_price_policy
from src.simulation.run_simulation import generate_synthetic_data, simulate, summarize


def load_inputs(use_real_data: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    if use_real_data:
        consumption = preprocess.aggregate_consumption(
            preprocess.clean_consumption(load_data.load_london_smart_meters()))
        solar = preprocess.clean_solar(load_data.load_solar_irradiance())
        merged = consumption.merge(solar, on="timestamp")
        return merged[["timestamp", "consumption_kwh"]], merged.drop(columns="consumption_kwh")
    return generate_synthetic_data()


def align_forecasts(test_index: pd.Series, *series: tuple[pd.Series, np.ndarray]) -> pd.DataFrame:
    """Aligne les prévisions des deux modèles (fenêtres de retards différentes) sur le temps."""
    out = pd.DataFrame({"timestamp": test_index})
    for name, (ts, values) in series:
        out = out.merge(pd.DataFrame({"timestamp": ts.to_numpy(), name: values}), on="timestamp")
    return out


def run(use_real_data: bool = False, verbose: bool = True) -> dict:
    consumption, solar = load_inputs(use_real_data)
    c_train, c_test = preprocess.train_test_split_time(consumption)
    s_train, s_test = preprocess.train_test_split_time(solar)

    # --- A : consommation
    lr = ConsumptionForecaster().fit(c_train)
    c_true, c_pred = lr.predict_with_target(c_test)
    lr.save()
    n_dropped = len(c_test) - len(c_pred)
    c_ts = c_test["timestamp"].iloc[n_dropped:]

    # --- B : solaire
    lstm = SolarForecaster()
    lstm.fit(s_train, verbose=verbose)
    s_true, s_pred = lstm.predict_with_target(s_test)
    lstm.save()
    s_ts = s_test["timestamp"].iloc[config.LSTM_SOLAR["seq_len"]:]

    reports = {"consommation": regression_report(c_true, c_pred),
               "solaire": regression_report(s_true, s_pred)}

    # --- C : tarification sur les prévisions alignées
    f = align_forecasts(c_test["timestamp"], ("demand_kwh", (c_ts, c_pred)), ("solar_kwh", (s_ts, s_pred)))
    hours = f["timestamp"].dt.hour.to_numpy()
    rng = np.random.default_rng(config.RANDOM_SEED)
    states = np.column_stack([f["demand_kwh"], f["solar_kwh"], rng.uniform(0, 1, len(f)),
                              np.sin(2 * np.pi * hours / 24), np.cos(2 * np.pi * hours / 24)])
    agent = PricingAgent()
    agent.fit(states, verbose=verbose)
    agent.save()

    # --- Simulation : réel observé, décisions prises sur les prévisions
    real = c_test.set_index("timestamp")["consumption_kwh"].reindex(f["timestamp"]).to_numpy()
    real_solar = s_test.set_index("timestamp")["pv_power"].reindex(f["timestamp"]).to_numpy()
    forecast_states = f[["demand_kwh", "solar_kwh"]].to_numpy()

    def agent_policy(state):
        t = min(int(agent_policy.t), len(forecast_states) - 1)
        agent_policy.t += 1
        s = state.copy()
        s[:2] = forecast_states[t]
        return agent.act(s)
    agent_policy.t = 0

    sims = {"prix_fixe": summarize(simulate(real, real_solar, fixed_price_policy(), hours)),
            "agent_dense": summarize(simulate(real, real_solar, agent_policy, hours))}
    reports["simulation"] = sims

    if verbose:
        for k, v in reports.items():
            print(f"\n== {k} ==")
            for kk, vv in v.items():
                print(f"  {kk}: {vv}")
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-data", action="store_true", help="utilise data/raw au lieu des données synthétiques")
    args = parser.parse_args()
    run(use_real_data=args.real_data)


if __name__ == "__main__":
    main()
