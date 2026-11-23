"""Enchaînement des 3 modèles :
régression linéaire (consommation) + LSTM (solaire) -> réseau dense (tarif) -> simulation.

Prérequis : data/processed/consumption_hourly.csv et solar_hourly.csv (notebooks 01 et 02).

    python -m src.pipeline              # utilise les modèles sauvegardés par les notebooks 03 et 04
    python -m src.pipeline --retrain    # ré-entraîne la régression et le LSTM
"""
import argparse

import numpy as np
import pandas as pd

from src import config
from src.data.load_data import load_processed
from src.models import linear_regression as lr
from src.models import lstm_solar
from src.models.pricing_dense import PricingAgent, best_tariff_index, make_features
from src.simulation.microgrid_env import fixed_price_policy, rule_based_policy
from src.simulation.run_simulation import compare_policies

MG = config.MICROGRID


def consumption_forecasts(model, consumption):
    """Prévisions de la régression sur toute la période, converties pour le micro-réseau (kWh/h).
    Retourne aussi la date de début du test (après train + validation)."""
    X, y = lr.build_features(consumption)
    _, _, (X_test, _) = lr.split_chronological(X, y, config.TRAIN_SIZE, config.VAL_SIZE)
    n = MG["n_households"]
    forecasts = pd.DataFrame({"demand_true": y.values * n, "demand_pred": model.predict(X) * n}, index=X.index)
    return forecasts, X_test.index[0]


def solar_forecasts(model, solar):
    """Prévisions LSTM de l'irradiance, puis production PV ESTIMÉE (formule de lstm_solar).
    Les dates sont décalées de 2016 vers 2013 pour s'aligner sur la consommation."""
    df = lstm_solar.add_time_features(solar)
    _, _, test = lstm_solar.split_chronological(df, config.TRAIN_SIZE, config.VAL_SIZE)
    index = model.target_index(df) - pd.DateOffset(years=config.SOLAR_YEAR_SHIFT)
    radiation_true = model.true_values(df)[:, 0]
    radiation_pred = model.predict(df)[:, 0]

    def to_pv(g):
        return lstm_solar.estimate_pv_power(g, MG["pv_peak_kw"], MG["performance_ratio"])

    forecasts = pd.DataFrame({"radiation_true": radiation_true, "radiation_pred": radiation_pred,
                              "pv_true": to_pv(radiation_true), "pv_pred": to_pv(radiation_pred)}, index=index)
    test_start = model.target_index(test)[0] - pd.DateOffset(years=config.SOLAR_YEAR_SHIFT)
    return forecasts, test_start


def build_grid(demand_fc, demand_test_start, pv_fc, pv_test_start):
    """Joint les deux prévisions sur les heures communes.
    'test' = heures où les DEUX modèles n'ont jamais vu les données."""
    grid = demand_fc.join(pv_fc, how="inner")
    grid["period"] = np.where(grid.index >= max(demand_test_start, pv_test_start), "test", "train")
    return grid


def train_pricing_agent(grid_train, verbose=0):
    X = make_features(grid_train["demand_pred"], grid_train["pv_pred"], grid_train.index.hour)
    y = best_tariff_index(grid_train["demand_pred"], grid_train["pv_pred"], grid_train.index.date)
    agent = PricingAgent()
    agent.fit(X, y, verbose=verbose)
    return agent


def get_forecasting_models(consumption, solar, retrain=False):
    if not retrain:
        return (lr.ConsumptionModel.load(config.MODELS_DIR, "linear_regression"),
                lstm_solar.SolarLSTM.load(config.MODELS_DIR, "lstm_solar"))

    # Mêmes réglages que les modèles retenus dans les notebooks 03 et 04
    X, y = lr.build_features(consumption)
    (X_tr, y_tr), (X_va, y_va), _ = lr.split_chronological(X, y, config.TRAIN_SIZE, config.VAL_SIZE)
    lr_model = lr.ConsumptionModel("lasso", 1e-4).fit(pd.concat([X_tr, X_va]), pd.concat([y_tr, y_va]))

    df = lstm_solar.add_time_features(solar)
    train, val, _ = lstm_solar.split_chronological(df, config.TRAIN_SIZE, config.VAL_SIZE)
    lstm_model = lstm_solar.SolarLSTM(window=6, horizon=1, units=(32,), dropout=0.1)
    lstm_model.fit(train, val, epochs=40, verbose=0, shuffle=True)
    return lr_model, lstm_model


def run(retrain=False):
    consumption, solar = load_processed()
    lr_model, lstm_model = get_forecasting_models(consumption, solar, retrain)

    demand_fc, demand_test_start = consumption_forecasts(lr_model, consumption)
    pv_fc, pv_test_start = solar_forecasts(lstm_model, solar)
    grid = build_grid(demand_fc, demand_test_start, pv_fc, pv_test_start)
    grid_train, grid_test = grid[grid["period"] == "train"], grid[grid["period"] == "test"]
    print(f"Heures communes : {len(grid)} (entraînement agent : {len(grid_train)}, test : {len(grid_test)})")

    agent = train_pricing_agent(grid_train)
    agent.save()
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    grid.to_csv(config.PROCESSED_DIR / "grid_all.csv")
    grid_test.to_csv(config.PROCESSED_DIR / "grid_test.csv")

    policies = {"Tarif fixe": fixed_price_policy(), "Heures pleines / creuses": rule_based_policy(),
                "Réseau dense": agent.make_policy()}
    table, _ = compare_policies(grid_test, policies)
    print(table.round(2).to_string())
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--retrain", action="store_true", help="ré-entraîne la régression et le LSTM")
    run(parser.parse_args().retrain)


if __name__ == "__main__":
    main()
