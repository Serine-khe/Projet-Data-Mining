"""Simulation du micro-réseau et indicateurs de comparaison des stratégies de tarification.

Usage (après avoir exécuté les notebooks 01 à 05, ou `python -m src.pipeline`) :
    python -m src.simulation.run_simulation
"""
import pandas as pd

from src import config
from src.simulation.microgrid_env import MicrogridEnv, fixed_price_policy, rule_based_policy


def simulate(grid: pd.DataFrame, policy, params=config.MICROGRID) -> pd.DataFrame:
    """grid : colonnes demand_true, pv_true, demand_pred, pv_pred (kWh/h), index horaire."""
    env = MicrogridEnv(grid["demand_true"], grid["pv_true"],
                       demand_forecast=grid["demand_pred"], pv_forecast=grid["pv_pred"],
                       hours=grid.index.hour, params=params)
    results = env.run(policy)
    results.index = grid.index
    return results


def summarize(results: pd.DataFrame) -> dict:
    """Indicateurs calculés sur les valeurs RÉELLES simulées."""
    pv_total = results["pv"].sum()
    return {
        "Déséquilibre |D - PV| (kWh)": results["imbalance"].sum(),
        "Import réseau (kWh)": results["grid_import"].sum(),
        "Export réseau (kWh)": results["grid_export"].sum(),
        "Pic d'import réseau (kW)": results["grid_import"].max(),
        "Pic de demande (kW)": results["demand"].max(),
        "Pic de demande nette D - PV (kW)": (results["demand"] - results["pv"]).max(),
        "Consommation totale (kWh)": results["demand"].sum(),
        "Prix moyen (€/kWh)": results["price"].mean(),
        "Facture des foyers (€)": (results["demand"] * results["price"]).sum(),
        "Autoconsommation PV (%)": 100 * (pv_total - results["grid_export"].sum()) / pv_total if pv_total > 0 else float("nan"),
    }


def compare_policies(grid: pd.DataFrame, policies: dict, params=config.MICROGRID) -> tuple[pd.DataFrame, dict]:
    """Simule chaque politique ; retourne le tableau d'indicateurs et les résultats détaillés."""
    results = {name: simulate(grid, policy, params) for name, policy in policies.items()}
    table = pd.DataFrame({name: summarize(r) for name, r in results.items()}).T
    return table, results


def main():
    from src.models.pricing_dense import PricingAgent

    grid = pd.read_csv(config.PROCESSED_DIR / "grid_test.csv", index_col=0, parse_dates=True)
    agent = PricingAgent.load()
    policies = {"Tarif fixe": fixed_price_policy(), "Heures pleines / creuses": rule_based_policy(),
                "Réseau dense": agent.make_policy()}
    table, results = compare_policies(grid, policies)
    config.SIMULATED_DIR.mkdir(parents=True, exist_ok=True)
    for name, r in results.items():
        file_name = name.lower().replace(" / ", "_").replace(" ", "_")
        r.to_csv(config.SIMULATED_DIR / f"simulation_{file_name}.csv")
    print(table.round(2).to_string())


if __name__ == "__main__":
    main()
