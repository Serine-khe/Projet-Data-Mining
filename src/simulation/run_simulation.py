"""Génération de données synthétiques et exécution de la simulation du micro-réseau."""
import argparse

import numpy as np
import pandas as pd

from src import config
from src.simulation.microgrid_env import MicrogridEnv, fixed_price_policy


def generate_synthetic_data(n_days: int = 120, seed: int = config.RANDOM_SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Crée des séries horaires réalistes de consommation et de production solaire."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2013-03-01", periods=n_days * 24, freq="h")
    hour, doy = ts.hour.to_numpy(), ts.dayofyear.to_numpy()

    # Consommation : pics matin/soir, effet saisonnier et week-end
    daily = 0.6 + 0.4 * np.exp(-((hour - 8) ** 2) / 6) + 0.8 * np.exp(-((hour - 19) ** 2) / 8)
    seasonal = 1 + 0.2 * np.cos(2 * np.pi * (doy - 15) / 365)
    weekend = np.where(ts.dayofweek >= 5, 1.1, 1.0)
    n = config.MICROGRID["n_households"]
    consumption = n * 0.5 * daily * seasonal * weekend * rng.lognormal(0, 0.1, len(ts))
    consumption_df = pd.DataFrame({"timestamp": ts, "consumption_kwh": consumption})

    # Irradiance : cloche diurne, saison, couverture nuageuse journalière
    elevation = np.clip(np.sin(np.pi * (hour - 6) / 12), 0, None)
    season = 0.6 + 0.4 * np.sin(2 * np.pi * (doy - 80) / 365)
    clouds = np.repeat(rng.uniform(0.3, 1.0, n_days), 24)
    ghi = 1000 * elevation * season * clouds * rng.uniform(0.9, 1.0, len(ts))
    temperature = 10 + 8 * season + 5 * elevation + rng.normal(0, 1, len(ts))
    pv_power = config.MICROGRID["pv_capacity_kw"] * ghi / 1000 * (1 - 0.004 * (temperature - 25))
    solar_df = pd.DataFrame({"timestamp": ts, "ghi": ghi, "temperature": temperature,
                             "pv_power": np.clip(pv_power, 0, None)})
    return consumption_df, solar_df


def save_synthetic_data(n_days: int = 120) -> None:
    config.SIMULATED_DIR.mkdir(parents=True, exist_ok=True)
    consumption, solar = generate_synthetic_data(n_days)
    consumption.to_csv(config.SIMULATED_DIR / "consumption.csv", index=False)
    solar.to_csv(config.SIMULATED_DIR / "solar.csv", index=False)
    print(f"Données synthétiques écrites dans {config.SIMULATED_DIR}")


def simulate(demand: np.ndarray, solar: np.ndarray, policy, hours=None) -> pd.DataFrame:
    env = MicrogridEnv(demand, solar, hours)
    return pd.DataFrame(env.run(policy))


def summarize(results: pd.DataFrame) -> dict:
    return {
        "profit_total": float(results["reward"].sum()),
        "import_total_kwh": float(results["grid_import"].sum()),
        "export_total_kwh": float(results["grid_export"].sum()),
        "autoconsommation": float(1 - results["grid_import"].sum() / results["demand"].sum()),
        "prix_moyen": float(results["price"].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate-only", action="store_true", help="génère uniquement les données synthétiques")
    parser.add_argument("--days", type=int, default=120)
    args = parser.parse_args()

    save_synthetic_data(args.days)
    if args.generate_only:
        return

    consumption, solar = generate_synthetic_data(args.days)
    hours = consumption["timestamp"].dt.hour.to_numpy()
    policies = {"prix_fixe": fixed_price_policy()}
    try:
        from src.models.pricing_dense import PricingAgent
        policies["agent_dense"] = PricingAgent.load().act
    except FileNotFoundError:
        print("Aucun agent sauvegardé : lancer `python -m src.pipeline` pour l'entraîner.")

    for name, policy in policies.items():
        res = simulate(consumption["consumption_kwh"].to_numpy(), solar["pv_power"].to_numpy(), policy, hours)
        res.to_csv(config.SIMULATED_DIR / f"simulation_{name}.csv", index=False)
        print(name, {k: round(v, 3) for k, v in summarize(res).items()})


if __name__ == "__main__":
    main()
