"""Chemins du projet et hyperparamètres des modèles."""
from pathlib import Path

# --- Chemins ---------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
LONDON_DIR = RAW_DIR / "london_smart_meters"
SOLAR_DIR = RAW_DIR / "solar_irradiance"
PROCESSED_DIR = DATA_DIR / "processed"
SIMULATED_DIR = DATA_DIR / "simulated"
MODELS_DIR = ROOT_DIR / "models_saved"
FIGURES_DIR = ROOT_DIR / "reports" / "figures"

# --- Général ---------------------------------------------------------------
RANDOM_SEED = 42
FREQ = "h"            # pas de temps après rééchantillonnage
TEST_SIZE = 0.2       # proportion test (découpage chronologique)

# --- Dev A (Serine) : régression linéaire (consommation) ----------------------------
LINEAR_REGRESSION = {
    "lags": [1, 2, 3, 24, 168],   # heures
    "rolling_windows": [24],
    "alpha": 1.0,                  # régularisation Ridge (0 = OLS)
}

# --- Dev B (Hadjer) : LSTM (solaire) ------------------------------------------------
LSTM_SOLAR = {
    "seq_len": 24,
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.2,
    "lr": 1e-3,
    "epochs": 20,
    "batch_size": 64,
    "features": ["ghi", "temperature", "hour_sin", "hour_cos", "doy_sin", "doy_cos"],
    "target": "pv_power",
}

# --- Dev C (Asma) : agent de tarification (réseau dense) --------------------------
PRICING_DENSE = {
    "hidden_sizes": [64, 32],
    "lr": 1e-3,
    "epochs": 50,
    "batch_size": 64,
    "price_min": 0.05,   # €/kWh
    "price_max": 0.40,   # €/kWh
}

# --- Simulation du micro-réseau --------------------------------------------
MICROGRID = {
    "n_households": 50,
    "pv_capacity_kw": 100.0,
    "battery_capacity_kwh": 200.0,
    "battery_power_kw": 50.0,
    "battery_efficiency": 0.95,
    "initial_soc": 0.5,
    "grid_import_price": 0.25,   # €/kWh
    "grid_export_price": 0.08,   # €/kWh
    "demand_elasticity": -0.3,   # élasticité prix de la demande
    "reference_price": 0.20,     # €/kWh
}
