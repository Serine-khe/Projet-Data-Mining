"""Chemins du projet et paramètres communs (micro-réseau, tarification)."""
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

RANDOM_SEED = 42

# --- Découpage temporel (identique dans les notebooks 03 et 04) ------------
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15          # le reste (15 %) sert de test

# --- Alignement des deux jeux de données -----------------------------------
# Irradiance : Hawaï 2016. Consommation : Londres 2011-2014.
# On décale l'irradiance de 3 ans (2016 -> 2013) en gardant mois, jour et heure.
SOLAR_YEAR_SHIFT = 3

# --- Micro-réseau simulé ---------------------------------------------------
MICROGRID = {
    "n_households": 100,          # foyers du micro-réseau (conso moyenne par foyer x 100)
    "pv_peak_kw": 150.0,          # puissance crête des panneaux (kWc)
    "performance_ratio": 0.80,    # pertes onduleur, câbles, température...
    "battery_capacity_kwh": 200.0,
    "battery_power_kw": 50.0,
    "battery_efficiency": 0.95,
    "initial_soc": 0.5,
    "demand_elasticity": -0.2,    # hypothèse : +10 % de prix -> environ -2 % de demande
    "reference_price": 0.20,      # €/kWh, tarif fixe de référence
}

# --- Agent de tarification (réseau dense) ----------------------------------
PRICING = {
    "tariffs": [0.10, 0.15, 0.20, 0.25, 0.30],   # niveaux de tarif possibles (€/kWh)
    "hidden_units": [32, 16],
    "epochs": 60,
    "batch_size": 32,
    "learning_rate": 1e-3,
    # Stratégie à règles (heures pleines / creuses) : prix moyen sur 24 h = 0,20 €/kWh
    "peak_hours": [17, 18, 19, 20, 21],     # tarif haut (pointe du soir)
    "solar_hours": [11, 12, 13, 14, 15],    # tarif bas (milieu de journée, soleil)
}
