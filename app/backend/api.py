"""Backend de l'interface web : API REST (FastAPI) au-dessus des modèles du projet.

Prérequis : notebooks 01 à 05 exécutés (ou `python -m src.pipeline`), pour avoir
data/processed/grid_all.csv et les modèles dans models_saved/.

Lancer depuis la racine du projet :
    python -m uvicorn app.backend.api:app --reload
puis ouvrir http://127.0.0.1:8000 (le frontend est servi par le même serveur).
Documentation automatique de l'API : http://127.0.0.1:8000/docs
"""
import os
from contextlib import asynccontextmanager
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from src import config  # noqa: E402
from src.models.linear_regression import compute_metrics  # noqa: E402
from src.models.pricing_dense import PricingAgent, make_features  # noqa: E402
from src.simulation.microgrid_env import fixed_price_policy, rule_based_policy  # noqa: E402
from src.simulation.run_simulation import compare_policies  # noqa: E402

FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"
GRID_PATH = config.PROCESSED_DIR / "grid_all.csv"

# Chargés une seule fois (au démarrage du serveur, ou au premier appel)
state = {"grid": None, "agent": None}


@asynccontextmanager
async def lifespan(app):
    # Charger les données et l'agent dès le démarrage : le premier appel n'attend pas TensorFlow
    if GRID_PATH.exists():
        test = get_grid().query("period == 'test'")
        get_agent().predict_price(make_features(test["demand_pred"], test["pv_pred"], test.index.hour))
    yield


app = FastAPI(title="Micro-réseau intelligent — API",
              description="Prévisions, tarification dynamique et simulation du micro-réseau.", lifespan=lifespan)


def get_grid():
    if state["grid"] is None:
        if not GRID_PATH.exists():
            raise HTTPException(503, "data/processed/grid_all.csv introuvable : exécuter les notebooks 01 à 05.")
        state["grid"] = pd.read_csv(GRID_PATH, index_col=0, parse_dates=True)
    return state["grid"]


def get_agent():
    if state["agent"] is None:
        state["agent"] = PricingAgent.load()
    return state["agent"]


def test_period(days=None):
    grid = get_grid()
    test = grid[grid["period"] == "test"]
    return test.iloc[:24 * days] if days else test


def series(index):
    return [t.strftime("%Y-%m-%d %H:%M") for t in index]


def rounded(values, digits=2):
    return [round(float(v), digits) for v in values]


# --------------------------------------------------------------------------
# Routes de l'API
# --------------------------------------------------------------------------
@app.get("/api/summary")
def summary():
    """Chiffres clés : période, configuration du micro-réseau, qualité des prévisions."""
    grid, test = get_grid(), test_period()
    return {
        "common_hours": len(grid),
        "train_hours": int((grid["period"] == "train").sum()),
        "test_hours": len(test),
        "test_start": str(test.index.min()), "test_end": str(test.index.max()),
        "microgrid": config.MICROGRID,
        "tariffs": config.PRICING["tariffs"],
        "forecast_metrics": {
            "demand": compute_metrics(test["demand_true"], test["demand_pred"]),
            "pv": compute_metrics(test["pv_true"], test["pv_pred"]),
        },
        "deficit_share_test": float((test["demand_true"] > test["pv_true"]).mean()),
    }


@app.get("/api/forecast")
def forecast(days: int = 7):
    """Demande et production PV, réelles et prévues, sur les premiers jours du test."""
    test = test_period(days)
    return {"time": series(test.index),
            "demand_true": rounded(test["demand_true"]), "demand_pred": rounded(test["demand_pred"]),
            "pv_true": rounded(test["pv_true"]), "pv_pred": rounded(test["pv_pred"])}


class SimulationRequest(BaseModel):
    elasticity: float = Field(config.MICROGRID["demand_elasticity"], ge=-1.0, le=0.0)
    battery_kwh: float = Field(config.MICROGRID["battery_capacity_kwh"], ge=1, le=1000)
    pv_peak_kw: float = Field(config.MICROGRID["pv_peak_kw"], ge=0, le=500)
    days_shown: int = Field(3, ge=1, le=17)


@app.post("/api/simulate")
def simulate(req: SimulationRequest):
    """Simule les 3 stratégies sur toute la période de test avec les paramètres choisis."""
    test = test_period().copy()
    # La production PV est proportionnelle à la puissance crête (formule linéaire)
    scale = req.pv_peak_kw / config.MICROGRID["pv_peak_kw"]
    test[["pv_true", "pv_pred"]] *= scale
    params = {**config.MICROGRID, "demand_elasticity": req.elasticity,
              "battery_capacity_kwh": req.battery_kwh, "pv_peak_kw": req.pv_peak_kw}

    # Les tarifs de l'agent ne dépendent que des prévisions : on les calcule en une seule fois
    # (mêmes tarifs que make_policy() heure par heure, mais beaucoup plus rapide)
    X = make_features(test["demand_pred"], test["pv_pred"], test.index.hour)
    agent_prices = iter(get_agent().predict_price(X))
    policies = {"Tarif fixe": fixed_price_policy(), "Heures pleines / creuses": rule_based_policy(),
                "Réseau dense": lambda s: float(next(agent_prices))}
    table, results = compare_policies(test, policies, params)
    reference = table.loc["Tarif fixe"]
    change = 100 * (table - reference) / reference

    n = 24 * req.days_shown
    return {
        "indicators": list(table.columns),
        "table": {name: {k: round(float(v), 3) for k, v in row.items()} for name, row in table.iterrows()},
        "change_vs_fixed": {name: {k: round(float(v), 2) for k, v in row.items()} for name, row in change.iterrows()},
        "time": series(test.index[:n]),
        "series": {name: {col: rounded(r[col].iloc[:n], 3) for col in ["price", "demand", "pv", "grid_import", "soc"]}
                   for name, r in results.items()},
    }


class TariffRequest(BaseModel):
    demand_forecast: float = Field(..., ge=0, description="Demande prévue de l'heure (kWh)")
    pv_forecast: float = Field(..., ge=0, description="Production PV prévue de l'heure (kWh)")
    hour: int = Field(..., ge=0, le=23)


@app.post("/api/tariff")
def tariff(req: TariffRequest):
    """Tarif proposé par le réseau dense pour une heure donnée (sans historique : écarts à la moyenne 24 h = 0)."""
    agent = get_agent()
    X = make_features([req.demand_forecast], [req.pv_forecast], [req.hour])
    probas = agent.model.predict(agent.scaler.transform(X), verbose=0)[0]
    best = int(np.argmax(probas))
    return {"tariff": agent.tariffs[best],
            "probabilities": {f"{t:.2f}": round(float(p), 4) for t, p in zip(agent.tariffs, probas)},
            "balance_kwh": round(req.demand_forecast - req.pv_forecast, 2)}


# --------------------------------------------------------------------------
# Frontend (fichiers statiques)
# --------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
