"""Démonstration Streamlit du micro-réseau intelligent.

Prérequis : notebooks 01 à 05 exécutés (ou `python -m src.pipeline`), pour avoir
data/processed/grid_all.csv et les modèles dans models_saved/.

Lancer depuis la racine du projet : streamlit run app/demo.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src import config  # noqa: E402
from src.models.linear_regression import compute_metrics  # noqa: E402
from src.simulation.microgrid_env import fixed_price_policy, rule_based_policy  # noqa: E402
from src.simulation.run_simulation import compare_policies  # noqa: E402

st.set_page_config(page_title="Micro-réseau intelligent", layout="wide")
st.title("Micro-réseau intelligent : prévision et tarification dynamique")
st.caption("Consommation : London Smart Meters (mesurée). Production PV : estimée à partir de l'irradiance "
           "HI-SEAS (Hawaï, dates décalées 2016 → 2013). Résultats de tarification : simulés.")

grid_path = config.PROCESSED_DIR / "grid_all.csv"
if not grid_path.exists():
    st.error("Fichier data/processed/grid_all.csv introuvable : exécuter d'abord les notebooks 01 à 05.")
    st.stop()


@st.cache_data
def load_grid():
    return pd.read_csv(grid_path, index_col=0, parse_dates=True)


@st.cache_resource
def load_agent():
    from src.models.pricing_dense import PricingAgent
    return PricingAgent.load()


grid = load_grid()
grid_test = grid[grid["period"] == "test"]

with st.sidebar:
    st.header("Paramètres de la simulation")
    n_days = st.slider("Nombre de jours de test affichés", 1, max(1, len(grid_test) // 24), 3)
    elasticity = st.slider("Élasticité de la demande au prix", -0.5, 0.0, config.MICROGRID["demand_elasticity"], 0.05)
    battery = st.slider("Capacité de la batterie (kWh)", 1, 500, int(config.MICROGRID["battery_capacity_kwh"]))
params = {**config.MICROGRID, "demand_elasticity": elasticity, "battery_capacity_kwh": float(battery)}

tab_forecast, tab_sim = st.tabs(["Prévisions", "Simulation"])

with tab_forecast:
    st.subheader("Qualité des prévisions sur la période de test")
    metrics = pd.DataFrame({
        "Demande (régression linéaire)": compute_metrics(grid_test["demand_true"], grid_test["demand_pred"]),
        "Production PV estimée (LSTM)": compute_metrics(grid_test["pv_true"], grid_test["pv_pred"]),
    }).T.drop(columns="MAPE_%")
    st.dataframe(metrics.style.format("{:.3f}"))
    view = grid_test.iloc[:24 * n_days]
    st.line_chart(view[["demand_true", "demand_pred"]].rename(columns={"demand_true": "Demande réelle", "demand_pred": "Demande prévue"}))
    st.line_chart(view[["pv_true", "pv_pred"]].rename(columns={"pv_true": "PV (irradiance mesurée)", "pv_pred": "PV (irradiance prévue)"}))

with tab_sim:
    policies = {"Tarif fixe": fixed_price_policy(), "Heures pleines / creuses": rule_based_policy(),
                "Réseau dense": load_agent().make_policy()}
    table, results = compare_policies(grid_test, policies, params)
    st.subheader("Indicateurs sur toute la période de test")
    st.dataframe(table.T.style.format("{:.2f}"))

    choice = st.selectbox("Stratégie à afficher", list(policies), index=2)
    r = results[choice].iloc[:24 * n_days]
    fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    ax[0].plot(r.index, r["demand"], label="Demande"); ax[0].plot(r.index, r["pv"], label="Production PV")
    ax[0].plot(r.index, r["grid_import"], label="Import réseau"); ax[0].set_ylabel("kWh"); ax[0].legend(ncol=3)
    ax[1].plot(r.index, r["soc"], color="tab:green"); ax[1].set_ylabel("Charge batterie")
    ax[2].step(r.index, r["price"], where="post", color="tab:red"); ax[2].set_ylabel("Tarif (€/kWh)")
    plt.tight_layout()
    st.pyplot(fig)
