"""Démonstration Streamlit du micro-réseau intelligent.

Lancer depuis la racine du projet : ``streamlit run app/demo.py``

==============================================================================
PROMPT — livrable « démonstration », à compléter par l'équipe à la fin
==============================================================================
La base actuelle compare un prix fixe et l'agent dense sur des données
synthétiques. À ajouter :
  Serine — un onglet « Consommation » : courbe prévision vs réel, métriques.
  Hadjer — un onglet « Solaire » : courbe prévision vs réel, métriques.
  Asma   — l'onglet « Simulation » : utiliser les vraies données de test
           (data/processed) au lieu de generate_synthetic_data, expliquer
           comment le prix réagit.
Astuce : st.tabs(["Consommation", "Solaire", "Simulation"]).
==============================================================================
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from src import config  # noqa: E402
from src.evaluation.plots import plot_simulation  # noqa: E402
from src.simulation.microgrid_env import fixed_price_policy  # noqa: E402
from src.simulation.run_simulation import generate_synthetic_data, simulate, summarize  # noqa: E402

st.set_page_config(page_title="Smart Grid", layout="wide")
st.title("Micro-réseau intelligent : prévision et tarification")

with st.sidebar:
    st.header("Paramètres")
    n_days = st.slider("Jours simulés", 3, 60, 14)
    fixed_price = st.slider("Prix fixe de référence (€/kWh)", 0.05, 0.40, config.MICROGRID["reference_price"])
    battery = st.slider("Capacité batterie (kWh)", 0, 500, int(config.MICROGRID["battery_capacity_kwh"]))
    config.MICROGRID["battery_capacity_kwh"] = max(battery, 1)

consumption, solar = generate_synthetic_data(n_days)
demand, pv = consumption["consumption_kwh"].to_numpy(), solar["pv_power"].to_numpy()
hours = consumption["timestamp"].dt.hour.to_numpy()

policies = {"Prix fixe": fixed_price_policy(fixed_price)}
model_path = config.MODELS_DIR / "pricing_dense.pt"
if model_path.exists():
    from src.models.pricing_dense import PricingAgent
    policies["Agent dense"] = PricingAgent.load(model_path).act
else:
    st.info("Aucun agent entraîné trouvé. Lancer `python -m src.pipeline` pour l'entraîner.")

cols = st.columns(len(policies))
for col, (name, policy) in zip(cols, policies.items()):
    res = simulate(demand, pv, policy, hours)
    summary = summarize(res)
    with col:
        st.subheader(name)
        st.metric("Profit total (€)", f"{summary['profit_total']:.0f}")
        st.metric("Import réseau (kWh)", f"{summary['import_total_kwh']:.0f}")
        st.metric("Autoconsommation", f"{summary['autoconsommation']:.1%}")
        st.pyplot(plot_simulation(res))
