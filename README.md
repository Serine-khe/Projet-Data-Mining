# Smart Grid Project

Prévision et pilotage d'un micro-réseau électrique intelligent à l'aide de trois modèles enchaînés :

| Modèle | Rôle | Responsable | Fichier |
|---|---|---|---|
| Régression linéaire | Prévision de la consommation des foyers (London Smart Meters) | Serine | `src/models/linear_regression.py` |
| LSTM | Prévision de la production solaire à partir de l'irradiance | Hadjer | `src/models/lstm_solar.py` |
| Réseau dense | Agent de tarification dynamique de l'énergie | Asma | `src/models/pricing_dense.py` |

Les prévisions des deux premiers modèles alimentent l'agent de tarification, évalué dans un environnement de simulation de micro-réseau (`src/simulation/`).

## Structure

```
smart-grid-project/
├── data/            # raw (non versionné), processed, simulated
├── notebooks/       # EDA et prototypage des modèles
├── src/             # code source (données, modèles, simulation, évaluation, pipeline)
├── models_saved/    # modèles entraînés (.pkl, .pt)
├── tests/           # tests unitaires (pytest)
├── app/             # démonstration Streamlit
├── reports/         # rapport technique et figures
└── docs/            # documentation d'architecture
```

Voir [docs/architecture.md](docs/architecture.md) pour le détail du flux de données.

## Installation

```bash
git clone <url-du-repo>
cd smart-grid-project
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

## Données

Les données brutes ne sont pas versionnées. Les placer dans :

- `data/raw/london_smart_meters/` — fichiers CSV du jeu *Smart meters in London* (colonnes `LCLid`, `DateTime`, `KWH/hh (per half hour)`).
- `data/raw/solar_irradiance/` — fichiers CSV d'irradiance (colonnes `timestamp`, `ghi`, `temperature`, et optionnellement `pv_power`).

Sans données réelles, le projet fonctionne sur des données synthétiques :

```bash
python -m src.simulation.run_simulation --generate-only
```

## Utilisation

```bash
# Pipeline complet : entraînement des 3 modèles + simulation
python -m src.pipeline

# Simulation seule avec les modèles sauvegardés
python -m src.simulation.run_simulation

# Démonstration interactive
streamlit run app/demo.py

# Tests
pytest
```

## Équipe

- Serine (Dev A) — régression linéaire (consommation)
- Hadjer (Dev B) — LSTM (solaire)
- Asma (Dev C) — agent de tarification (réseau dense)

## Répartition du travail

Chaque fichier à compléter commence par un bloc `PROMPT` qui décrit ce qu'il faut écrire, et contient des `# TODO (prénom)`. Pour retrouver ses tâches :

```bash
grep -rn "TODO (Serine)" src notebooks
```

| Qui | Fichiers | Test de validation |
|---|---|---|
| Serine | `src/models/linear_regression.py`, `_demand_response` dans `microgrid_env.py`, notebooks 01 et 03 | `pytest tests/test_models.py -k linear` |
| Hadjer | `src/models/lstm_solar.py`, `_dispatch_battery` dans `microgrid_env.py`, notebooks 02 et 04 | `pytest tests/test_models.py -k lstm` |
| Asma | `src/models/pricing_dense.py`, `_reward` dans `microgrid_env.py`, `src/pipeline.py`, notebook 05 | `pytest tests/test_models.py -k pricing` |
| Équipe | `tests/test_simulation.py` (après les 3 méthodes de l'environnement), `app/demo.py`, rapport | `pytest` |

Tant que les TODO ne sont pas écrits, les tests correspondants échouent avec `NotImplementedError` : c'est normal.
