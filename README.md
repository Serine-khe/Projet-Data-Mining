# Smart Grid Project — Régulation énergétique et micro-réseau intelligent

Projet de Data Mining : équilibrer l'offre solaire et la demande des foyers d'un micro-réseau à l'aide de trois modèles enchaînés.

| Modèle | Rôle | Fichier |
|---|---|---|
| Régression linéaire (Lasso) | Prévision de la consommation des foyers (London Smart Meters) | `src/models/linear_regression.py` |
| LSTM (Keras) | Prévision de l'irradiance, puis de la production photovoltaïque **estimée** | `src/models/lstm_solar.py` |
| Réseau de neurones dense (Keras) | Choix d'un tarif horaire parmi 5 niveaux | `src/models/pricing_dense.py` |

Les prévisions des deux premiers modèles alimentent l'agent de tarification, évalué dans un micro-réseau **simulé** (100 foyers, 150 kWc de panneaux, batterie de 200 kWh) et comparé à un tarif fixe et à une règle heures pleines / heures creuses.

Rapport : [`reports/rapport_technique.docx`](reports/rapport_technique.docx) et [`.pdf`](reports/rapport_technique.pdf). Architecture : [`docs/architecture.md`](docs/architecture.md).

## Structure

```
smart-grid-project/
├── data/
│   ├── raw/             # fichiers bruts (non versionnés, voir « Données »)
│   ├── processed/       # séries horaires nettoyées (notebooks 01, 02 et 05)
│   └── simulated/       # résultats détaillés des simulations
├── notebooks/           # 01-02 : exploration, 03 : régression, 04 : LSTM, 05 : tarification + workflow complet
├── src/
│   ├── config.py        # chemins, paramètres du micro-réseau et de l'agent
│   ├── data/            # chargement et prétraitement
│   ├── models/          # les trois modèles
│   ├── simulation/      # environnement du micro-réseau, comparaison des stratégies
│   ├── evaluation/      # métriques et graphiques
│   └── pipeline.py      # enchaînement des 3 modèles
├── models_saved/        # modèles entraînés (non versionnés)
├── tests/               # tests unitaires (pytest)
├── app/demo.py          # démonstration Streamlit
├── reports/             # rapport technique, figures, tableau de résultats
└── docs/architecture.md
```

## Installation

```bash
git clone https://github.com/Serine-khe/Projet-Data-Mining.git
cd Projet-Data-Mining
python -m venv .venv
source .venv/Scripts/activate      # Git Bash sous Windows ; Linux / macOS : source .venv/bin/activate
pip install -r requirements.txt
```

Testé avec Python 3.10 et TensorFlow 2.21 (CPU).

## Données

Les fichiers bruts ne sont pas dans le dépôt (trop volumineux). Les télécharger depuis Kaggle (compte gratuit nécessaire) et les placer ainsi :

```
data/raw/london_smart_meters/elec_block.csv              # consommation moyenne par foyer, 112 blocs, pas de 30 min
data/raw/london_smart_meters/weather_hourly_darksky.csv  # météo horaire de Londres
data/raw/solar_irradiance/SolarPrediction.csv            # irradiance HI-SEAS (Hawaï), sept.-déc. 2016
```

- Consommation et météo : jeu Kaggle *Processed dataset of smart meters in London* (https://www.kaggle.com/datasets/patrick0302/10qs-load-forecasting-london-dataset), qui contient les deux fichiers. Il est construit à partir de *Smart meters in London* (https://www.kaggle.com/datasets/jeanmidev/smart-meters-in-london), issu du projet Low Carbon London (https://data.london.gov.uk/dataset/smartmeter-energy-use-data-in-london-households). `elec_block.csv` est une version agrégée par blocs de ces données.
- Irradiance : jeu Kaggle *Solar Radiation Prediction* (https://www.kaggle.com/datasets/dronio/SolarEnergy).

Les deux jeux n'ont pas de période commune : l'irradiance de 2016 est décalée en 2013 (même mois, jour et heure). La production photovoltaïque n'est pas mesurée : elle est estimée par `P = 150 kWc × G / 1000 × 0,8`.

## Utilisation

Exécuter les notebooks dans l'ordre (depuis le dossier `notebooks/`) :

1. `01_eda_consumption.ipynb` → `data/processed/consumption_hourly.csv`
2. `02_eda_solar.ipynb` → `data/processed/solar_hourly.csv`
3. `03_linear_regression.ipynb` → `models_saved/linear_regression.pkl`
4. `04_lstm_solar.ipynb` → `models_saved/lstm_solar.keras`
5. `05_pricing_agent.ipynb` → agent de tarification, simulation et comparaison des stratégies

En ligne de commande (depuis la racine du projet) :

```bash
python -m src.data.preprocess          # recrée data/processed à partir de data/raw (comme les notebooks 01 et 02)
python -m src.pipeline                 # enchaîne les 3 modèles (régression et LSTM sauvegardés)
python -m src.pipeline --retrain       # ré-entraîne aussi la régression et le LSTM
python -m src.simulation.run_simulation
streamlit run app/demo.py              # démonstration (après le notebook 05 ou la pipeline)
pytest                                 # tests unitaires
```

## Principaux résultats (jeux de test)

| | Résultat |
|---|---|
| Consommation (régression Lasso) | MAE 0,016 kWh, RMSE 0,022 kWh, **R² 0,981** (meilleure référence : R² 0,947) |
| Irradiance (LSTM, 1 h à l'avance) | RMSE 113 W/m², **R² 0,820** (persistance : R² 0,807) |
| Tarification (réseau dense, 14-31 déc., simulé) | déséquilibre −5,2 %, import −3,7 %, pic d'import −7,8 %, mais prix moyen +8,7 % |

Des scores de test ont été consultés pendant la mise au point du LSTM et de l'agent : ces deux résultats sont probablement un peu optimistes. Détails, limites et discussion dans le rapport.

## Équipe

Serine, Hadjer et Asma.
