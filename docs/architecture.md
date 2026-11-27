# Architecture

## Flux de données

```
data/raw/london_smart_meters/elec_block.csv ─┐
data/raw/london_smart_meters/weather_*.csv  ─┼─► notebook 01 / preprocess.py ─► consumption_hourly.csv
                                             │
data/raw/solar_irradiance/SolarPrediction.csv ─► notebook 02 / preprocess.py ─► solar_hourly.csv

consumption_hourly.csv ─► Régression Lasso (notebook 03) ─► demande prévue par foyer × 100 foyers
solar_hourly.csv       ─► LSTM (notebook 04) ─► irradiance prévue ─► production PV ESTIMÉE
                                                       (dates décalées 2016 -> 2013)
                                    │
                                    ▼  jointure sur les heures communes (sept.-déc. 2013)
                    Réseau dense (notebook 05) : tarif parmi {0,10 ... 0,30} €/kWh
                                    │
                                    ▼
                    MicrogridEnv : demande RÉELLE × réaction au prix, PV réelle,
                    batterie, import / export ─► indicateurs (run_simulation.summarize)
```

## Modules

| Module | Responsabilité |
|---|---|
| `src/config.py` | Chemins, découpage temporel, paramètres du micro-réseau et de l'agent |
| `src/data/load_data.py` | Lecture des fichiers bruts et traités |
| `src/data/preprocess.py` | Mêmes étapes que les notebooks 01 et 02 (blocs -> horaire, UTC -> heure d'Hawaï) |
| `src/models/linear_regression.py` | Variables, découpage, régression linéaire / Ridge / Lasso, baselines |
| `src/models/lstm_solar.py` | Fenêtres glissantes, LSTM Keras, persistance, formule PV estimée |
| `src/models/pricing_dense.py` | Tarifs idéaux (étiquettes), variables de l'agent, réseau dense Keras |
| `src/simulation/microgrid_env.py` | Simulation heure par heure, tarif fixe, règle heures pleines / creuses |
| `src/simulation/run_simulation.py` | Simulation d'une stratégie, indicateurs, comparaison |
| `src/pipeline.py` | Enchaînement complet (utilisé aussi par le notebook 05) |
| `app/backend/api.py` | Interface web, backend : API REST FastAPI (prévisions, simulation, tarif) |
| `app/frontend/` | Interface web, frontend : page HTML / CSS / JavaScript (Chart.js) qui appelle l'API |
| `app/demo.py` | Démonstration Streamlit |

## Interfaces

- `ConsumptionModel.predict(X)` -> kWh par foyer, avec `X, y = build_features(df)`.
- `SolarLSTM.predict(df)` -> irradiance (W/m²), forme `(n, 1)`, alignée sur `target_index(df)`.
- `estimate_pv_power(G)` -> production PV estimée (kW).
- `best_tariff_index(demand, pv, days)` -> indice du tarif idéal pour chaque heure.
- `PricingAgent.make_policy()` -> fonction `state -> prix`, utilisée par `MicrogridEnv.run`.
- `state = {"demand_forecast", "pv_forecast", "soc", "hour"}` : la politique ne voit que des prévisions.

## Modèle du micro-réseau (une heure)

1. Prix choisi par la politique à partir des prévisions.
2. Demande : `D = D_réelle × (prix / 0,20) ** (-0,2)`.
3. Le solaire couvre la demande ; écart `D - PV`.
4. Batterie : décharge si déficit, charge si surplus (50 kW max, 200 kWh, rendement 95 %).
5. Le reste est importé du réseau principal ou exporté.

## Étiquettes de l'agent

Pour chaque journée : tarifs qui minimisent `somme |D_h(p_h) - PV_h|` avec un prix moyen de 0,20 €/kWh (multiplicateur `mu` cherché sur une grille). Le réseau dense apprend ensuite à les reproduire à partir de l'heure courante et des 24 heures passées (apprentissage supervisé).

## Conventions

- Découpage chronologique 70 / 15 / 15, configurations départagées sur la validation (des scores de test ont toutefois été consultés pendant le développement, voir le rapport, section 11).
- Normalisations ajustées sur l'entraînement uniquement ; graines fixées (résultats reproductibles).
- Données brutes et modèles sauvegardés hors du dépôt.
