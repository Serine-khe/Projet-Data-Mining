# Architecture

## Vue d'ensemble

```
 data/raw ──► load_data ──► preprocess ──► features
                                              │
              ┌───────────────────────────────┴───────────────────────┐
              ▼                                                       ▼
   Dev A : ConsumptionForecaster                         Dev B : SolarForecaster
   (régression linéaire, lags + calendrier)              (LSTM, irradiance + météo)
              │  demande prévue                                       │  production prévue
              └──────────────────────┬────────────────────────────────┘
                                     ▼
                        Dev C : PricingAgent (réseau dense)
                        état = [demande, solaire, SoC, heure]
                                     │  prix €/kWh
                                     ▼
                        MicrogridEnv (simulation)
                        foyers + PV + batterie + réseau principal
                                     │
                                     ▼
                        evaluation (metrics, plots) ──► reports/figures
```

## Modules

| Module | Responsabilité |
|---|---|
| `src/config.py` | Chemins et hyperparamètres centralisés |
| `src/data/load_data.py` | Lecture des CSV bruts, renommage des colonnes |
| `src/data/preprocess.py` | Nettoyage, agrégation horaire, découpage chronologique |
| `src/data/features.py` | Variables calendaires cycliques, retards, fenêtres LSTM |
| `src/models/linear_regression.py` | Prévision de consommation (scikit-learn, `.pkl`) |
| `src/models/lstm_solar.py` | Prévision solaire (PyTorch, `.pt`) |
| `src/models/pricing_dense.py` | Agent de prix (PyTorch, `.pt`) |
| `src/simulation/microgrid_env.py` | Environnement pas-à-pas (bilan énergétique, batterie, élasticité) |
| `src/simulation/run_simulation.py` | Données synthétiques, comparaison de politiques |
| `src/evaluation/` | MAE, RMSE, MAPE, R², graphiques |
| `src/pipeline.py` | Orchestration bout en bout |
| `app/demo.py` | Interface Streamlit |

## Interfaces entre modèles

- `ConsumptionForecaster.predict(df)` → `np.ndarray` (kWh/h), `df` contient `timestamp`, `consumption_kwh`.
- `SolarForecaster.predict(df)` → `np.ndarray` (kWh/h), `df` contient `timestamp`, `ghi`, `temperature`, `pv_power`.
- `PricingAgent.act(state)` → `float` (€/kWh), `state = [demand_kwh, solar_kwh, soc, hour_sin, hour_cos]`.
- `MicrogridEnv.step(price)` → `StepResult(state, reward, done, info)`.

## Modèle du micro-réseau

À chaque heure *t* :

1. La demande réagit au prix : `D = D_base · (prix / prix_ref) ^ élasticité`.
2. Le solaire couvre la demande ; le déficit ou surplus net est `D − PV`.
3. La batterie se décharge (déficit) ou se charge (surplus), limitée par sa puissance, sa capacité et son rendement.
4. Le reste est importé ou exporté sur le réseau principal.
5. Récompense = `D · prix + export · prix_export − import · prix_import`.

## Conventions

- Découpage train/test chronologique, jamais aléatoire.
- Les modèles sauvegardés vont dans `models_saved/` (non versionné).
- Les données brutes restent hors du dépôt ; `data/simulated/` permet de tout exécuter sans elles.
