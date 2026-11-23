"""
pricing_dense.py
================
Agent de tarification dynamique (Dev C) : réseau de neurones dense (Keras).

Entrées (pour chaque heure) : demande prévue (régression linéaire), production PV
prévue (LSTM), écart demande - production, cet écart et la demande comparés à leur
moyenne sur les 24 dernières heures (passé uniquement), heure (sin / cos).
Sortie : un des niveaux de tarif de config.PRICING["tariffs"] (classification).

Étiquettes d'entraînement (tarif « idéal ») : pour chaque journée, on choisit les tarifs
horaires qui minimisent le déséquilibre total
    somme_h |D_h(p_h) - PV_h|     avec  D(p) = D * (p / p_ref) ** e  (réaction au prix)
sous la contrainte : prix moyen de la journée = p_ref (0,20 €/kWh, comme le tarif fixe).
On ne cherche donc pas à vendre plus cher, mais à répartir le prix dans la journée :
cher quand il manque beaucoup d'énergie, bon marché quand il y en a en trop.
La contrainte est traitée avec un multiplicateur mu (recherche sur une grille).

Le réseau dense apprend à reproduire ces tarifs en ne voyant que l'heure courante.
C'est de l'apprentissage supervisé, pas de l'apprentissage par renforcement.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src import config

FEATURES = ["demand_forecast", "pv_forecast", "balance", "balance_vs_24h", "demand_vs_24h",
            "hour_sin", "hour_cos"]


def make_features(demand, pv, hours) -> np.ndarray:
    """Matrice d'entrée de l'agent (une ligne par heure, dans l'ordre chronologique).
    Les moyennes sur 24 h n'utilisent que l'heure courante et les heures passées."""
    demand, pv = pd.Series(np.asarray(demand, float)), pd.Series(np.asarray(pv, float))
    hours = np.asarray(hours)
    balance = demand - pv
    balance_vs_24h = balance - balance.rolling(24, min_periods=1).mean()
    demand_vs_24h = demand - demand.rolling(24, min_periods=1).mean()
    return np.column_stack([demand, pv, balance, balance_vs_24h, demand_vs_24h,
                            np.sin(2 * np.pi * hours / 24), np.cos(2 * np.pi * hours / 24)])


def best_day_tariffs(demand, pv, tariffs=config.PRICING["tariffs"], params=config.MICROGRID) -> np.ndarray:
    """Indices des tarifs idéaux pour UNE journée (voir l'explication en haut du fichier)."""
    demand, pv = np.asarray(demand, float), np.asarray(pv, float)
    prices = np.asarray(tariffs, float)
    p_ref, e = params["reference_price"], params["demand_elasticity"]

    new_demand = demand[:, None] * (prices[None, :] / p_ref) ** e     # (heures, tarifs)
    imbalance = np.abs(new_demand - pv[:, None])                      # kWh
    price_gap = (prices - p_ref) / p_ref

    best_choice, best_score = None, None
    for mu in np.linspace(-3, 3, 601):
        # mu > 0 : augmenter le prix "coûte" ; on ajuste mu pour que la moyenne vaille p_ref
        choice = (imbalance + mu * demand.mean() * price_gap).argmin(axis=1)
        gap = abs(prices[choice].mean() - p_ref)
        total = imbalance[np.arange(len(choice)), choice].sum()
        score = (round(gap, 4), total)
        if best_score is None or score < best_score:
            best_choice, best_score = choice, score
    return best_choice


def best_tariff_index(demand, pv, days, tariffs=config.PRICING["tariffs"]) -> np.ndarray:
    """Étiquettes pour toute une période : best_day_tariffs appliqué jour par jour."""
    demand, pv, days = np.asarray(demand, float), np.asarray(pv, float), np.asarray(days)
    labels = np.empty(len(demand), dtype=int)
    for day in np.unique(days):
        mask = days == day
        labels[mask] = best_day_tariffs(demand[mask], pv[mask], tariffs)
    return labels


class PricingAgent:
    """Petit réseau dense : 5 entrées -> couches cachées ReLU -> softmax sur les tarifs."""

    def __init__(self, tariffs=None, hidden_units=None, learning_rate=None):
        self.tariffs = list(tariffs or config.PRICING["tariffs"])
        self.hidden_units = list(hidden_units or config.PRICING["hidden_units"])
        self.learning_rate = learning_rate or config.PRICING["learning_rate"]
        self.scaler = StandardScaler()
        self.model = None
        self.history = None

    def build(self):
        from tensorflow import keras
        from tensorflow.keras import layers

        keras.utils.set_random_seed(config.RANDOM_SEED)   # graines Python, NumPy et TensorFlow
        model = keras.Sequential(name="pricing_dense")
        model.add(layers.Input(shape=(len(FEATURES),)))
        for units in self.hidden_units:
            model.add(layers.Dense(units, activation="relu"))
        model.add(layers.Dense(len(self.tariffs), activation="softmax"))
        model.compile(optimizer=keras.optimizers.Adam(self.learning_rate),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        self.model = model
        return model

    def fit(self, X, y, epochs=None, batch_size=None, validation_split=0.2, verbose=0):
        """X : make_features(...), y : indices de tarif (best_tariff_index).
        Les 20 % les plus récents servent de validation (Keras prend la fin du tableau)."""
        X_scaled = self.scaler.fit_transform(X)
        if self.model is None:
            self.build()
        self.history = self.model.fit(X_scaled, np.asarray(y), validation_split=validation_split,
                                      epochs=epochs or config.PRICING["epochs"],
                                      batch_size=batch_size or config.PRICING["batch_size"],
                                      verbose=verbose)
        return self.history

    def predict_index(self, X) -> np.ndarray:
        probas = self.model.predict(self.scaler.transform(X), verbose=0)
        return probas.argmax(axis=1)

    def predict_price(self, X) -> np.ndarray:
        return np.array(self.tariffs)[self.predict_index(X)]

    def make_policy(self):
        """Politique pour MicrogridEnv.run : un état -> un prix.
        Elle garde en mémoire les prévisions des 24 dernières heures (une nouvelle à chaque simulation)."""
        past = {"demand": [], "pv": [], "hour": []}

        def policy(state):
            past["demand"].append(state["demand_forecast"])
            past["pv"].append(state["pv_forecast"])
            past["hour"].append(state["hour"])
            for key in past:
                del past[key][:-24]
            X = make_features(past["demand"], past["pv"], past["hour"])[-1:]
            probas = self.model(self.scaler.transform(X), training=False).numpy()  # plus rapide que predict()
            return float(self.tariffs[probas.argmax()])
        return policy

    def save(self, directory=config.MODELS_DIR, name="pricing_dense"):
        os.makedirs(directory, exist_ok=True)
        self.model.save(os.path.join(directory, f"{name}.keras"))
        joblib.dump(self.scaler, os.path.join(directory, f"{name}_scaler.pkl"))
        meta = {"tariffs": self.tariffs, "hidden_units": self.hidden_units,
                "learning_rate": self.learning_rate}
        with open(os.path.join(directory, f"{name}_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, directory=config.MODELS_DIR, name="pricing_dense"):
        from tensorflow import keras

        with open(os.path.join(directory, f"{name}_meta.json")) as f:
            agent = cls(**json.load(f))
        agent.scaler = joblib.load(os.path.join(directory, f"{name}_scaler.pkl"))
        agent.model = keras.models.load_model(os.path.join(directory, f"{name}.keras"))
        return agent
