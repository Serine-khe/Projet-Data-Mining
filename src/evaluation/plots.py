"""Graphiques d'évaluation."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config


def _save(fig, name: str | None):
    if name:
        config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(config.FIGURES_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
    return fig


def plot_forecast(y_true, y_pred, title: str = "Prévision vs réel", n: int = 24 * 7,
                  ylabel: str = "kWh", name: str | None = None):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(np.asarray(y_true)[:n], label="Réel")
    ax.plot(np.asarray(y_pred)[:n], label="Prévision", linestyle="--")
    ax.set(title=title, xlabel="Heure", ylabel=ylabel)
    ax.legend()
    return _save(fig, name)


def plot_training_history(history: list[float], title: str = "Courbe d'apprentissage", name: str | None = None):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(history) + 1), history)
    ax.set(title=title, xlabel="Époque", ylabel="Perte (MSE)")
    return _save(fig, name)


def plot_simulation(results: pd.DataFrame, n: int = 24 * 3, name: str | None = None):
    r = results.iloc[:n]
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(r["demand"], label="Demande")
    axes[0].plot(r["solar"], label="Solaire")
    axes[0].plot(r["grid_import"], label="Import réseau")
    axes[0].set_ylabel("kWh")
    axes[0].legend()
    axes[1].plot(r["soc"], color="tab:green")
    axes[1].set_ylabel("État de charge")
    axes[2].step(range(len(r)), r["price"], color="tab:red")
    axes[2].set(ylabel="Prix (€/kWh)", xlabel="Heure")
    fig.suptitle("Simulation du micro-réseau")
    return _save(fig, name)
