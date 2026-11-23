"""Environnement de simulation d'un micro-réseau : foyers + panneaux solaires + batterie + réseau.

À chaque heure :
  1. la politique de tarification choisit un prix à partir des PRÉVISIONS
     (demande prévue, production prévue, état de la batterie, heure) ;
  2. les foyers réagissent au prix (élasticité) : D = D_base * (prix / prix_ref) ** e ;
  3. le solaire RÉEL couvre d'abord la demande ;
  4. la batterie absorbe le surplus ou couvre le déficit (limites de puissance et de capacité) ;
  5. le reste est importé du réseau principal ou exporté vers lui.
"""
import numpy as np
import pandas as pd

from src import config


class MicrogridEnv:
    def __init__(self, demand, pv, demand_forecast=None, pv_forecast=None, hours=None,
                 params=config.MICROGRID):
        self.demand = np.asarray(demand, dtype=float)       # demande de base réelle (kWh/h)
        self.pv = np.asarray(pv, dtype=float)               # production réelle (kWh/h)
        if len(self.demand) != len(self.pv):
            raise ValueError("demand et pv doivent avoir la même longueur")
        # Sans prévisions, la politique voit les vraies valeurs
        self.demand_forecast = self.demand if demand_forecast is None else np.asarray(demand_forecast, dtype=float)
        self.pv_forecast = self.pv if pv_forecast is None else np.asarray(pv_forecast, dtype=float)
        self.hours = np.arange(len(self.demand)) % 24 if hours is None else np.asarray(hours)
        self.p = params
        self.reset()

    def reset(self):
        self.t = 0
        self.soc = self.p["initial_soc"]
        self.history = []
        return self.state()

    def state(self):
        """Ce que la politique de tarification peut observer à l'heure t."""
        t = min(self.t, len(self.demand) - 1)
        return {"demand_forecast": self.demand_forecast[t], "pv_forecast": self.pv_forecast[t],
                "soc": self.soc, "hour": self.hours[t]}

    def demand_response(self, base_demand, price):
        """Demande des foyers au prix donné (élasticité constante)."""
        return base_demand * (price / self.p["reference_price"]) ** self.p["demand_elasticity"]

    def dispatch_battery(self, net):
        """net > 0 : déficit, net < 0 : surplus.
        Retourne (flux batterie, import réseau, export réseau) ; flux > 0 = charge."""
        capacity = self.p["battery_capacity_kwh"]
        power = self.p["battery_power_kw"]
        eff = self.p["battery_efficiency"]
        energy = self.soc * capacity

        if net > 0:   # décharge pour couvrir le déficit
            discharge = min(net, power, energy * eff)
            energy -= discharge / eff
            battery_flow, grid_import, grid_export = -discharge, net - discharge, 0.0
        else:         # charge avec le surplus
            charge = min(-net, power, (capacity - energy) / eff)
            energy += charge * eff
            battery_flow, grid_import, grid_export = charge, 0.0, -net - charge

        self.soc = energy / capacity
        return battery_flow, grid_import, grid_export

    def step(self, price):
        demand = self.demand_response(self.demand[self.t], price)
        pv = self.pv[self.t]
        battery_flow, grid_import, grid_export = self.dispatch_battery(demand - pv)

        info = {"t": self.t, "price": price, "base_demand": self.demand[self.t], "demand": demand,
                "pv": pv, "imbalance": abs(demand - pv), "battery_flow": battery_flow,
                "soc": self.soc, "grid_import": grid_import, "grid_export": grid_export}
        self.history.append(info)
        self.t += 1
        return info

    def run(self, policy):
        """Simule toute la période. policy(state) -> prix en €/kWh."""
        self.reset()
        while self.t < len(self.demand):
            self.step(policy(self.state()))
        return pd.DataFrame(self.history)


def fixed_price_policy(price=config.MICROGRID["reference_price"]):
    """Référence : toujours le même tarif."""
    return lambda state: price


def rule_based_policy(peak_hours=config.PRICING["peak_hours"], solar_hours=config.PRICING["solar_hours"],
                      tariffs=config.PRICING["tariffs"]):
    """Stratégie simple à règles (heures pleines / heures creuses), sans prévision :
    tarif haut à la pointe du soir, tarif bas en milieu de journée, tarif normal sinon.
    Avec 5 heures hautes et 5 heures basses, le prix moyen sur 24 h reste 0,20 €/kWh."""
    low, normal, high = min(tariffs), config.MICROGRID["reference_price"], max(tariffs)

    def policy(state):
        if state["hour"] in peak_hours:
            return high
        if state["hour"] in solar_hours:
            return low
        return normal
    return policy
