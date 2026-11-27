import pytest

from src import config

pytest.importorskip("fastapi")
pytest.importorskip("tensorflow")
if not (config.PROCESSED_DIR / "grid_all.csv").exists() or not (config.MODELS_DIR / "pricing_dense.keras").exists():
    pytest.skip("exécuter d'abord les notebooks 01 à 05 (données traitées et modèles)", allow_module_level=True)

from fastapi.testclient import TestClient  # noqa: E402

from app.backend.api import app  # noqa: E402

client = TestClient(app)


def test_summary():
    r = client.get("/api/summary")
    assert r.status_code == 200
    s = r.json()
    assert s["test_hours"] > 0 and 0 < s["forecast_metrics"]["demand"]["R2"] <= 1


def test_forecast_lengths():
    f = client.get("/api/forecast?days=2").json()
    assert len(f["time"]) == len(f["demand_true"]) == len(f["pv_pred"]) == 48


def test_simulate_default_matches_reference_prices():
    r = client.post("/api/simulate", json={"days_shown": 1})
    assert r.status_code == 200
    sim = r.json()
    assert set(sim["table"]) == {"Tarif fixe", "Heures pleines / creuses", "Réseau dense"}
    assert sim["table"]["Tarif fixe"]["Prix moyen (€/kWh)"] == pytest.approx(config.MICROGRID["reference_price"])
    assert len(sim["series"]["Réseau dense"]["price"]) == 24


def test_simulate_rejects_bad_input():
    assert client.post("/api/simulate", json={"elasticity": 0.5}).status_code == 422


def test_tariff_reacts_to_balance():
    deficit = client.post("/api/tariff", json={"demand_forecast": 70, "pv_forecast": 0, "hour": 19}).json()
    surplus = client.post("/api/tariff", json={"demand_forecast": 40, "pv_forecast": 100, "hour": 12}).json()
    assert deficit["tariff"] in config.PRICING["tariffs"]
    assert deficit["tariff"] > surplus["tariff"]
    assert sum(deficit["probabilities"].values()) == pytest.approx(1, abs=1e-3)


def test_frontend_is_served():
    r = client.get("/")
    assert r.status_code == 200 and "Micro-réseau" in r.text and "simForm" in r.text
    assert client.get("/static/app.js").status_code == 200
