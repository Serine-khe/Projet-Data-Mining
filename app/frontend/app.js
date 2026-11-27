// Frontend du micro-réseau : appelle l'API FastAPI (app/backend/api.py) et affiche les résultats.

const COLORS = { demand: "#1f4e79", pv: "#c08a00", import: "#b03a2e", battery: "#2e7d4f", grey: "#8a94a3" };
const STRATEGY_COLORS = { "Tarif fixe": "#8a94a3", "Heures pleines / creuses": "#c08a00", "Réseau dense": "#1f4e79" };
const fr = (x, d = 1) => Number(x).toLocaleString("fr-FR", { minimumFractionDigits: d, maximumFractionDigits: d });
const signed = (x, d = 1) => `${x > 0 ? "+" : ""}${fr(x, d)}`.replace("-", "−");

Chart.defaults.color = "#5f6b7a";
Chart.defaults.borderColor = "#eceef1";
Chart.defaults.font.family = "'IBM Plex Sans', system-ui, sans-serif";
Chart.defaults.font.size = 12;
Chart.defaults.plugins.legend.align = "start";
Chart.defaults.plugins.legend.labels.boxWidth = 18;
Chart.defaults.plugins.legend.labels.boxHeight = 2;

const charts = {};
let lastSimulation = null;
let selectedStrategy = "Réseau dense";

async function api(path, body) {
  const options = body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {};
  const response = await fetch(path, options);
  if (!response.ok) throw new Error((await response.json()).detail || response.statusText);
  return response.json();
}

function drawChart(id, config) {
  if (charts[id]) charts[id].destroy();
  charts[id] = new Chart(document.getElementById(id), config);
}

function line(label, data, color, extra = {}) {
  return { label, data, borderColor: color, backgroundColor: color, borderWidth: 1.6, pointRadius: 0, tension: 0, ...extra };
}
const dashed = { borderDash: [4, 3], borderWidth: 1.2 };

function timeAxis(labels) {
  return {
    ticks: { maxTicksLimit: window.innerWidth < 600 ? 4 : 8, maxRotation: 0, callback: (v) => labels[v].slice(5, 13).replace("-", "/") },
    grid: { display: false },
  };
}

function baseOptions(scales, extra = {}) {
  return {
    maintainAspectRatio: false, animation: false,
    interaction: { mode: "index", intersect: false },
    scales, ...extra,
  };
}

// ---------------------------------------------------------------------------
// Chiffres clés
// ---------------------------------------------------------------------------
async function loadSummary() {
  const s = await api("/api/summary");
  const kpis = [
    ["R² demande (régression)", fr(s.forecast_metrics.demand.R2, 3), ""],
    ["RMSE demande", fr(s.forecast_metrics.demand.RMSE, 2), "kWh"],
    ["R² production PV (LSTM)", fr(s.forecast_metrics.pv.R2, 3), ""],
    ["RMSE production PV", fr(s.forecast_metrics.pv.RMSE, 1), "kWh"],
    ["Heures de test", s.test_hours, "14–31 déc. 2013"],
    ["Heures en déficit", `${Math.round(100 * s.deficit_share_test)} %`, "demande > PV"],
  ];
  document.getElementById("kpis").innerHTML = kpis.map(([label, value, unit]) =>
    `<div><dt>${label}</dt><dd>${value}${unit ? `<small>${unit}</small>` : ""}</dd></div>`).join("");
}

// ---------------------------------------------------------------------------
// Prévisions
// ---------------------------------------------------------------------------
async function loadForecast() {
  const days = document.getElementById("forecastDays").value;
  const f = await api(`/api/forecast?days=${days}`);
  drawChart("forecastChart", {
    type: "line",
    data: {
      labels: f.time,
      datasets: [
        line("Demande mesurée", f.demand_true, COLORS.demand),
        line("Demande prévue", f.demand_pred, COLORS.demand, dashed),
        line("PV estimée (irradiance mesurée)", f.pv_true, COLORS.pv),
        line("PV prévue (LSTM)", f.pv_pred, COLORS.pv, dashed),
      ],
    },
    options: baseOptions({ x: timeAxis(f.time), y: { title: { display: true, text: "kWh / h" }, beginAtZero: true } }),
  });
}

// ---------------------------------------------------------------------------
// Simulation
// ---------------------------------------------------------------------------
function readControls() {
  return {
    elasticity: Number(document.getElementById("elasticity").value),
    battery_kwh: Math.max(1, Number(document.getElementById("battery").value)),
    pv_peak_kw: Number(document.getElementById("pvPeak").value),
    days_shown: Number(document.getElementById("days").value),
  };
}

function updateOutputs() {
  const c = readControls();
  document.getElementById("elasticityOut").textContent = fr(c.elasticity, 2).replace("-", "−");
  document.getElementById("batteryOut").textContent = `${document.getElementById("battery").value} kWh`;
  document.getElementById("pvOut").textContent = `${c.pv_peak_kw} kWc`;
  document.getElementById("daysOut").textContent = c.days_shown;
}

async function runSimulation(event) {
  if (event) event.preventDefault();
  const button = document.getElementById("simButton");
  button.disabled = true;
  button.textContent = "Calcul…";
  try {
    lastSimulation = await api("/api/simulate", readControls());
    renderSimulation();
  } catch (err) {
    alert("Erreur de simulation : " + err.message);
  } finally {
    button.disabled = false;
    button.textContent = "Simuler";
  }
}

function renderSimulation() {
  const sim = lastSimulation;
  const dense = sim.change_vs_fixed["Réseau dense"];
  const deltas = [
    ["Déséquilibre |D − PV|", dense["Déséquilibre |D - PV| (kWh)"], true],
    ["Énergie importée", dense["Import réseau (kWh)"], true],
    ["Pic d'import", dense["Pic d'import réseau (kW)"], true],
    ["Prix moyen", dense["Prix moyen (€/kWh)"], false],
  ];
  document.getElementById("deltas").innerHTML = deltas.map(([label, v, lowerIsBetter]) => {
    const good = lowerIsBetter ? v <= 0 : v <= 0.5;
    return `<div><dt>${label} — réseau dense</dt><dd class="${good ? "good" : "bad"}">${signed(v)} %</dd></div>`;
  }).join("");

  // Onglets : stratégie affichée sur le graphique d'énergie
  const tabs = document.getElementById("strategyTabs");
  tabs.innerHTML = Object.keys(sim.series).map((name) =>
    `<button type="button" class="${name === selectedStrategy ? "active" : ""}" data-name="${name}">${name}</button>`).join("");
  tabs.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => { selectedStrategy = b.dataset.name; renderSimulation(); }));

  const s = sim.series[selectedStrategy];
  drawChart("energyChart", {
    type: "line",
    data: {
      labels: sim.time,
      datasets: [
        line("Demande", s.demand, COLORS.demand),
        line("Production PV", s.pv, COLORS.pv),
        line("Import réseau", s.grid_import, COLORS.import),
        line("Charge batterie (%)", s.soc.map((x) => 100 * x), COLORS.battery, { yAxisID: "soc", ...dashed }),
      ],
    },
    options: baseOptions({
      x: timeAxis(sim.time),
      y: { title: { display: true, text: "kWh / h" }, beginAtZero: true },
      soc: { position: "right", min: 0, max: 100, grid: { display: false }, title: { display: true, text: "batterie (%)" } },
    }),
  });

  drawChart("priceChart", {
    type: "line",
    data: {
      labels: sim.time,
      datasets: Object.entries(sim.series).map(([name, x]) =>
        line(name, x.price, STRATEGY_COLORS[name], { stepped: true, borderWidth: name === "Réseau dense" ? 1.8 : 1.2 })),
    },
    options: baseOptions({ x: timeAxis(sim.time), y: { min: 0.08, max: 0.32, title: { display: true, text: "€/kWh" } } }),
  });

  // Tableau des indicateurs
  const names = Object.keys(sim.table);
  const digits = (k) => (k.includes("€/kWh") ? 3 : 1);
  document.getElementById("simTable").innerHTML =
    `<thead><tr><th>Indicateur</th>${names.map((n) => `<th>${n}</th>`).join("")}</tr></thead><tbody>` +
    sim.indicators.map((k) => `<tr><td>${k}</td>${names.map((n) => {
      const change = sim.change_vs_fixed[n][k];
      const note = n === "Tarif fixe" || !isFinite(change) ? "" : `<small>${signed(change)} %</small>`;
      return `<td>${fr(sim.table[n][k], digits(k))}${note}</td>`;
    }).join("")}</tr>`).join("") + "</tbody>";
}

// ---------------------------------------------------------------------------
// Agent
// ---------------------------------------------------------------------------
async function askAgent(event) {
  if (event) event.preventDefault();
  const body = {
    demand_forecast: Number(document.getElementById("aDemand").value),
    pv_forecast: Number(document.getElementById("aPv").value),
    hour: Number(document.getElementById("aHour").value),
  };
  const r = await api("/api/tariff", body);
  document.getElementById("tariffValue").textContent = `${fr(r.tariff, 2)} €/kWh`;
  document.getElementById("tariffBalance").textContent =
    r.balance_kwh >= 0 ? `déficit de ${fr(r.balance_kwh)} kWh` : `surplus de ${fr(-r.balance_kwh)} kWh`;
  const labels = Object.keys(r.probabilities);
  drawChart("probaChart", {
    type: "bar",
    data: {
      labels: labels.map((t) => `${t.replace(".", ",")} €`),
      datasets: [{ label: "Probabilité (%)", data: labels.map((t) => 100 * r.probabilities[t]),
                   backgroundColor: labels.map((t) => (Number(t) === r.tariff ? COLORS.demand : "#c9d0d9")),
                   barPercentage: 0.6 }],
    },
    options: baseOptions({ y: { min: 0, max: 100, title: { display: true, text: "probabilité (%)" } }, x: { grid: { display: false } } },
                         { plugins: { legend: { display: false } } }),
  });
}

// ---------------------------------------------------------------------------
async function init() {
  const status = document.getElementById("status");
  try {
    await loadSummary();
    status.textContent = "API : connectée";
    status.className = "ok";
  } catch (err) {
    status.textContent = "API : indisponible";
    status.className = "err";
    document.getElementById("kpis").innerHTML = `<div><dt>Erreur</dt><dd style="font-size:14px">${err.message}</dd></div>`;
    return;
  }
  document.getElementById("forecastDays").addEventListener("change", loadForecast);
  ["elasticity", "battery", "pvPeak", "days"].forEach((id) => document.getElementById(id).addEventListener("input", updateOutputs));
  document.getElementById("simForm").addEventListener("submit", runSimulation);
  document.getElementById("agentForm").addEventListener("submit", askAgent);
  updateOutputs();
  await Promise.all([loadForecast(), runSimulation(), askAgent()]);
}

init();
