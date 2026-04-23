function getModel() {
  return document.getElementById("modelSelect").value;
}

let generatedSampleData = [];
let edaCache = null;

function activateRightTab(tabPanelId) {
  const tabBtn = document.querySelector(`#rightPanelTabs button[data-bs-target="#${tabPanelId}"]`);
  if (!tabBtn || typeof bootstrap === "undefined") return;
  const tab = new bootstrap.Tab(tabBtn);
  tab.show();
}

function setDataOverviewAvailability(hasData) {
  const card = document.getElementById("dataOverviewCard");
  const buttons = document.querySelectorAll(".overview-toggle-btn");
  const stateBadge = document.getElementById("overviewState");

  buttons.forEach((btn) => btn.classList.toggle("d-none", !hasData));

  if (!hasData) {
    if (card) card.classList.add("is-hidden");
    if (stateBadge) {
      stateBadge.className = "badge text-bg-secondary";
      stateBadge.innerText = "Hidden";
    }
    const overviewDownload = document.getElementById("overviewDownloadArea");
    if (overviewDownload) overviewDownload.innerHTML = "";
    return;
  }

  if (card) card.classList.remove("is-hidden");
  if (stateBadge) {
    stateBadge.className = "badge text-bg-success";
    stateBadge.innerText = "Visible";
  }
  const overviewDownload = document.getElementById("overviewDownloadArea");
  if (overviewDownload) {
    overviewDownload.innerHTML =
      `<a class="btn btn-outline-primary btn-sm" href="/download_overview">Download Data Overview CSV</a>`;
  }
}

function syncOverviewToggleText() {
  const card = document.getElementById("dataOverviewCard");
  const buttons = document.querySelectorAll(".overview-toggle-btn");
  if (!card || !buttons.length) return;
  const isHidden = card.classList.contains("is-hidden");
  buttons.forEach((btn) => {
    btn.innerText = isHidden ? "Show Data Overview" : "Hide Data Overview";
  });
}

function toggleDataOverview() {
  const card = document.getElementById("dataOverviewCard");
  const stateBadge = document.getElementById("overviewState");
  if (!card || !stateBadge) return;
  card.classList.toggle("is-hidden");
  const isHidden = card.classList.contains("is-hidden");
  stateBadge.className = isHidden ? "badge text-bg-secondary" : "badge text-bg-success";
  stateBadge.innerText = isHidden ? "Hidden" : "Visible";
  syncOverviewToggleText();
}

function renderPreviewTable(rows) {
  renderTable("previewTable", rows);
}

function renderPredictionTable(rows) {
  renderTable("predictionTable", rows);
}

function renderTable(containerId, rows) {
  const container = document.getElementById(containerId);
  if (!container) return;
  const normalizedRows = Array.isArray(rows) ? rows : Object.values(rows || {});
  if (!normalizedRows.length) {
    container.innerHTML = "";
    return;
  }
  const keys = Object.keys(normalizedRows[0]);
  let html = `<table class="table table-sm table-bordered"><thead><tr>`;
  keys.forEach(k => html += `<th>${k}</th>`);
  html += `</tr></thead><tbody>`;
  normalizedRows.forEach(r => {
    html += `<tr>`;
    keys.forEach(k => {
      const val = r[k];
      const displayVal = k === "predicted_price" && val !== undefined && val !== null
        ? `Rs ${Number(val).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`
        : (val ?? "");
      html += `<td>${displayVal}</td>`;
    });
    html += `</tr>`;
  });
  html += `</tbody></table>`;
  container.innerHTML = html;
}

function renderOverviewData(rows) {
  const normalizedRows = Array.isArray(rows) ? rows : Object.values(rows || {});
  if (!normalizedRows.length) {
    renderPreviewTable([]);
    setDataOverviewAvailability(false);
    syncOverviewToggleText();
    return;
  }
  renderPreviewTable(normalizedRows);
  setDataOverviewAvailability(true);
  syncOverviewToggleText();
}

async function predictManual() {
  const payload = {
    model: getModel(),
    crop: document.getElementById("crop").value || "rice",
    state: document.getElementById("state").value || "india",
    month_num: document.getElementById("month_num").value || 1,
    rainfall_mm: document.getElementById("rainfall_mm").value || "",
    temperature_c: document.getElementById("temperature_c").value || "",
    humidity_pct: document.getElementById("humidity_pct").value || "",
    yield_kg_per_ha: document.getElementById("yield_kg_per_ha").value || "",
    area_ha: document.getElementById("area_ha").value || "",
    lag_price_1: document.getElementById("lag_price_1").value || "",
    lag_price_3: document.getElementById("lag_price_3").value || "",
    rolling_price_3: document.getElementById("rolling_price_3").value || ""
  };

  const res = await fetch("/predict_manual", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload)
  });
  const data = await res.json();

  document.getElementById("predictionCard").innerText =
    `Predicted Price (${data.model.toUpperCase()}): Rs ${data.prediction}`;
  document.getElementById("downloadArea").innerHTML = "";
  renderPredictionTable([]);
  renderOverviewData([]);
  loadMetricsDiagnostics();
  activateRightTab("predictionPanel");
}

async function predictCSV() {
  const fileInput = document.getElementById("csvFile");
  if (!fileInput.files.length) {
    alert("Please choose a CSV file.");
    return;
  }

  const fd = new FormData();
  fd.append("file", fileInput.files[0]);
  fd.append("model", getModel());

  const res = await fetch("/predict_csv", { method: "POST", body: fd });
  const data = await res.json();

  document.getElementById("predictionCard").innerText =
    `${data.message} | Model: ${data.model.toUpperCase()} | Rows: ${data.rows}`;

  document.getElementById("downloadArea").innerHTML =
    `<a class="btn btn-outline-primary btn-sm" href="${data.download_url}">Download Predictions CSV</a>`;

  renderOverviewData(data.input_preview || []);
  renderPredictionTable(data.predictions || data.preview || []);
  loadMetricsDiagnostics();
  activateRightTab("predictionPanel");
}

async function generateSample() {
  const n = document.getElementById("sampleN").value || 10;
  const res = await fetch(`/generate_sample?n=${n}`);
  const data = await res.json();

  generatedSampleData = data.sample_data || [];
  document.getElementById("predictSampleBtn").disabled = generatedSampleData.length === 0;
  document.getElementById("predictionCard").innerText =
    `${data.message} | Rows: ${data.rows}. Click "Predict Generated Sample" to run prediction.`;
  document.getElementById("downloadArea").innerHTML = "";
  renderPredictionTable([]);
  renderOverviewData(data.preview || []);
  activateRightTab("predictionPanel");
}

async function predictSampleData() {
  if (!generatedSampleData.length) {
    alert("Please generate sample data first.");
    return;
  }

  const payload = {
    model: getModel(),
    sample_data: generatedSampleData
  };

  const res = await fetch("/predict_sample", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload)
  });
  const data = await res.json();

  if (!res.ok) {
    alert(data.error || "Sample prediction failed.");
    return;
  }

  document.getElementById("predictionCard").innerText =
    `Predicted ${data.rows} rows using ${data.model.toUpperCase()} model.`;
  document.getElementById("downloadArea").innerHTML =
    `<a class="btn btn-outline-primary btn-sm" href="${data.download_url}">Download Predictions CSV</a>`;
  renderPredictionTable(data.predictions || data.preview || []);
  loadMetricsDiagnostics();
  activateRightTab("predictionPanel");
}

async function loadEDA() {
  try {
    activateRightTab("edaPanel");
    const priceFile = document.getElementById("edaPriceFile")?.files?.[0];
    const yieldFile = document.getElementById("edaYieldFile")?.files?.[0];
    const formData = new FormData();
    if (priceFile) formData.append("price_file", priceFile);
    if (yieldFile) formData.append("yield_file", yieldFile);

    const res = await fetch("/eda_analysis", {
      method: "POST",
      body: formData
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.error || "Failed to load EDA");
    }
    edaCache = data;
    renderEdaDashboard(data);
  } catch (err) {
    const el = document.getElementById("edaOverviewText");
    if (el) el.innerText = `EDA load error: ${err.message}`;
  }
}

function plotFromJson(targetId, jsonStr) {
  const el = document.getElementById(targetId);
  if (!el) return;
  if (!jsonStr) {
    if (typeof Plotly !== "undefined") Plotly.purge(el);
    el.innerHTML = "";
    return;
  }
  const fig = JSON.parse(jsonStr);
  
  fig.layout = fig.layout || {};
  fig.layout.paper_bgcolor = 'rgba(0,0,0,0)';
  fig.layout.plot_bgcolor = 'rgba(0,0,0,0)';
  fig.layout.font = { family: 'Inter, sans-serif' };

  if (fig.data) {
    fig.data.forEach(trace => {
        if (!trace.marker) trace.marker = {};
        if (!trace.line) trace.line = {};
        // Make the main color match the green theme if no color is set
        if (trace.type === 'bar' || trace.type === 'histogram') {
            if (!trace.marker.color) trace.marker.color = '#177B43';
        } else if (trace.type === 'scatter') {
            if (!trace.line.color) trace.line.color = '#177B43';
            if (!trace.marker.color) trace.marker.color = '#177B43';
        }
    });
  }

  if (typeof Plotly !== "undefined") {
    Plotly.react(el, fig.data || [], fig.layout, { responsive: true });
  }
}

function renderEdaDashboard(data) {
  const o = data.overview || {};
  const overviewText = {
    mode: data.mode || "combined",
    rows_cols: `Price: ${o.price_rows || 0}x${o.price_cols || 0}, Yield/Weather: ${o.yield_rows || 0}x${o.yield_cols || 0}`,
    missing_values_count: o.missing_values_count,
    duplicate_rows_count: o.duplicate_rows_count,
    date_range: o.date_range,
    unique_crops: o.unique_crops,
    unique_states_markets: o.unique_states_markets,
    analyst_note: o.analyst_note
  };
  const overviewEl = document.getElementById("edaOverviewText");
  if (overviewEl) overviewEl.innerText = JSON.stringify(overviewText, null, 2);

  const c = data.charts || {};
  plotFromJson("edaMissingPlot", c.missing_values);
  plotFromJson("edaPriceHist", c.price_histogram);
  plotFromJson("edaPriceBox", c.price_boxplot);
  plotFromJson("edaSeasonalTrend", c.seasonal_trend);
  plotFromJson("edaCropTop", c.crop_price_top);
  plotFromJson("edaCropLow", c.crop_price_low);
  plotFromJson("edaStatePrice", c.state_price);
  plotFromJson("edaYieldPrice", c.yield_vs_price);
  plotFromJson("edaRainPrice", c.rainfall_vs_price);
  plotFromJson("edaTempPrice", c.temperature_vs_price);
  plotFromJson("edaHumPrice", c.humidity_vs_price);
  plotFromJson("edaCorr", c.correlation);
  plotFromJson("edaOutliers", c.outliers);

  const insightsEl = document.getElementById("edaInsights");
  if (insightsEl) {
    insightsEl.innerHTML = "";
    (data.insights || []).forEach((x) => {
      const li = document.createElement("li");
      li.innerText = x;
      insightsEl.appendChild(li);
    });
  }

  const modelSelect = document.getElementById("edaFeatureModel");
  if (modelSelect) {
    modelSelect.innerHTML = "";
    Object.keys(data.feature_importance || {}).forEach((m) => {
      const opt = document.createElement("option");
      opt.value = m;
      opt.innerText = m.toUpperCase();
      modelSelect.appendChild(opt);
    });
  }
  updateFeatureImportance();
}

function updateFeatureImportance() {
  if (!edaCache) return;
  const modelName = document.getElementById("edaFeatureModel")?.value;
  const figJson = edaCache?.feature_importance?.[modelName];
  if (figJson) plotFromJson("edaFeatureImportance", figJson);

  const residual = edaCache?.residual_analysis?.[modelName];
  if (residual) {
    plotFromJson("edaResidualScatter", residual.actual_vs_pred);
    plotFromJson("edaResidualHist", residual.residual_hist);
  }
}

function renderMetricsDiagnostics(diag) {
  if (!diag || !diag.models || !Object.keys(diag.models).length) {
    const cmEl = document.getElementById("confusionMatrixPlot");
    const rocEl = document.getElementById("rocAucPlot");
    if (cmEl) cmEl.innerHTML = `<div class="text-muted small">Confusion matrix is available when the current input has actual target values.</div>`;
    if (rocEl) rocEl.innerHTML = `<div class="text-muted small">ROC/AUC is available when the current input has actual target values.</div>`;
    return;
  }

  const modelKeys = Object.keys(diag.models || {});
  const bestModel = diag.best_model || (modelKeys.length ? modelKeys[0] : null);
  const best = bestModel ? diag.models[bestModel] : null;
  if (best && best.confusion_matrix) {
    const cm = best.confusion_matrix;
    const cmFig = {
      data: [
        {
          z: cm,
          x: ["Pred Down", "Pred Up"],
          y: ["Actual Down", "Actual Up"],
          type: "heatmap",
          colorscale: "Blues",
          text: cm,
          texttemplate: "%{text}",
          showscale: true
        }
      ],
      layout: {
        title: `Confusion Matrix - ${bestModel.toUpperCase()}`,
        margin: { t: 50, l: 70, r: 20, b: 50 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { family: 'Inter, sans-serif' }
      }
    };
    Plotly.newPlot("confusionMatrixPlot", cmFig.data, cmFig.layout);
  }

  const rocTraces = [];
  Object.entries(diag.models).forEach(([modelName, modelData]) => {
    const roc = modelData.roc_curve;
    if (!roc || !roc.fpr || !roc.tpr) return;
    rocTraces.push({
      x: roc.fpr,
      y: roc.tpr,
      mode: "lines",
      type: "scatter",
      name: `${modelName.toUpperCase()} (AUC=${Number(roc.auc).toFixed(3)})`
    });
  });
  rocTraces.push({
    x: [0, 1],
    y: [0, 1],
    mode: "lines",
    type: "scatter",
    name: "Random",
    line: { dash: "dash", color: "#6b7280" }
  });
  if (rocTraces.length > 1) {
    Plotly.newPlot(
      "rocAucPlot",
      rocTraces,
      {
        title: "ROC Curves",
        xaxis: { title: "False Positive Rate" },
        yaxis: { title: "True Positive Rate" },
        margin: { t: 50, l: 60, r: 20, b: 60 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { family: 'Inter, sans-serif' }
      }
    );
  }
}

function renderMetricsTable(rows, source) {
  const tbody = document.getElementById("metricsBody");
  if (!tbody) return;
  const sourceEl = document.getElementById("metricsSourceText");
  const noteEl = document.getElementById("metricsNoteText");
  if (sourceEl) sourceEl.innerText = `Source: ${source || "N/A"}`;

  const list = Array.isArray(rows) ? rows : [];
  if (!list.length) {
    tbody.innerHTML = `<tr><td colspan="9" class="text-muted">Run a prediction to view metrics for the current website input.</td></tr>`;
    if (noteEl) noteEl.innerText = "";
    return;
  }

  const hasActual = list.some((r) => r.MAE !== null && r.MAE !== undefined);
  if (noteEl) {
    noteEl.innerText = hasActual
      ? "Metrics are computed from the current input data."
      : "No actual target price found in current input; showing prediction-only summary.";
  }

  const fmt = (v) => (v === null || v === undefined || Number.isNaN(v) ? "-" : v);
  tbody.innerHTML = list
    .map(
      (r) => `<tr>
      <td>${r.model ?? ""}</td>
      <td>${fmt(r.MAE)}</td>
      <td>${fmt(r.RMSE)}</td>
      <td>${fmt(r.R2)}</td>
      <td>${fmt(r.Accuracy)}</td>
      <td>${fmt(r.Precision)}</td>
      <td>${fmt(r.Recall)}</td>
      <td>${fmt(r.F1)}</td>
      <td>${fmt(r.AvgPredPrice)}</td>
    </tr>`
    )
    .join("");
}

async function loadMetricsDiagnostics() {
  try {
    const res = await fetch("/metrics");
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.error || "Failed to load metrics diagnostics");
    }
    renderMetricsTable(data.metrics || [], data.source);
    renderMetricsDiagnostics(data.diagnostics || {});
  } catch (err) {
    const cm = document.getElementById("confusionMatrixPlot");
    const roc = document.getElementById("rocAucPlot");
    if (cm) cm.innerHTML = `<div class="text-danger small">Confusion matrix load error: ${err.message}</div>`;
    if (roc) roc.innerHTML = `<div class="text-danger small">ROC/AUC load error: ${err.message}</div>`;
    renderMetricsTable([], "Error loading metrics");
  }
}

function initEdaPlaceholders() {
  const overviewEl = document.getElementById("edaOverviewText");
  if (overviewEl) {
    overviewEl.innerText = "Click 'Run EDA' to generate charts for selected datasets.";
  }
  [
    "edaMissingPlot", "edaPriceHist", "edaPriceBox", "edaSeasonalTrend", "edaCropTop", "edaCropLow",
    "edaStatePrice", "edaYieldPrice", "edaRainPrice", "edaTempPrice", "edaHumPrice",
    "edaCorr", "edaOutliers", "edaFeatureImportance", "edaResidualScatter", "edaResidualHist"
  ].forEach((id) => {
    const el = document.getElementById(id);
    if (el) {
      if (typeof Plotly !== "undefined") Plotly.purge(el);
      el.innerHTML = "";
    }
  });
  const insightsEl = document.getElementById("edaInsights");
  if (insightsEl) insightsEl.innerHTML = "";
}

document.addEventListener("DOMContentLoaded", () => {
  setDataOverviewAvailability(false);
  syncOverviewToggleText();
  loadMetricsDiagnostics();
  initEdaPlaceholders();

  // Fix plot resizing issues when switching tabs
  const tabEls = document.querySelectorAll('button[data-bs-toggle="tab"]');
  tabEls.forEach(el => {
    el.addEventListener('shown.bs.tab', () => {
      window.dispatchEvent(new Event('resize'));
    });
  });
});
