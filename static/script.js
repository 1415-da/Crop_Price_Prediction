function getModel() {
  return document.getElementById("modelSelect").value;
}

let generatedSampleData = [];

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
    return;
  }

  if (card) card.classList.remove("is-hidden");
  if (stateBadge) {
    stateBadge.className = "badge text-bg-success";
    stateBadge.innerText = "Visible";
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
}

async function loadEDA() {
  const res = await fetch("/eda");
  const data = await res.json();

  const summary = {
    summary: data.summary,
    missing: data.missing
  };
  document.getElementById("edaSummary").innerText = JSON.stringify(summary, null, 2);

  if (data.correlation_plot) {
    const fig = JSON.parse(data.correlation_plot);
    Plotly.newPlot("corrPlot", fig.data, fig.layout);
  }
}

document.addEventListener("DOMContentLoaded", () => {
  setDataOverviewAvailability(false);
  syncOverviewToggleText();
});
