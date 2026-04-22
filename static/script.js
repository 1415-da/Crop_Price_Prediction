function getModel() {
  return document.getElementById("modelSelect").value;
}

function renderPreviewTable(rows) {
  const container = document.getElementById("previewTable");
  if (!rows || !rows.length) {
    container.innerHTML = "";
    return;
  }
  const keys = Object.keys(rows[0]);
  let html = `<table class="table table-sm table-bordered"><thead><tr>`;
  keys.forEach(k => html += `<th>${k}</th>`);
  html += `</tr></thead><tbody>`;
  rows.forEach(r => {
    html += `<tr>`;
    keys.forEach(k => html += `<td>${r[k] ?? ""}</td>`);
    html += `</tr>`;
  });
  html += `</tbody></table>`;
  container.innerHTML = html;
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
  renderPreviewTable([]);
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

  renderPreviewTable(data.preview || []);
}

async function generateSample() {
  const n = document.getElementById("sampleN").value || 10;
  const model = getModel();
  const res = await fetch(`/generate_sample?n=${n}&model=${model}`);
  const data = await res.json();

  document.getElementById("predictionCard").innerText =
    `${data.message} | Model: ${data.model.toUpperCase()} | Rows: ${data.rows}`;

  document.getElementById("downloadArea").innerHTML =
    `<a class="btn btn-outline-primary btn-sm" href="${data.download_url}">Download Predictions CSV</a>`;

  renderPreviewTable(data.preview || []);
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
