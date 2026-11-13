"use strict";

const datasets = {
  "bike": {
    "name": "Bike NYC",
    "info": "16 \u00d7 8 grid \u00b7 hourly counts",
    "rmse": [
      19.87,
      12.65,
      12.5,
      10.24,
      10.29,
      10.15,
      9.38,
      9.16,
      8.88,
      8.53,
      8.15,
      8.02,
      6.52,
      6.32,
      6.18,
      6.09,
      6.28,
      6.11,
      6.01,
      5.87
    ],
    "cpc": [
      0.37,
      0.42,
      0.42,
      0.58,
      0.59,
      0.59,
      0.61,
      0.64,
      0.65,
      0.67,
      0.67,
      0.68,
      0.69,
      0.7,
      0.7,
      0.72,
      0.72,
      0.71,
      0.74,
      0.75
    ],
    "mae": [
      5.28,
      4.51,
      4.48,
      4.15,
      4.16,
      4.1,
      4.22,
      4.17,
      4.13,
      4.07,
      3.68,
      3.59,
      3.16,
      3.08,
      3.04,
      3.04,
      3.0,
      2.98,
      2.99,
      2.86
    ]
  },
  "taxi": {
    "name": "Taxi Beijing",
    "info": "32 \u00d7 32 grid \u00b7 30-minute counts",
    "rmse": [
      46.12,
      25.98,
      25.61,
      20.04,
      20.0,
      19.99,
      19.32,
      18.89,
      18.28,
      14.91,
      14.35,
      14.12,
      9.82,
      9.58,
      9.46,
      9.32,
      9.46,
      9.42,
      9.11,
      8.95
    ],
    "cpc": [
      0.39,
      0.4,
      0.4,
      0.45,
      0.47,
      0.44,
      0.48,
      0.49,
      0.49,
      0.54,
      0.56,
      0.58,
      0.61,
      0.61,
      0.62,
      0.62,
      0.63,
      0.64,
      0.64,
      0.67
    ],
    "mae": [
      33.26,
      19.87,
      19.33,
      16.31,
      16.04,
      16.15,
      15.26,
      15.2,
      15.08,
      12.08,
      12.0,
      11.95,
      8.22,
      7.98,
      7.88,
      7.34,
      7.75,
      7.71,
      7.26,
      7.08
    ]
  },
  "spain": {
    "name": "Spain OD",
    "info": "2,735 irregular regions \u00b7 hourly counts",
    "rmse": [
      52.39,
      45.66,
      43.84,
      30.06,
      29.78,
      29.69,
      null,
      null,
      null,
      28.22,
      28.1,
      28.05,
      24.18,
      23.68,
      23.29,
      21.74,
      22.08,
      21.72,
      21.55,
      20.81
    ],
    "cpc": [
      0.29,
      0.34,
      0.36,
      0.43,
      0.42,
      0.43,
      null,
      null,
      null,
      0.45,
      0.45,
      0.45,
      0.59,
      0.62,
      0.61,
      0.67,
      0.64,
      0.64,
      0.67,
      0.69
    ],
    "mae": [
      29.16,
      28.97,
      28.83,
      18.2,
      18.98,
      17.65,
      null,
      null,
      null,
      14.26,
      13.58,
      13.26,
      10.42,
      10.12,
      9.97,
      9.94,
      10.01,
      9.86,
      9.85,
      9.5
    ]
  }
};
// Values transcribed from the ablation and multi-horizon sections of the poster.
const ablationResults = {
  bike: [[6.11, 0.71], [6.05, 0.74], [5.91, 0.75], [5.87, 0.75]],
  taxi: [[9.42, 0.64], [9.28, 0.65], [9.02, 0.66], [8.95, 0.67]],
  spain: [[21.72, 0.64], [21.33, 0.67], [20.99, 0.69], [20.81, 0.69]],
};
const horizonResults = {
  bike: [[5.87, 0.75, 2.86], [5.94, 0.75, 3.01], [6.04, 0.74, 3.08], [6.11, 0.72, 3.11]],
  taxi: [[8.95, 0.67, 7.08], [9.18, 0.62, 7.18], [9.38, 0.61, 7.28], [9.47, 0.59, 7.33]],
  spain: [[20.81, 0.69, 9.50], [21.55, 0.66, 9.92], [22.30, 0.64, 10.16], [22.99, 0.60, 10.23]],
};
const ablationLabels = ["Mobility only", "Mobility + weather", "Mobility + attractiveness", "Full TS-Mob"];
const horizonLabels = ["1 step", "3 steps", "5 steps", "10 steps"];

function renderStudyRows(id, labels, values, highlightLast = false) {
  const rows = values.map((metrics, index) => {
    const row = document.createElement("tr");
    if (highlightLast && index === values.length - 1) row.className = "our-model";
    const label = document.createElement("th");
    label.scope = "row";
    label.textContent = labels[index];
    row.append(label);
    for (const value of metrics) {
      const cell = document.createElement("td");
      cell.textContent = value.toFixed(2);
      row.append(cell);
    }
    return row;
  });
  document.getElementById(id).replaceChildren(...rows);
}

const models = ["MA", "ARIMA", "VAR", "RNN", "LSTM", "GRU", "ST-ResNet", "DMVSTNet", "ACMF", "CrowdNet", "DSTAGNN", "MSAGGN", "TimeLLM (zero-shot)", "Moirai-S (zero-shot)", "TimesFM (zero-shot)", "Moirai-L (zero-shot)", "Moirai-S (fine-tuned)", "TimesFM (fine-tuned)", "Moirai-L (fine-tuned)", "TS-Mob"];
const modelGroups = ["Traditional time series", "Traditional time series", "Traditional time series", "Deep learning", "Deep learning", "Deep learning", "Deep learning", "Deep learning", "Deep learning", "Deep learning", "Deep learning", "Deep learning", "Foundation models \u00b7 zero-shot", "Foundation models \u00b7 zero-shot", "Foundation models \u00b7 zero-shot", "Foundation models \u00b7 zero-shot", "Foundation models \u00b7 fine-tuned", "Foundation models \u00b7 fine-tuned", "Foundation models \u00b7 fine-tuned", "TS-Mob"];
const datasetButtons = [...document.querySelectorAll("[data-dataset]")];
let activeDataset = "bike";

function renderResults() {
  const dataset = datasets[activeDataset];
  document.getElementById("dataset-title").textContent = dataset.name + " model comparison";
  document.getElementById("dataset-info").textContent = dataset.info;
  for (const button of datasetButtons) button.setAttribute("aria-pressed", String(button.dataset.dataset === activeDataset));

  const rows = [];
  const order = [models.length - 1, ...models.slice(0, -1).map((_, index) => index)];
  let previousGroup;
  for (const index of order) {
    const model = models[index];
    if (model !== "TS-Mob" && modelGroups[index] !== previousGroup) {
      const groupRow = document.createElement("tr");
      groupRow.className = "model-group";
      const label = document.createElement("th");
      label.colSpan = 4;
      label.textContent = modelGroups[index];
      groupRow.append(label);
      rows.push(groupRow);
    }
    previousGroup = modelGroups[index];
    const row = document.createElement("tr");
    if (model === "TS-Mob") row.className = "our-model";
    const heading = document.createElement("th");
    heading.scope = "row";
    heading.textContent = model.replace(" (zero-shot)", "").replace(" (fine-tuned)", "");
    row.append(heading);
    for (const metric of ["rmse", "mae", "cpc"]) {
      const cell = document.createElement("td");
      const value = dataset[metric][index];
      cell.textContent = value === null ? "—" : value.toFixed(2);
      if (value === null) cell.setAttribute("aria-label", "Not applicable");
      row.append(cell);
    }
    rows.push(row);
  }
  document.getElementById("result-rows").replaceChildren(...rows);
  document.getElementById("availability-note").textContent = activeDataset === "spain"
    ? "All 19 baselines. Dashes mark grid-convolutional models that do not apply to Spain’s irregular regions."
    : "All 19 baselines from the extended paper. Best results are highlighted.";
  document.getElementById("result-status").textContent = dataset.name + ". RMSE, MAE and CPC for all 20 models.";
}

for (const button of datasetButtons) button.addEventListener("click", () => {
  activeDataset = button.dataset.dataset;
  renderResults();
});
document.getElementById("ablation-dataset").addEventListener("change", event => {
  renderStudyRows("ablation-rows", ablationLabels, ablationResults[event.target.value], true);
});
document.getElementById("horizon-dataset").addEventListener("change", event => {
  renderStudyRows("horizon-rows", horizonLabels, horizonResults[event.target.value]);
});
renderResults();
renderStudyRows("ablation-rows", ablationLabels, ablationResults.bike, true);
renderStudyRows("horizon-rows", horizonLabels, horizonResults.bike);

for (const button of document.querySelectorAll("[data-copy]")) {
  button.addEventListener("click", async () => {
    const source = document.getElementById(button.dataset.copy);
    const status = document.getElementById(button.dataset.status);
    try {
      await navigator.clipboard.writeText(source.textContent);
      status.textContent = "Copied to clipboard.";
    } catch {
      const selection = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(source);
      selection.removeAllRanges();
      selection.addRange(range);
      status.textContent = "Text selected. Press Ctrl+C or Command+C to copy.";
    }
  });
}
