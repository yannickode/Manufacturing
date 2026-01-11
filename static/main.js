// PART 1:

// PART 2: UPLOAD AND MODELS 
// ---------------------------
document.addEventListener("DOMContentLoaded", () => {
  const uploadForm = document.getElementById("upload-form");
  const exampleBtn = document.getElementById("example-btn");
  const previewDiv = document.getElementById("preview");

  const featuresInput = document.getElementById("features-input");
  const targetInput = document.getElementById("target-input");
  const clustersInput = document.getElementById("clusters-input");
  const knnInput = document.getElementById("knn-input");

  const metricsBody = document.getElementById("metrics-body");

  const btnLR = document.getElementById("btn-lr");
  const btnSVM = document.getElementById("btn-svm");
  const btnCluster = document.getElementById("btn-cluster");
  const btnNN = document.getElementById("btn-nn");
  const btnKNN = document.getElementById("btn-knn");
  const btnReset = document.getElementById("btn-reset");

  const plot2dDiv = document.getElementById("plot-2d");
  const plot3dDiv = document.getElementById("plot-3d");

  let loaded = false;
  let activeModels = []; // ["lr","svm",...]


  function getSelectedFeatures() {
    const select = document.getElementById("features-select");
    return Array.from(select.selectedOptions).map(o => o.value);
  }

  // function parseFeatures() {
  //   const raw = (featuresInput.value || "").trim();
  //   if (!raw) return [];
  //   return raw.split(",").map(s => s.trim()).filter(Boolean);
  // }

  function clearMetricsTable() {
    metricsBody.innerHTML = `<tr><td colspan="5">No model trained yet.</td></tr>`;
  }

  function addMetricRow(row) {
    // remove "No model trained yet."
    if (metricsBody.querySelector("td[colspan='5']")) metricsBody.innerHTML = "";

    const rmse = row.rmse === null || row.rmse === undefined ? "-" : row.rmse.toFixed(4);
    const mae = row.mae === null || row.mae === undefined ? "-" : row.mae.toFixed(4);
    const r2 = row.r2 === null || row.r2 === undefined ? "-" : row.r2.toFixed(4);

    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><strong>${row.model_type}</strong></td>
      <td>${rmse}</td>
      <td>${mae}</td>
      <td>${r2}</td>
      <td>${row.notes || ""}</td>
    `;
    metricsBody.appendChild(tr);
  }

  async function postJSON(url, payload) {
    const resp = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || "Request failed");
    return data;
  }

  async function uploadFilePreview(file) {
    const fd = new FormData();
    fd.append("datafile", file);

    const resp = await fetch("/upload_data", { method: "POST", body: fd });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || "Upload failed");
    return data;
  }


  function populateColumnDropdowns(columns) {
    const featureSelect = document.getElementById("features-select");
    const targetSelect = document.getElementById("target-select");

    // clear old options
    featureSelect.innerHTML = "";
    targetSelect.innerHTML = `<option value="">-- None (for clustering) --</option>`;

    columns.forEach(col => {
      const opt1 = document.createElement("option");
      opt1.value = col;
      opt1.textContent = col;
      featureSelect.appendChild(opt1);

      const opt2 = document.createElement("option");
      opt2.value = col;
      opt2.textContent = col;
      targetSelect.appendChild(opt2);
    });
  }

  async function refresh2DPlot() {
    const features = getSelectedFeatures();
    const target = document.getElementById("target-select").value;
    const n_clusters = parseInt(clustersInput.value || "3", 10);
    const n_neighbors = parseInt(knnInput.value || "5", 10);

    const data = await postJSON("/model_fit_2d", {
      models: activeModels,
      features,
      target,
      n_clusters,
      n_neighbors
    });

    Plotly.newPlot(plot2dDiv, data.figure.data, data.figure.layout, {
      responsive: true
      // Plotly modebar already includes "Download plot as png"
    });
  }

  async function refresh3DPlotIfNeeded() {
    const features = getSelectedFeatures();
    const target = document.getElementById("target-select").value;

    if (features.length !== 3) {
      plot3dDiv.innerHTML = `<div style="padding:14px;color:#56607a;">Enter exactly 3 features to enable 3D plot.</div>`;
      return;
    }

    const data = await postJSON("/plot3d", { features, target });
    Plotly.newPlot(plot3dDiv, data.figure.data, data.figure.layout, { responsive: true });
  }

  async function runModel(modelKey) {
    if (!loaded) {
      alert("Load a dataset first (Upload or Example).");
      return;
    }

    const features = getSelectedFeatures();
    const target = document.getElementById("target-select").value;
    const n_clusters = parseInt(clustersInput.value || "3", 10);
    const n_neighbors = parseInt(knnInput.value || "5", 10);

    if (!features.length) {
      alert("Enter feature columns (comma separated).");
      return;
    }

    // 1) Metrics (table row)
    const metrics = await postJSON("/model_metrics", {
      model_type: modelKey,
      features,
      target,
      n_clusters,
      n_neighbors
    });
    addMetricRow(metrics);

    // 2) Overlay plot: add model and redraw
    if (!activeModels.includes(modelKey)) activeModels.push(modelKey);
    await refresh2DPlot();

    // 3) 3D plot only if 3 features
    await refresh3DPlotIfNeeded();
  }

  // ---------- Upload form ----------
  uploadForm.addEventListener("submit", async (e) => {
    e.preventDefault();

    const input = document.getElementById("datafile");
    if (!input.files.length) {
      alert("Choose a CSV or Excel file first.");
      return;
    }

    try {
      const data = await uploadFilePreview(input.files[0]);
      loaded = true;
      populateColumnDropdowns(data.columns);

      previewDiv.innerHTML = `
        <p><strong>Loaded.</strong></p>
        ${data.preview_html}
      `;

      activeModels = [];
      clearMetricsTable();
      plot2dDiv.innerHTML = "";
      plot3dDiv.innerHTML = `<div style="padding:14px;color:#56607a;">Enter exactly 3 features to enable 3D plot.</div>`;
    } catch (err) {
      alert(err.message);
    }
  });

  // ---------- Example dataset ----------
  exampleBtn.addEventListener("click", async () => {
    try {
      const resp = await fetch("/load_example", { method: "POST" });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.error || "Example load failed");

      loaded = true;
      populateColumnDropdowns(data.columns);

      previewDiv.innerHTML = `
        <p><strong>Example dataset loaded.</strong> Shape: ${data.shape[0]} x ${data.shape[1]}</p>
        ${data.preview_html}
      `;

      // auto-fill placeholders to help user
      featuresInput.value = "feature1,feature2";
      targetInput.value = "target";

      activeModels = [];
      clearMetricsTable();
      plot2dDiv.innerHTML = "";
      plot3dDiv.innerHTML = `<div style="padding:14px;color:#56607a;">Enter exactly 3 features to enable 3D plot.</div>`;
    } catch (err) {
      alert(err.message);
    }
  });

  // ---------- Model buttons ----------
  btnLR.addEventListener("click", () => runModel("lr"));
  btnSVM.addEventListener("click", () => runModel("svm"));
  btnCluster.addEventListener("click", () => runModel("cluster"));
  btnNN.addEventListener("click", () => runModel("nn"));
  btnKNN.addEventListener("click", () => runModel("knn"));

  btnReset.addEventListener("click", () => {
    activeModels = [];
    clearMetricsTable();
    plot2dDiv.innerHTML = "";
    plot3dDiv.innerHTML = `<div style="padding:14px;color:#56607a;">Enter exactly 3 features to enable 3D plot.</div>`;
  });

  // Initial placeholder
  plot3dDiv.innerHTML = `<div style="padding:14px;color:#56607a;">Enter exactly 3 features to enable 3D plot.</div>`;

  // HISTOGRAM
  const btnHistogram = document.getElementById("btn-histogram");
  const btnGauge = document.getElementById("btn-gauge");

  btnHistogram.addEventListener("click", () => {
    if (!loaded) {
      alert("Load a dataset first.");
      return;
    }
    const features = getSelectedFeatures();
    if (!features.length) {
      alert("Enter feature columns first.");
      return;
    }
    renderHistograms(features);
  });

  btnGauge.addEventListener("click", () => {
    if (!loaded) {
      alert("Load a dataset first.");
      return;
    }
    const features = getSelectedFeatures();
    if (!features.length) {
      alert("Enter feature columns first.");
      return;
    }
    runGaugeRR(features);
  });

});
// END 
// ----

// PART 3: HISTOGRAM 
// -------------------
async function renderHistograms(features) {
  const container = document.getElementById("histogram-container");
  container.innerHTML = ""; // remove placeholder text

  for (const feature of features) {
    const div = document.createElement("div");
    div.className = "plot-box";
    container.appendChild(div);

    const res = await fetch("/histogram", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ column: feature })
    });

    const data = await res.json();

    if (data.error) {
      div.innerHTML = `<p>${data.error}</p>`;
      continue;
    }

    Plotly.newPlot(div, data.figure.data, data.figure.layout, {
      responsive: true
    });
  }
}
// END 
// ------

// PART 4: GAUGE R&R OVERVIEW
// ----------------------------
async function runGaugeRR(features) {
  const res = await fetch("/gauge_rr", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ features })
  });

  const data = await res.json();
  const box = document.getElementById("gauge-rr-results");

  box.innerHTML = `
    <h4>Calculation Overview</h4>

    <p><strong>Mean (per operator):</strong><br>
      ${Object.entries(data.means)
        .map(([k,v]) => `${k}: ${v.toFixed(3)}`)
        .join("<br>")}
    </p>

    <p><strong>Range (per operator):</strong><br>
      ${Object.entries(data.ranges)
        .map(([k,v]) => `${k}: ${v.toFixed(3)}`)
        .join("<br>")}
    </p>

    <p><strong>Repeatability (EV):</strong> ${data.repeatability.toFixed(4)}</p>
    <p><strong>Reproducibility (AV):</strong> ${data.reproducibility.toFixed(4)}</p>
    <p><strong>Total Gauge R&R:</strong> ${data.total_grr.toFixed(4)}</p>

    <p class="note">${data.note}</p>
  `;
}
// END
// ------


// PART 5: Control Chart Limits 
// -------------------------------
document.getElementById("btn-control-limits")?.addEventListener("click", async () => {
  const res = await fetch("/control_limits", { method: "POST" });
  const data = await res.json();

  if (data.error) {
    alert(data.error);
    return;
  }

  const box = document.getElementById("control-limits-table");
  box.innerHTML = `
    <table class="metrics-table">
      <tr><th>X̄̄</th><td>${data.Xbarbar}</td></tr>
      <tr><th>R̄</th><td>${data.Rbar}</td></tr>
      <tr><th>UCL</th><td>${data.UCL}</td></tr>
      <tr><th>CL</th><td>${data.CL}</td></tr>
      <tr><th>LCL</th><td>${data.LCL}</td></tr>
    </table>
  `;
});

// PART 6: Cp & Cpk 
// -------------------
document.getElementById("btn-cpk")?.addEventListener("click", async () => {
  const lsl = document.getElementById("lsl-input").value;
  const usl = document.getElementById("usl-input").value;

  const res = await fetch("/cpk", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ lsl, usl })
  });

  const data = await res.json();
  if (data.error) return alert(data.error);

  document.getElementById("cpk-results").innerHTML = `
    <p><strong>Mean:</strong> ${data.mean}</p>
    <p><strong>Std Dev:</strong> ${data.std}</p>
    <p><strong>Cp:</strong> ${data.cp}</p>
    <p><strong>Cpk:</strong> ${data.cpk}</p>


    <p><strong>Cp:</strong> ${data.cp}</p>
    <p><strong>Cpk:</strong> ${data.cpk}</p>
    <p>
      <strong>Industry Judgment:</strong>
      <span style="color:${data.judgment_color};font-weight:bold">
        ${data.judgment}
      </span>
    </p>
  `;

  // FROM PART 7: Including what's above until where there is space
  renderCpkHistogram(data.values, lsl, usl, data.mean);
});


// PART 7: Cp/Cpk Histogram 
// -------------------------
function renderCpkHistogram(values, lsl, usl, mean) {
  const trace = {
    x: values,
    type: "histogram",
    nbinsx: 20,
    marker: { color: "rgba(90,120,250,0.7)" }
  };

  const layout = {
    title: "Process Capability Histogram",
    xaxis: { title: "Measurement" },
    yaxis: { title: "Frequency" },
    shapes: [
      { type: "line", x0: lsl, x1: lsl, y0: 0, y1: 1, yref: "paper",
        line: { color: "red", width: 2, dash: "dash" }},
      { type: "line", x0: usl, x1: usl, y0: 0, y1: 1, yref: "paper",
        line: { color: "red", width: 2, dash: "dash" }},
      { type: "line", x0: mean, x1: mean, y0: 0, y1: 1, yref: "paper",
        line: { color: "blue", width: 2 }}
    ]
  };

  Plotly.newPlot("cpk-histogram", [trace], layout, { responsive: true });
}


// PART 8: Interactive X-R Chart 
// -------------------------------
async function renderXRChart() {
  const res = await fetch("/xr_chart", { method: "POST" });
  const data = await res.json();

  const xbarTrace = {
    x: data.subgroup,
    y: data.means,
    mode: "lines+markers",
    name: "X̄"
  };

  const rTrace = {
    x: data.subgroup,
    y: data.ranges,
    mode: "lines+markers",
    name: "R"
  };

  Plotly.newPlot("xr-chart", [xbarTrace, rTrace], {
    title: "X̄–R Control Chart",
    xaxis: { title: "Subgroup" },
    yaxis: { title: "Value" }
  }, { responsive: true });
}


// =========================
// SIMPLE AUTO IMAGE SLIDER
// =========================
document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".slider").forEach(slider => {
    const slides = slider.querySelectorAll(".slide");
    if (slides.length <= 1) return;

    let index = 0;
    slides[index].classList.add("active");

    setInterval(() => {
      slides[index].classList.remove("active");
      index = (index + 1) % slides.length;
      slides[index].classList.add("active");
    }, 5000); // ⏱️ 5 seconds
  });
});
