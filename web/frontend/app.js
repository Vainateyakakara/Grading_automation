const predictBtn = document.getElementById("predictBtn");
const batchBtn = document.getElementById("batchBtn");
const statusBtn = document.getElementById("statusBtn");

const predictResult = document.getElementById("predictResult");
const batchResult = document.getElementById("batchResult");
const statusPanel = document.getElementById("statusPanel");

async function loadModelInfo() {
  try {
    const res = await fetch("/api/model-info");
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || "Unable to read model info");
    }
    statusPanel.textContent = JSON.stringify(data, null, 2);
  } catch (err) {
    statusPanel.textContent = `Error: ${err.message}`;
  }
}

predictBtn?.addEventListener("click", async () => {
  const reference = document.getElementById("reference").value;
  const student = document.getElementById("student").value;

  predictResult.textContent = "Scoring...";

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ reference_answer: reference, student_answer: student }),
    });

    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || "Prediction failed");
    }

    const src = data.source_model ? ` | source: ${data.source_model}` : "";
    predictResult.textContent = `Predicted score: ${data.predicted_score.toFixed(3)} (range: ${data.min_score} - ${data.max_score})${src}`;
  } catch (err) {
    predictResult.textContent = `Error: ${err.message}`;
  }
});

batchBtn?.addEventListener("click", async () => {
  const fileInput = document.getElementById("batchFile");
  const file = fileInput.files?.[0];

  if (!file) {
    batchResult.textContent = "Choose a CSV file first.";
    return;
  }

  batchResult.textContent = "Processing batch...";

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch("/api/predict-batch", {
      method: "POST",
      body: formData,
    });

    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || "Batch prediction failed");
    }

    batchResult.textContent = `Done. Rows: ${data.rows}. Output: ${data.output_file}`;
  } catch (err) {
    batchResult.textContent = `Error: ${err.message}`;
  }
});

statusBtn?.addEventListener("click", loadModelInfo);

loadModelInfo();
