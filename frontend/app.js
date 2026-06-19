const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const dropContent = document.getElementById("dropContent");
const preview = document.getElementById("preview");
const results = document.getElementById("results");
const loading = document.getElementById("loading");
const error = document.getElementById("error");
const className = document.getElementById("className");
const gaugeFill = document.getElementById("gaugeFill");
const gaugeText = document.getElementById("gaugeText");
const top3 = document.getElementById("top3");
const compareOriginal = document.getElementById("compareOriginal");
const heatmapPanel = document.getElementById("heatmapPanel");
const heatmapImg = document.getElementById("heatmapImg");

const circumference = 2 * Math.PI * 52;

// Inject SVG gradient for gauge
function injectGaugeGradient() {
  const svg = document.querySelector(".gauge");
  if (!svg) return;
  const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
  defs.innerHTML = `<linearGradient id="gaugeGrad" x1="0%" y1="0%" x2="100%" y2="0%">
    <stop offset="0%" stop-color="#f59e0b"/>
    <stop offset="100%" stop-color="#d97706"/>
  </linearGradient>`;
  svg.prepend(defs);
}
injectGaugeGradient();

function handleFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    showError("Please upload an image file.");
    return;
  }

  error.classList.add("hidden");
  loading.classList.remove("hidden");
  results.classList.add("hidden");
  heatmapPanel.classList.add("hidden");

  const reader = new FileReader();
  reader.onload = (e) => {
    preview.src = e.target.result;
    preview.classList.remove("hidden");
    dropContent.style.display = "none";
    compareOriginal.src = e.target.result;
  };
  reader.readAsDataURL(file);

  const formData = new FormData();
  formData.append("file", file);

  const formData2 = new FormData();
  formData2.append("file", file);

  fetch("/predict", { method: "POST", body: formData })
    .then(r => r.json())
    .then(pred => {
      loading.classList.add("hidden");
      results.classList.remove("hidden");

      className.textContent = pred.class_name.replace(/_/g, " ");

      const pct = Math.min(Math.round(pred.confidence * 100), 100);
      const offset = circumference - (pct / 100) * circumference;
      requestAnimationFrame(() => {
        gaugeFill.style.strokeDashoffset = offset;
      });
      gaugeText.textContent = pct + "%";

      const medals = ["\u{1F947}", "\u{1F948}", "\u{1F949}"];
      top3.innerHTML = `<div class="top3-list">${pred.top_3
        .map((item, i) => {
          const ipct = Math.min(Math.round(item.probability * 100), 100);
          const rank = i + 1;
          return `<div class="top3-item">
            <div class="top3-rank rank-${rank}">${rank}</div>
            <span class="top3-name">${item.class_name.replace(/_/g, " ")}</span>
            <div class="top3-bar-wrap"><div class="top3-bar" style="width:${ipct}%"></div></div>
            <span class="top3-pct">${ipct}%</span>
          </div>`;
        })
        .join("")}</div>`;
    })
    .catch(err => {
      loading.classList.add("hidden");
      showError("Failed to analyze image. Is the server running?");
      console.error(err);
    });

  fetch("/gradcam", { method: "POST", body: formData2 })
    .then(r => r.json())
    .then(cam => {
      if (cam.heatmap_base64) {
        heatmapPanel.classList.remove("hidden");
        heatmapImg.src = "data:image/png;base64," + cam.heatmap_base64;
      }
    })
    .catch(err => console.error("GradCAM failed:", err));
}

function showError(msg) {
  error.textContent = msg;
  error.classList.remove("hidden");
}

dropZone.addEventListener("click", () => fileInput.click());

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("dragover");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("dragover");
});

dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("dragover");
  handleFile(e.dataTransfer.files[0]);
});

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});
