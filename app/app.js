const S = {
  features: [],
  embedding: null,
  scores: [],
  pendingScores: null,
  sliders: [],
  resultCount: 0,
  results: [],
  visibleResults: [],
  activeResultIndex: 0,
  panelOpen: window.matchMedia("(min-width: 960px)").matches,
  previewUrl: null,
};

const BARS = [];
const FEATURE_ROWS = [];
const SLIDER_MAX = 6;
const SLIDER_MIN = -SLIDER_MAX;
const SLIDER_STEP = 0.05;
const AUTO_PREFILL_MAX = 4.5;
const RESET_ICON = `
  <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
    <path d="M20 11a8 8 0 1 1-2.34-5.66L20 8" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"></path>
    <path d="M20 4v4h-4" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"></path>
  </svg>
`;

const E = {
  body: document.body,
  metaResults: document.getElementById("meta-results"),
  featureCount: document.getElementById("feature-count"),
  uploadZone: document.getElementById("upload-zone"),
  fileInput: document.getElementById("file-input"),
  uploadEmpty: document.getElementById("upload-empty"),
  uploadLoaded: document.getElementById("upload-loaded"),
  queryImg: document.getElementById("query-img"),
  resetAllBtn: document.getElementById("reset-all-btn"),
  featureList: document.getElementById("feature-list"),
  panelToggle: document.getElementById("panel-toggle"),
  panelToggleIcon: document.getElementById("panel-toggle-icon"),
  emptyState: document.getElementById("empty-state"),
  loadingState: document.getElementById("loading-state"),
  imageGrid: document.getElementById("image-grid"),
  featureModal: document.getElementById("feature-modal"),
  featureModalBackdrop: document.getElementById("feature-modal-backdrop"),
  featureModalClose: document.getElementById("feature-modal-close"),
  featureModalTitle: document.getElementById("feature-modal-title"),
  featureModalDescription: document.getElementById("feature-modal-description"),
  featureModalBody: document.getElementById("feature-modal-body"),
  featureModalHigh: document.getElementById("feature-modal-high"),
  featureModalLow: document.getElementById("feature-modal-low"),
  resultViewer: document.getElementById("result-viewer"),
  resultViewerBackdrop: document.getElementById("result-viewer-backdrop"),
  resultViewerClose: document.getElementById("result-viewer-close"),
  resultViewerPrev: document.getElementById("result-viewer-prev"),
  resultViewerNext: document.getElementById("result-viewer-next"),
  resultViewerImage: document.getElementById("result-viewer-image"),
  resultViewerTitle: document.getElementById("result-viewer-title"),
  resultViewerPath: document.getElementById("result-viewer-path"),
  resultViewerDownload: document.getElementById("result-viewer-download"),
  resultViewerCopyPath: document.getElementById("result-viewer-copy-path"),
};

function debounce(fn, ms) {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), ms);
  };
}

function fmt(v) {
  return `${v >= 0 ? "+" : ""}${v.toFixed(2)}`;
}

function quantizeSliderValue(v) {
  let value = Math.max(SLIDER_MIN, Math.min(SLIDER_MAX, Number.isFinite(v) ? v : 0));
  value = Math.round(value / SLIDER_STEP) * SLIDER_STEP;
  return Math.round(value * 100) / 100;
}

function scoresToAutoSliderValues(scores) {
  const numericScores = scores.map((score) => (Number.isFinite(score) ? score : 0));
  const maxAbs = numericScores.reduce((best, score) => Math.max(best, Math.abs(score)), 0);

  if (maxAbs < 1e-6) {
    return numericScores.map(() => 0);
  }

  return numericScores.map((score) => {
    if (Math.abs(score) < 1e-6) {
      return 0;
    }

    const ratio = Math.abs(score) / maxAbs;
    const expanded = Math.pow(ratio, 0.65);
    const scaled = Math.sign(score) * expanded * AUTO_PREFILL_MAX;
    const quantized = quantizeSliderValue(scaled);

    if (quantized === 0) {
      return Math.sign(score) * SLIDER_STEP;
    }

    return quantized;
  });
}

function activeSliderCount() {
  return S.sliders.filter((value) => Math.abs(value) >= 0.05).length;
}

function updateMeta() {
  E.metaResults.textContent = S.embedding ? String(S.visibleResults.length) : "-";
  E.featureCount.textContent = `${activeSliderCount()}/${S.features.length} active`;
}

function setStatus(status) {
  E.body.dataset.status = status.toLowerCase();
}

function setPanelOpen(open) {
  S.panelOpen = open;
  E.body.dataset.panel = open ? "open" : "closed";
  E.panelToggleIcon.textContent = open ? "<" : ">";
  E.panelToggle.setAttribute("aria-label", open ? "Collapse panel" : "Expand panel");
  E.panelToggle.title = open ? "Collapse panel" : "Expand panel";
}

function showEmptyCanvas() {
  E.emptyState.hidden = false;
  E.loadingState.hidden = true;
  E.imageGrid.hidden = true;
}

function showLoadingCanvas() {
  E.emptyState.hidden = true;
  E.loadingState.hidden = false;
  E.imageGrid.hidden = true;
}

function showGridCanvas() {
  E.emptyState.hidden = true;
  E.loadingState.hidden = true;
  E.imageGrid.hidden = false;
}

function isResultViewerOpen() {
  return !E.resultViewer.hidden;
}

function isFeatureModalOpen() {
  return !E.featureModal.hidden;
}

function renderFeatureExampleList(target, items) {
  target.innerHTML = "";

  if (!items || !items.length) {
    const empty = document.createElement("div");
    empty.className = "feature-example-empty";
    empty.textContent = "No examples available for this feature.";
    target.appendChild(empty);
    return;
  }

  items.forEach((item, index) => {
    const card = document.createElement("div");
    card.className = "feature-example-card";

    const top = document.createElement("div");
    top.className = "feature-example-top";

    const label = document.createElement("span");
    label.className = "feature-example-label";
    label.textContent = `Example ${String(index + 1).padStart(2, "0")}`;

    const value = document.createElement("span");
    value.className = `feature-example-value mono ${item.value >= 0 ? "pos" : "neg"}`;
    value.textContent = fmt(item.value);

    top.appendChild(label);
    top.appendChild(value);

    const img = new Image();
    img.src = item.image;
    img.alt = "";
    img.loading = "lazy";

    const meta = document.createElement("div");
    meta.className = "feature-example-meta";

    const path = document.createElement("span");
    path.className = "feature-example-path";
    path.textContent = item.path
      ? item.path.split(/[/\\]/).pop() || "Image"
      : "Image";

    meta.appendChild(path);
    card.appendChild(top);
    card.appendChild(img);
    card.appendChild(meta);
    target.appendChild(card);
  });
}

function openFeatureModal(detail) {
  E.featureModalTitle.textContent = detail.name || "Feature";
  E.featureModalDescription.textContent = detail.description || "No description available.";
  renderFeatureExampleList(E.featureModalHigh, detail.high_examples || []);
  renderFeatureExampleList(E.featureModalLow, detail.low_examples || []);
  E.featureModal.hidden = false;
  E.featureModal.setAttribute("aria-hidden", "false");
}

function closeFeatureModal() {
  E.featureModal.hidden = true;
  E.featureModal.setAttribute("aria-hidden", "true");
}

function renderResultViewer() {
  const item = S.visibleResults[S.activeResultIndex];
  if (!item) return;

  E.resultViewerImage.src = item.download || item.image;
  E.resultViewerTitle.textContent = `R${String(S.activeResultIndex + 1).padStart(2, "0")}`;
  E.resultViewerPath.textContent = item.path || "Path unavailable";
  E.resultViewerDownload.href = item.download || item.image;
  E.resultViewerDownload.download = item.path
    ? item.path.split(/[/\\]/).pop() || `slider-result-${S.activeResultIndex + 1}.jpg`
    : `slider-result-${S.activeResultIndex + 1}.jpg`;
  E.resultViewerPrev.disabled = S.visibleResults.length <= 1;
  E.resultViewerNext.disabled = S.visibleResults.length <= 1;
}

function openResultViewer(index) {
  if (!S.visibleResults.length) return;
  S.activeResultIndex = Math.max(0, Math.min(index, S.visibleResults.length - 1));
  renderResultViewer();
  E.resultViewer.hidden = false;
  E.resultViewer.setAttribute("aria-hidden", "false");
}

function closeResultViewer() {
  E.resultViewer.hidden = true;
  E.resultViewer.setAttribute("aria-hidden", "true");
}

function stepResultViewer(direction) {
  if (!S.visibleResults.length) return;
  const total = S.visibleResults.length;
  S.activeResultIndex = (S.activeResultIndex + direction + total) % total;
  renderResultViewer();
}

async function showFeatureDetail(index) {
  const detail = S.features[index];
  if (!detail) return;
  openFeatureModal(detail);
}

function revokePreviewUrlIfNeeded() {
  if (S.previewUrl && S.previewUrl.startsWith("blob:")) {
    URL.revokeObjectURL(S.previewUrl);
  }
}

function showQueryPreview(src) {
  revokePreviewUrlIfNeeded();
  S.previewUrl = src;
  E.queryImg.src = src;
  E.uploadEmpty.hidden = true;
  E.uploadLoaded.hidden = false;
}

function clearQueryPreview() {
  revokePreviewUrlIfNeeded();
  S.previewUrl = null;
  E.queryImg.removeAttribute("src");
  E.uploadEmpty.hidden = false;
  E.uploadLoaded.hidden = true;
}

function sortFeatureRows() {
  if (!FEATURE_ROWS.length) return;

  FEATURE_ROWS.sort((a, b) => {
    const activeA = Math.abs(S.sliders[a.index] || 0);
    const activeB = Math.abs(S.sliders[b.index] || 0);
    if (activeB !== activeA) return activeB - activeA;

    const scoreA = Math.abs(S.scores[a.index] || 0);
    const scoreB = Math.abs(S.scores[b.index] || 0);
    if (scoreB !== scoreA) return scoreB - scoreA;

    return a.index - b.index;
  });

  FEATURE_ROWS.forEach(({ row }) => {
    E.featureList.appendChild(row);
  });
}

function updateScores(scores) {
  scores.forEach((score, index) => {
    const target = document.getElementById(`feature-score-${index}`);
    if (!target) return;

    if (Math.abs(score) < 0.03) {
      target.textContent = "";
      target.className = "feature-score mono";
      return;
    }

    target.textContent = fmt(score);
    target.className = `feature-score mono ${score >= 0 ? "pos" : "neg"}`;
  });
}

function updateFeatureSteeringState(index) {
  const value = S.sliders[index] || 0;
  const row = document.getElementById(`feature-row-${index}`);
  const valueEl = document.getElementById(`steer-value-${index}`);
  const resetBtn = document.getElementById(`feature-reset-${index}`);

  if (row) {
    row.dataset.active = Math.abs(value) >= 0.05 ? "true" : "false";
  }

  if (valueEl) {
    valueEl.textContent = fmt(value);
    valueEl.className =
      Math.abs(value) > 0.005
        ? `steer-value mono ${value >= 0 ? "pos" : "neg"}`
        : "steer-value mono";
  }

  if (resetBtn) {
    resetBtn.disabled = Math.abs(value) <= 0.005;
  }
}

function applyAutoSliderScores(scores) {
  if (!BARS.length) {
    return false;
  }

  const autoSliderValues = scoresToAutoSliderValues(scores);
  BARS.forEach((bar, index) => {
    const nextValue = autoSliderValues[index] || 0;
    bar.setValue(nextValue, false);
  });

  updateScores(scores);
  sortFeatureRows();
  updateMeta();
  return true;
}

class BipolarBar {
  constructor(index, onCommit) {
    this.index = index;
    this.onCommit = onCommit;
    this.value = 0;
    this.dragging = false;
    this.dragStartX = 0;
    this.dragStartValue = 0;
    this.build();
    this.bind();
  }

  build() {
    this.el = document.createElement("div");
    this.el.className = "bipolar-bar";
    this.el.tabIndex = 0;

    const ticks = document.createElement("div");
    ticks.className = "bipolar-ticks";
    ticks.innerHTML = "<i></i><i></i><i></i>";

    this.track = document.createElement("div");
    this.track.className = "bipolar-track";

    this.fill = document.createElement("div");
    this.fill.className = "bipolar-fill";

    const center = document.createElement("div");
    center.className = "bipolar-center";

    this.track.appendChild(this.fill);
    this.track.appendChild(center);
    this.el.appendChild(ticks);
    this.el.appendChild(this.track);
    this.render();
  }

  bind() {
    this.el.addEventListener("mousedown", (event) => {
      event.preventDefault();
      this.dragging = true;
      this.dragStartX = event.clientX;
      this.dragStartValue = this.value;
      this.fill.style.transition = "none";
      document.addEventListener("mousemove", this.handleMouseMove);
      document.addEventListener("mouseup", this.handleMouseUp);
    });

    this.el.addEventListener("touchstart", (event) => {
      this.dragging = true;
      this.dragStartX = event.touches[0].clientX;
      this.dragStartValue = this.value;
      this.fill.style.transition = "none";
    }, { passive: true });

    this.el.addEventListener("touchmove", (event) => {
      if (!this.dragging) return;
      event.preventDefault();
      const dx = event.touches[0].clientX - this.dragStartX;
      const width = this.track.getBoundingClientRect().width || 1;
      this.update(this.dragStartValue + (dx / width) * (SLIDER_MAX - SLIDER_MIN), false);
    }, { passive: false });

    this.el.addEventListener("touchend", () => {
      if (!this.dragging) return;
      this.dragging = false;
      this.fill.style.transition = "";
      this.onCommit(this.value);
    });

    this.el.addEventListener("dblclick", () => {
      this.setValue(0);
    });

    this.el.addEventListener("keydown", (event) => {
      const step = event.shiftKey ? 0.5 : 0.1;
      if (event.key === "ArrowRight") {
        event.preventDefault();
        this.setValue(this.value + step);
      }
      if (event.key === "ArrowLeft") {
        event.preventDefault();
        this.setValue(this.value - step);
      }
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        this.setValue(0);
      }
    });
  }

  handleMouseMove = (event) => {
    if (!this.dragging) return;
    const dx = event.clientX - this.dragStartX;
    const width = this.track.getBoundingClientRect().width || 1;
    this.update(this.dragStartValue + (dx / width) * (SLIDER_MAX - SLIDER_MIN), false);
  };

  handleMouseUp = () => {
    if (!this.dragging) return;
    this.dragging = false;
    this.fill.style.transition = "";
    document.removeEventListener("mousemove", this.handleMouseMove);
    document.removeEventListener("mouseup", this.handleMouseUp);
    this.onCommit(this.value);
  };

  update(nextValue, notify = true) {
    const value = quantizeSliderValue(nextValue);
    this.value = value;
    S.sliders[this.index] = value;
    updateFeatureSteeringState(this.index);
    updateMeta();
    sortFeatureRows();
    this.render();
    if (notify) {
      this.onCommit(value);
    }
  }

  setValue(nextValue, notify = true) {
    this.update(nextValue, notify);
  }

  render() {
    const width = Math.abs(this.value / SLIDER_MAX) * 50;
    this.fill.style.width = `${width}%`;
    this.fill.style.background = this.value >= 0 ? "var(--signal)" : "var(--alert)";
    this.fill.style.left = this.value >= 0 ? "50%" : `${50 - width}%`;
  }
}

function buildFeatureRows(features) {
  E.featureList.innerHTML = "";
  BARS.length = 0;
  FEATURE_ROWS.length = 0;
  S.sliders = new Array(features.length).fill(0);

  features.forEach((feature, index) => {
    const row = document.createElement("div");
    row.className = "feature-row";
    row.id = `feature-row-${index}`;
    row.dataset.active = "false";

    const idx = document.createElement("div");
    idx.className = "feature-index mono";
    idx.textContent = String(index + 1).padStart(2, "0");

    const preview = document.createElement("button");
    preview.className = "feature-thumb feature-thumb-button present";
    preview.type = "button";
    preview.title = `Open ${feature.name} examples`;
    preview.setAttribute("aria-label", `Open ${feature.name} examples`);
    preview.addEventListener("click", () => {
      showFeatureDetail(index);
    });
    if (feature.preview_present || feature.preview_absent) {
      const img = new Image();
      img.src = feature.preview_present || feature.preview_absent;
      img.alt = "";
      preview.appendChild(img);
    }

    const main = document.createElement("div");
    main.className = "feature-main";

    const top = document.createElement("div");
    top.className = "feature-main-top";

    const copy = document.createElement("div");
    copy.className = "feature-copy";

    const trigger = document.createElement("button");
    trigger.className = "feature-label-trigger";
    trigger.type = "button";
    trigger.setAttribute("aria-label", `Show full description for ${feature.name}`);

    const name = document.createElement("span");
    name.className = "feature-name";
    name.textContent = feature.name;
    name.title = feature.description || feature.name;

    const description = document.createElement("div");
    description.className = "feature-description";
    description.textContent = feature.description || "";

    trigger.appendChild(name);
    if (feature.description) {
      trigger.appendChild(description);
    }
    copy.appendChild(trigger);

    const popover = document.createElement("div");
    popover.className = "feature-popover";
    popover.innerHTML = `
      <h4 class="feature-popover-title">${feature.name}</h4>
      <p class="feature-popover-desc">${feature.description || "No description available for this feature."}</p>
    `;
    copy.appendChild(popover);

    const meta = document.createElement("div");
    meta.className = "feature-meta";

    const score = document.createElement("span");
    score.className = "feature-score mono";
    score.id = `feature-score-${index}`;

    const steer = document.createElement("span");
    steer.className = "steer-value mono";
    steer.id = `steer-value-${index}`;
    steer.textContent = fmt(0);

    const reset = document.createElement("button");
    reset.className = "feature-reset";
    reset.id = `feature-reset-${index}`;
    reset.innerHTML = RESET_ICON;
    reset.disabled = true;
    reset.title = `Reset ${feature.name}`;
    reset.setAttribute("aria-label", `Reset ${feature.name}`);
    reset.addEventListener("click", (event) => {
      event.stopPropagation();
      BARS[index].setValue(0);
    });

    meta.appendChild(score);
    meta.appendChild(steer);
    meta.appendChild(reset);

    top.appendChild(copy);
    top.appendChild(meta);
    main.appendChild(top);

    const bar = new BipolarBar(index, debounce(() => retrieve(), 120));
    BARS.push(bar);
    main.appendChild(bar.el);

    row.appendChild(idx);
    row.appendChild(preview);
    row.appendChild(main);
    E.featureList.appendChild(row);
    FEATURE_ROWS.push({ index, row });
    updateFeatureSteeringState(index);
  });

  updateMeta();
  sortFeatureRows();

  if (S.pendingScores && S.pendingScores.length) {
    applyAutoSliderScores(S.pendingScores);
  }
}

function renderGrid(images) {
  S.results = images;
  S.visibleResults = images.slice();
  E.imageGrid.innerHTML = "";

  S.visibleResults.forEach((item, index) => {
    const button = document.createElement("button");
    button.className = "grid-item";

    const img = new Image();
    img.src = item.image;
    img.alt = `Result ${index + 1}`;
    img.loading = "lazy";

    const badge = document.createElement("span");
    badge.className = "grid-index mono micro";
    badge.textContent = `R${String(index + 1).padStart(2, "0")}`;

    const overlay = document.createElement("div");
    overlay.className = "grid-overlay";
    overlay.innerHTML = `<span class="micro mono">open</span>`;

    button.appendChild(img);
    button.appendChild(badge);
    button.appendChild(overlay);
    button.addEventListener("click", () => openResultViewer(index));

    E.imageGrid.appendChild(button);
  });

  S.resultCount = S.results.length;
  updateMeta();
  showGridCanvas();
}

const API = {
  async request(path, options = {}) {
    const response = await fetch(path, options);
    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || `Request failed: ${response.status}`);
    }
    return response.json();
  },

  features() {
    return this.request("/api/features");
  },

  encode(file) {
    const form = new FormData();
    form.append("file", file);
    return this.request("/api/encode", { method: "POST", body: form });
  },

  retrieve(embedding, sliders, k = 30) {
    return this.request("/api/retrieve", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        embedding,
        sliders,
        k,
      }),
    });
  },
};

async function handleFile(file) {
  setStatus("encoding");
  showLoadingCanvas();

  try {
    const { embedding, scores } = await API.encode(file);
    S.embedding = embedding;
    S.scores = scores;
    S.pendingScores = scores.slice();
    E.resetAllBtn.disabled = false;

    applyAutoSliderScores(scores);
    await retrieve();
  } catch (error) {
    console.error("Encode failed:", error);
    S.resultCount = 0;
    updateMeta();
    setStatus("error");
    showEmptyCanvas();
  }
}

async function retrieve() {
  if (!S.embedding) return;

  setStatus(activeSliderCount() > 0 ? "steering" : "retrieving");
  showLoadingCanvas();

  try {
    const { images } = await API.retrieve(S.embedding, S.sliders, 30);
    renderGrid(images);
    setStatus(activeSliderCount() > 0 ? "steering" : "ranked");
  } catch (error) {
    console.error("Retrieve failed:", error);
    S.resultCount = 0;
    updateMeta();
    setStatus("error");
    showEmptyCanvas();
  }
}

function resetAll() {
  BARS.forEach((bar) => bar.setValue(0, false));
  S.sliders.fill(0);
  updateMeta();
  sortFeatureRows();
  if (S.embedding) {
    retrieve();
  }
}

function initUploadZone() {
  E.uploadZone.addEventListener("click", () => E.fileInput.click());
  E.uploadZone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      E.fileInput.click();
    }
  });

  E.fileInput.addEventListener("change", () => {
    const file = E.fileInput.files && E.fileInput.files[0];
    if (!file) return;
    const preview = URL.createObjectURL(file);
    showQueryPreview(preview);
    handleFile(file);
  });

  E.uploadZone.addEventListener("dragover", (event) => {
    event.preventDefault();
    E.uploadZone.classList.add("drag-over");
  });

  E.uploadZone.addEventListener("dragleave", () => {
    E.uploadZone.classList.remove("drag-over");
  });

  E.uploadZone.addEventListener("drop", (event) => {
    event.preventDefault();
    E.uploadZone.classList.remove("drag-over");
    const file = event.dataTransfer.files && event.dataTransfer.files[0];
    if (!file || !file.type.startsWith("image/")) return;
    const preview = URL.createObjectURL(file);
    showQueryPreview(preview);
    handleFile(file);
  });
}

function initPanelToggle() {
  setPanelOpen(S.panelOpen);
  E.panelToggle.addEventListener("click", () => {
    setPanelOpen(!S.panelOpen);
  });
}

function initFeatureModal() {
  E.featureModalClose.addEventListener("click", closeFeatureModal);
  E.featureModalBackdrop.addEventListener("click", closeFeatureModal);
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !E.featureModal.hidden) {
      closeFeatureModal();
    }
  });
}

function initResultViewer() {
  E.resultViewerClose.addEventListener("click", closeResultViewer);
  E.resultViewerBackdrop.addEventListener("click", closeResultViewer);
  E.resultViewerPrev.addEventListener("click", () => stepResultViewer(-1));
  E.resultViewerNext.addEventListener("click", () => stepResultViewer(1));
  E.resultViewerCopyPath.addEventListener("click", async () => {
    const item = S.visibleResults[S.activeResultIndex];
    if (!item || !item.path) return;
    try {
      await navigator.clipboard.writeText(item.path);
      E.resultViewerCopyPath.textContent = "Copied";
      setTimeout(() => {
        E.resultViewerCopyPath.textContent = "Copy path";
      }, 1200);
    } catch (error) {
      console.error("Copy path failed:", error);
    }
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && isResultViewerOpen()) {
      closeResultViewer();
      return;
    }
    if (!isResultViewerOpen() || isFeatureModalOpen()) return;
    if (event.key === "ArrowRight") {
      event.preventDefault();
      stepResultViewer(1);
    }
    if (event.key === "ArrowLeft") {
      event.preventDefault();
      stepResultViewer(-1);
    }
  });
}

async function init() {
  initPanelToggle();
  initUploadZone();
  initFeatureModal();
  initResultViewer();

  E.resetAllBtn.addEventListener("click", resetAll);
  updateMeta();
  clearQueryPreview();
  showEmptyCanvas();
  setStatus("idle");

  try {
    const { features } = await API.features();
    S.features = features;
    buildFeatureRows(features);
  } catch (error) {
    console.error("Feature load failed:", error);
    setStatus("error");
  }
}

document.addEventListener("DOMContentLoaded", init);
