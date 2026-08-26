/* Neural Style Transfer - frontend.
   Uploads two images, starts a job, polls it, and swaps in the live preview. */

const $ = (id) => document.getElementById(id);

const POLL_MS = 900;

const state = {
  content: null,
  style: null,
  jobId: null,
  timer: null,
  lastPreview: -1,
};

/* ------------------------------------------------------------ formatting */

const fmtLoss = (v) => {
  if (v === undefined || v === null) return "—";
  const abs = Math.abs(v);
  if (abs >= 1e6) return (v / 1e6).toFixed(2) + "M";
  if (abs >= 1e3) return (v / 1e3).toFixed(1) + "k";
  return v.toFixed(1);
};

const fmtWeight = (exp) => {
  const v = Math.pow(10, exp);
  if (v >= 1000) return v.toExponential(0).replace("e+", "e");
  return v < 10 ? v.toFixed(1) : Math.round(v).toString();
};

/* ------------------------------------------------------------- drop zones */

function wireDrop(kind) {
  const zone = $(`drop-${kind}`);
  const input = $(`file-${kind}`);
  const img = zone.querySelector(".drop__img");

  const accept = (file) => {
    if (!file || !file.type.startsWith("image/")) {
      showError("That file is not an image. Use JPEG, PNG or WEBP.");
      return;
    }
    state[kind] = file;
    img.src = URL.createObjectURL(file);
    img.alt = `${kind} image preview`;
    zone.classList.add("has-image");
    clearError();
    refreshRunButton();
  };

  input.addEventListener("change", () => accept(input.files[0]));

  ["dragenter", "dragover"].forEach((ev) =>
    zone.addEventListener(ev, (e) => {
      e.preventDefault();
      zone.classList.add("is-dragging");
    })
  );
  ["dragleave", "drop"].forEach((ev) =>
    zone.addEventListener(ev, (e) => {
      e.preventDefault();
      zone.classList.remove("is-dragging");
    })
  );
  zone.addEventListener("drop", (e) => accept(e.dataTransfer.files[0]));
}

/* ---------------------------------------------------------------- options */

async function loadOptions() {
  const labels = {
    "vgg16": "VGG16",
    "vgg16-experimental": "VGG16 (wide taps)",
    "vgg19": "VGG19 — paper",
    "lbfgs": "L-BFGS",
    "adam": "Adam",
    "content": "Content image",
    "random": "Noise",
    "style": "Style image",
  };

  try {
    const res = await fetch("/api/options");
    if (!res.ok) throw new Error(res.statusText);
    const opt = await res.json();

    const fill = (el, values, selected) => {
      el.innerHTML = "";
      values.forEach((v) => {
        const o = document.createElement("option");
        o.value = v;
        o.textContent = labels[v] || v;
        if (v === selected) o.selected = true;
        el.appendChild(o);
      });
    };

    fill($("model"), opt.models, "vgg19");
    fill($("optimizer"), opt.optimizers, "lbfgs");
    fill($("init-method"), opt.init_methods, "content");

    $("height").max = opt.max_height;
    $("iterations").max = opt.max_iterations;
    $("iterations").value = opt.default_iterations.lbfgs;

    $("optimizer").addEventListener("change", (e) => {
      $("iterations").value = opt.default_iterations[e.target.value];
    });

    const badge = $("device-badge");
    badge.textContent = opt.device === "cpu" ? "cpu — expect minutes" : `${opt.device} — fast`;
    badge.removeAttribute("data-empty");
  } catch (err) {
    showError("Cannot reach the server. Is it running?");
  }
}

/* ----------------------------------------------------------------- sliders */

function wireSliders() {
  const bind = (id, out, fn) => {
    const el = $(id);
    const update = () => ($(out).textContent = fn(parseFloat(el.value)));
    el.addEventListener("input", update);
    update();
  };
  bind("height", "height-out", (v) => Math.round(v));
  bind("content-weight", "content-weight-out", fmtWeight);
  bind("style-weight", "style-weight-out", fmtWeight);
  bind("tv-weight", "tv-weight-out", (v) => (v % 1 === 0 ? v.toFixed(0) : v.toFixed(1)));
}

/* -------------------------------------------------------------------- run */

function refreshRunButton() {
  const btn = $("run");
  const ready = state.content && state.style;
  btn.disabled = !ready || state.jobId !== null;
  if (state.jobId) btn.textContent = "Running…";
  else btn.textContent = ready ? "Run transfer" : "Add both images";
}

function showError(msg) {
  const el = $("error");
  el.textContent = msg;
  el.hidden = false;
}
function clearError() {
  $("error").hidden = true;
}

async function startTransfer() {
  clearError();

  const body = new FormData();
  body.append("content", state.content);
  body.append("style", state.style);
  body.append("height", $("height").value);
  body.append("content_weight", Math.pow(10, parseFloat($("content-weight").value)));
  body.append("style_weight", Math.pow(10, parseFloat($("style-weight").value)));
  body.append("tv_weight", $("tv-weight").value);
  body.append("model", $("model").value);
  body.append("optimizer", $("optimizer").value);
  body.append("init_method", $("init-method").value);
  body.append("iterations", $("iterations").value);

  try {
    const res = await fetch("/api/transfer", { method: "POST", body });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "The server rejected the request.");

    state.jobId = data.id;
    state.lastPreview = -1;
    $("readout").hidden = false;
    $("output-actions").hidden = true;
    $("cancel").hidden = false;
    $("stage").dataset.state = "running";
    $("stage-empty").hidden = true;
    refreshRunButton();
    poll();
  } catch (err) {
    showError(err.message);
  }
}

async function cancelTransfer() {
  if (!state.jobId) return;
  await fetch(`/api/jobs/${state.jobId}/cancel`, { method: "POST" });
}

/* ------------------------------------------------------------------- poll */

function renderProgress(job) {
  $("readout-step").textContent = `${job.iteration} / ${job.total_iterations}`;
  $("readout-elapsed").textContent = `${job.elapsed.toFixed(1)}s`;
  $("bar-fill").style.width = `${(job.progress * 100).toFixed(1)}%`;

  const l = job.losses || {};
  $("loss-content").textContent = fmtLoss(l.content);
  $("loss-style").textContent = fmtLoss(l.style);
  $("loss-tv").textContent = fmtLoss(l.tv);
  $("loss-total").textContent = fmtLoss(l.total);

  // The budget bar shows which term currently dominates the total loss.
  const parts = [l.content || 0, l.style || 0, l.tv || 0];
  const sum = parts.reduce((a, b) => a + b, 0) || 1;
  const segs = $("budget").children;
  parts.forEach((p, i) => {
    segs[i].style.flexGrow = Math.max(p / sum, 0.01);
  });

  if (job.preview_version > state.lastPreview) {
    state.lastPreview = job.preview_version;
    const img = $("result-img");
    img.src = `/api/jobs/${job.id}/preview?v=${job.preview_version}`;
    img.hidden = false;
  }
}

function finish(job) {
  clearInterval(state.timer);
  state.timer = null;
  state.jobId = null;
  $("cancel").hidden = true;
  $("stage").dataset.state = "idle";
  refreshRunButton();

  if (job.status === "done") {
    const img = $("result-img");
    img.src = `/api/jobs/${job.id}/result`;
    img.hidden = false;
    $("download").href = `/api/jobs/${job.id}/result`;
    $("output-actions").hidden = false;
    const s = job.settings;
    $("output-meta").textContent =
      `${s.model} · ${s.optimizer} · ${s.iterations} iters · ${s.height}px · ${job.elapsed.toFixed(1)}s`;
  } else if (job.status === "cancelled") {
    showError("Stopped. The partial preview above is the last state reached.");
  } else {
    showError(job.error || "The run failed.");
  }
}

function poll() {
  const tick = async () => {
    if (!state.jobId) return;
    try {
      const res = await fetch(`/api/jobs/${state.jobId}`);
      if (!res.ok) throw new Error("Lost track of the job. It may have expired.");
      const job = await res.json();
      renderProgress(job);
      if (["done", "failed", "cancelled"].includes(job.status)) finish(job);
    } catch (err) {
      clearInterval(state.timer);
      state.timer = null;
      state.jobId = null;
      refreshRunButton();
      showError(err.message);
    }
  };
  tick();
  state.timer = setInterval(tick, POLL_MS);
}

/* ------------------------------------------------------------------- init */

wireDrop("content");
wireDrop("style");
wireSliders();
loadOptions();
refreshRunButton();

$("run").addEventListener("click", startTransfer);
$("cancel").addEventListener("click", cancelTransfer);
