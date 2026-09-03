/* ptm-ipf web UI front end.  Plain JavaScript, no dependencies.
 *
 * The server holds the expensive state (the PTM result); this side holds only
 * presentation state: the camera, the selection criteria being edited, and
 * the snapshot of the analysis settings that produced the current result so
 * that colour-only changes can be applied without re-running PTM.
 */
"use strict";

const $ = (id) => document.getElementById(id);

const state = {
  meta: null,
  camera: { az: -125, el: 20, zoom: 1.0 },
  analysed: null,       // analysis settings of the cached server result
  generation: -1,
  running: false,
  selectionCount: null,
  criteria: [],
  atomInfo: null,
  command: null,       // last /api/command payload, for the one-line copy
  columns: null,       // the file's columns and the mapping controls built from them
  customColormap: null,  // name of a colour scale uploaded this session
  colormapTarget: null,  // which select opened the upload dialog
  status: { text: "idle", kind: "" },   // what the header says when nothing is running
  slice: { axis: null, low: null, high: null },  // extent of the atoms along the slice normal
  series: null,          // the frames the open file belongs to, from /api/series
  seriesPath: null,      // the file that series was detected for
  seriesRunning: false,
  poleFamily: null,      // Laue family the pole list was last initialised for
};

/* ------------------------------------------------------------------ */
/* helpers                                                            */
/* ------------------------------------------------------------------ */
async function api(path, options) {
  const response = await fetch(path, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || response.statusText);
  return payload;
}
const postJSON = (path, body) =>
  api(path, { method: "POST", headers: { "Content-Type": "application/json" },
              body: JSON.stringify(body) });

// A failed image request may carry a JSON error, or nothing readable at all if
// the server died mid-request; either way the caller needs a sentence.
async function errorMessage(response) {
  const payload = await response.json().catch(() => ({}));
  return payload.error || `the server answered ${response.status} ${response.statusText}`;
}

function debounce(fn, ms) {
  let timer;
  return (...args) => { clearTimeout(timer); timer = setTimeout(() => fn(...args), ms); };
}

function setStatus(text, kind) {
  const el = $("status");
  el.textContent = text;
  el.className = "status" + (kind ? " " + kind : "");
  state.status = { text, kind: kind || "" };
}

/* ------------------------------------------------------------------ */
/* the job tracker in the header                                      */
/* ------------------------------------------------------------------ */
/* Everything slow happens on the server, so the browser only knows that a
 * request is outstanding.  Each one registers here while it runs; the header
 * lists them over an indeterminate bar, and the card it belongs to shows a
 * spinner.  A PTM run or a series render owns the bar instead, with the real
 * fraction. */
const jobs = new Map();
let jobSerial = 0;

function beginJob(label, busyId) {
  const id = ++jobSerial;
  jobs.set(id, label);
  if (busyId && $(busyId)) $(busyId).hidden = false;
  renderJobs();
  return id;
}

function endJob(id, busyId) {
  jobs.delete(id);
  if (busyId && $(busyId)) $(busyId).hidden = true;
  renderJobs();
}

function renderJobs() {
  if (state.running || state.seriesRunning) return;
  const bar = $("header-progress");
  if (jobs.size) {
    bar.hidden = false;
    bar.removeAttribute("value");
    const labels = [...new Set(jobs.values())];
    const text = labels.join(", ") + "...";
    $("status").textContent = text;
    $("status").className = "status busy";
    return;
  }
  bar.hidden = true;
  bar.value = 0;
  // An error that arrived while a job ran must not be wiped by its completion.
  if ($("status").className.includes("error")) return;
  $("status").textContent = state.status.text;
  $("status").className = "status" + (state.status.kind ? " " + state.status.kind : "");
}

/* Fetch an image into an <img> with its card marked busy.  Responses that
 * arrive after a newer request for the same image are dropped, so a slow
 * old frame never overwrites the current one. */
const imageSerial = {};
async function loadImage(imgId, url, busyId, label) {
  const serial = (imageSerial[imgId] || 0) + 1;
  imageSerial[imgId] = serial;
  const job = beginJob(label, busyId);
  try {
    const response = await fetch(url);
    if (!response.ok) throw new Error(await errorMessage(response));
    const blob = await response.blob();
    if (imageSerial[imgId] !== serial) return null;
    const img = $(imgId);
    const old = img.src;
    img.src = URL.createObjectURL(blob);
    if (old && old.startsWith("blob:")) URL.revokeObjectURL(old);
    return response;
  } finally {
    endJob(job, busyId);
  }
}

// A URLSearchParams as a plain object, which is what the series job takes.
function paramsObject(params) {
  const out = {};
  for (const [key, value] of params) if (key !== "gen") out[key] = value;
  return out;
}

function hexToRgb(hex) {
  return [1, 3, 5].map((i) => Math.round(parseInt(hex.slice(i, i + 2), 16) / 2.55) / 100);
}
function rgbToHex(rgb) {
  return "#" + rgb.map((c) => Math.round(c * 255).toString(16).padStart(2, "0")).join("");
}

/* ------------------------------------------------------------------ */
/* gathering the form state                                           */
/* ------------------------------------------------------------------ */
function analysisParams() {
  const structures = [...document.querySelectorAll("#structures input:checked")]
    .map((el) => el.value);
  return {
    path: $("path").value.trim(),
    structures: structures.length ? structures : undefined,
    rmsd_cutoff: parseFloat($("rmsd").value) || 0,
    frame_index: parseInt($("frame-index").value, 10) || 0,
    columns: columnMapping(),
    slab: null,
  };
}

/* The slice as an analysis restriction: PTM then runs on those atoms alone
 * (plus a margin so the neighbours at the faces are right). */
function sliceSlab() {
  if (!sliceActive()) return null;
  const axis = $("slice-axis").value.trim();
  const distance = parseFloat($("slice-distance").value);
  if (!Number.isFinite(distance)) return null;
  return { axis, distance, width: parseFloat($("slice-width").value) || 0 };
}

function colourParams() {
  const axes = {};
  for (const name of ["rd", "td", "nd", "ed"]) {
    const value = $("axis-" + name).value.trim();
    if (value) axes[name] = value;
  }
  const only = [...document.querySelectorAll("#color-only input:checked")].map((el) => el.value);
  const all = document.querySelectorAll("#color-only input").length;
  return {
    direction: $("direction").value.trim() || "z",
    axes,
    color_only: only.length && only.length < all ? only : null,
    other_color: hexToRgb($("other-color").value),
    rotations: rotationParams(),
  };
}

function uiOptions() {
  const sliced = sliceActive();
  const tripod = paramsObject(tripodParams(new URLSearchParams()));
  return {
    poles: $("poles").value.split(",").map((p) => p.trim()).filter(Boolean),
    pole_mode: $("pole-mode").value,
    pole_structure: $("pole-structure").value || null,
    c_over_a: parseFloat($("c-over-a").value) || null,
    render_size: viewSize(),
    hide_other: $("hide-other").checked,
    slice_axis: sliced ? $("slice-axis").value.trim() : null,
    slice_distance: sliced ? parseFloat($("slice-distance").value) : null,
    slice_width: sliced ? parseFloat($("slice-width").value) || null : null,
    tripod: Boolean(tripod.tripod),
    tripod_axes: tripod.tripod_axes ? tripod.tripod_axes.split(";") : null,
    fill_radius: $("fill-on").checked ? parseFloat($("fill-radius").value) || 6 : null,
    fill_min_neighbours: parseInt($("fill-min").value, 10) || 3,
    export_directions: exportDirections(),
  };
}

// Semicolons, not commas: a direction may itself be a vector such as 1,1,0.
function exportDirections() {
  return $("export-directions").value.split(";").map((d) => d.trim()).filter(Boolean);
}

/* ------------------------------------------------------------------ */
/* analysis + status polling                                          */
/* ------------------------------------------------------------------ */
async function runAnalysis(full, slab) {
  let analysis;
  if (full || !state.analysed) {
    analysis = analysisParams();
    analysis.slab = slab === undefined ? null : slab;
  } else {
    analysis = state.analysed;
  }
  if (!analysis.path) { setStatus("choose a configuration file first", "error"); return; }
  try {
    const outcome = await postJSON("/api/analyse", { ...analysis, ...colourParams() });
    if (!outcome.accepted) { setStatus(outcome.reason, "error"); return; }
    if (outcome.recoloured) { await refreshStatus(); return; }
    state.running = true;
    setStatus("running polyhedral template matching…", "busy");
    pollUntilDone();
  } catch (error) {
    setStatus(error.message, "error");
  }
}

function pollUntilDone() {
  const timer = setInterval(async () => {
    try {
      const status = await api("/api/status");
      if (status.state === "running") {
        showProgress(status);
        return;
      }
      clearInterval(timer);
      state.running = false;
      showProgress(null);
      if (status.state === "error") {
        setStatus("analysis failed: " + status.error, "error");
        renderJobs();
        return;
      }
      applyStatus(status);
    } catch (error) {
      clearInterval(timer);
      state.running = false;
      showProgress(null);
      setStatus(error.message, "error");
    }
  }, 500);
}

/* OVITO says nothing while it works, so the server interpolates the bar inside
 * each stage from a throughput it calibrates on earlier runs.  That makes the
 * number an estimate, and the wording says so rather than implying a
 * measurement. */
function showProgress(status) {
  const bar = $("analysis-progress");
  const header = $("header-progress");
  if (!status) {
    bar.hidden = true; bar.value = 0;
    if (!state.seriesRunning) { header.hidden = true; header.value = 0; }
    renderJobs();
    return;
  }
  const fraction = typeof status.progress === "number" ? status.progress : 0;
  bar.hidden = false;
  bar.value = fraction;
  header.hidden = false;
  header.value = fraction;
  const percent = Math.round(100 * fraction);
  const left = status.stage_remaining > 1
    ? `, about ${Math.ceil(status.stage_remaining)} s left in this step`
    : "";
  setStatus(
    `${status.stage}… roughly ${percent}%${left} (${status.elapsed.toFixed(0)} s so far)`,
    "busy");
}

async function refreshStatus() {
  applyStatus(await api("/api/status"));
}

function applyStatus(status) {
  if (!status.result) return;
  state.analysed = {
    path: status.result.path,
    structures: status.result.structures,
    rmsd_cutoff: analysisParams().rmsd_cutoff,
    frame_index: analysisParams().frame_index,
    columns: columnMapping(),
    slab: status.result.slab || null,
  };
  state.selectionCount = status.selection ? status.selection.count : null;
  updateSummary(status.result);
  updateSelectionCount();
  $("cmd-button").disabled = false;
  const changed = status.generation !== state.generation;
  state.generation = status.generation;
  const slab = status.result.slab;
  const full = status.result.full_n_atoms;
  const sliceNote = slab
    ? ` (slice along ${slab.axis}` + (full ? ` of ${full.toLocaleString()}` : "") + ")"
    : "";
  setStatus(`${status.result.n_atoms.toLocaleString()} atoms${sliceNote} · IPF ` +
            status.result.direction_label, "");
  if (changed) {
    updateColorMapLink();
    updateSliceBounds().then(refreshAll);
    updateSeriesFromPath();
  }
  syncAnalyseSlice();
}

function updateSummary(result) {
  $("summary").hidden = false;
  $("summary-head").textContent =
    `${result.n_atoms.toLocaleString()} atoms · IPF ${result.direction_label} = ` +
    `[${result.direction.map((c) => c.toFixed(3)).join(" ")}]`;
  const swatches = { fcc: "#4c9f70", hcp: "#c94f4f", bcc: "#4f74c9", other: "#8a8a8a" };
  $("counts").replaceChildren(...Object.entries(result.counts).map(([name, count]) => {
    const chip = document.createElement("span");
    chip.className = "chip";
    const dot = document.createElement("span");
    dot.className = "dot";
    dot.style.background = swatches[name] || "#b58f4a";
    const percent = (100 * count / result.n_atoms).toFixed(1);
    chip.append(dot, `${name} ${count.toLocaleString()} (${percent}%)`);
    return chip;
  }));
  if (result.cell) {
    const lengths = [0, 1, 2].map((i) =>
      Math.hypot(result.cell[0][i], result.cell[1][i], result.cell[2][i]).toFixed(2));
    $("cell").textContent = `cell ${lengths.join(" × ")} Å`;
  }
  $("view-label").textContent = "· " + result.direction_label;
  const dominant = [...result.colorable]
    .sort((a, b) => (result.counts[b] || 0) - (result.counts[a] || 0))[0];
  state.dominant = dominant;
  populateStructureSelects(result.colorable, dominant);
  populateTypeOptions(result.type_names);
  initialisePoles(dominant);
}

/* ------------------------------------------------------------------ */
/* pole families                                                      */
/* ------------------------------------------------------------------ */
function laueOf(structure) {
  const entry = (state.meta.structures || []).find((s) => s.name === structure);
  return entry ? entry.laue : null;
}

/* The pole list starts from the family of the structure that dominates the
 * result, so an aluminium cell does not open with hexagonal poles. */
function initialisePoles(dominant) {
  const family = laueOf(dominant);
  if (!family || family === state.poleFamily) return;
  const presets = (state.meta.defaults.pole_presets || {})[family] || [];
  state.poleFamily = family;
  if (presets.length) $("poles").value = presets.slice(0, 3).join(",");
  fillPolePresets(family);
}

function fillPolePresets(family) {
  const select = $("pole-preset");
  const presets = (state.meta.defaults.pole_presets || {})[family] || [];
  select.replaceChildren(new Option("add a family...", ""));
  for (const pole of presets) select.add(new Option(pole, pole));
}

function addPolePreset() {
  const select = $("pole-preset");
  const pole = select.value;
  select.value = "";
  if (!pole) return;
  const poles = $("poles").value.split(",").map((p) => p.trim()).filter(Boolean);
  if (!poles.includes(pole)) poles.push(pole);
  $("poles").value = poles.join(",");
  debouncedFigures();
}

/* ------------------------------------------------------------------ */
/* 3D view                                                            */
/* ------------------------------------------------------------------ */
let renderInFlight = false;
let renderQueued = false;

function viewSize() {
  const box = $("view-box");
  const width = Math.max(480, Math.min(1600, Math.round(box.clientWidth || 900)));
  return [width, Math.round(width * 0.72)];
}

function viewQuery(extra) {
  const [w, h] = viewSize();
  const params = new URLSearchParams({
    az: state.camera.az.toFixed(1),
    el: state.camera.el.toFixed(1),
    zoom: state.camera.zoom.toFixed(3),
    w, h,
    hide_other: $("hide-other").checked ? 1 : 0,
    highlight: $("highlight-mode").value,
    gen: state.generation,
  });
  sliceParams(params);
  addFillParams(params);
  tripodParams(params);
  for (const [key, value] of Object.entries(extra || {})) params.set(key, value);
  return params;
}

/* ------------------------------------------------------------------ */
/* the slice                                                          */
/* ------------------------------------------------------------------ */
/* One slice serves the 3D view, the pole figures, the IPF density and the
 * IPF map.  The slider places the plane as a fraction of the atoms' extent
 * along the normal; the number box says where that is in angstroms, and
 * either can be edited.  The extent comes from the server once per normal. */
function sliceActive() {
  return $("slice-on").checked && Boolean($("slice-axis").value.trim());
}

function sliceParams(params) {
  if (!sliceActive()) return params;
  const axis = $("slice-axis").value.trim();
  params.set("slice_axis", axis);
  const distance = parseFloat($("slice-distance").value);
  if (Number.isFinite(distance) && state.slice.axis === axis && state.slice.low !== null) {
    params.set("slice_distance", distance.toFixed(3));
  } else {
    params.set("slice_frac", ($("slice-frac").value / 100).toFixed(3));
  }
  params.set("slice_width", $("slice-width").value || 0);
  return params;
}

function figuresSliced() {
  return sliceActive() && $("slice-figures").checked;
}

async function updateSliceBounds() {
  if (state.generation < 0 || !sliceActive()) return;
  const axis = $("slice-axis").value.trim();
  try {
    const bounds = await api("/api/slicebounds?axis=" + encodeURIComponent(axis) +
                             "&gen=" + state.generation);
    state.slice = { axis, low: bounds.min, high: bounds.max };
    const box = $("slice-distance");
    box.min = bounds.min;
    box.max = bounds.max;
    // A number typed before the bounds were known keeps its meaning; the
    // slider follows it.  Otherwise the slider's position is what counts.
    const typed = parseFloat(box.value);
    if (box.dataset.typed === "1" && Number.isFinite(typed)) syncSliceSlider();
    else syncSliceNumber();
  } catch (error) {
    state.slice = { axis: null, low: null, high: null };
  }
  syncAnalyseSlice();
}

function syncSliceNumber() {
  const { low, high } = state.slice;
  if (low === null) return;
  const fraction = $("slice-frac").value / 100;
  $("slice-distance").value = (low + fraction * (high - low)).toFixed(2);
  $("slice-distance").dataset.typed = "0";
}

function syncSliceSlider() {
  const { low, high } = state.slice;
  const distance = parseFloat($("slice-distance").value);
  if (low === null || !Number.isFinite(distance) || high <= low) return;
  const fraction = Math.max(0, Math.min(1, (distance - low) / (high - low)));
  $("slice-frac").value = Math.round(100 * fraction);
}

function syncAnalyseSlice() {
  $("analyse-slice").disabled = state.generation < 0 || !sliceActive() ||
    !Number.isFinite(parseFloat($("slice-distance").value));
}

function refreshSliced() {
  refreshView();
  if ($("slice-figures").checked) {
    refreshFigures();
    // The IPF map is drawn on request; once it has been, it follows the slice.
    if (state.flatDrawn) refreshFlatMap();
  }
  syncAnalyseSlice();
}

/* ------------------------------------------------------------------ */
/* the triad                                                          */
/* ------------------------------------------------------------------ */
function tripodParams(params) {
  if (!$("tripod").checked) return params;
  params.set("tripod", 1);
  const mode = $("tripod-mode").value;
  if (mode === "cell") {
    params.set("tripod_axes", "x;y;z");
  } else if (mode === "custom") {
    const axes = [], labels = [];
    for (const i of [1, 2, 3]) {
      const axis = $("tripod-axis-" + i).value.trim();
      if (!axis) continue;
      axes.push(axis);
      labels.push($("tripod-label-" + i).value.trim());
    }
    if (axes.length) params.set("tripod_axes", axes.join(";"));
    if (labels.some(Boolean)) params.set("tripod_labels", labels.join(";"));
  }
  params.set("tripod_size", ($("tripod-size").value / 100).toFixed(2));
  params.set("tripod_x", ($("tripod-x").value / 100).toFixed(2));
  params.set("tripod_y", ($("tripod-y").value / 100).toFixed(2));
  return params;
}

function bindTripod() {
  $("tripod-advanced-toggle").addEventListener("click", () => {
    $("tripod-advanced").hidden = !$("tripod-advanced").hidden;
  });
  $("tripod-mode").addEventListener("change", () => {
    $("tripod-custom").hidden = $("tripod-mode").value !== "custom";
    debouncedView();
  });
  for (const id of ["tripod-size", "tripod-x", "tripod-y"]) {
    $(id).addEventListener("input", () => {
      $(id + "-value").textContent = ($(id).value / 100).toFixed(2);
      debouncedView();
    });
  }
  for (const i of [1, 2, 3]) {
    $("tripod-axis-" + i).addEventListener("change", debouncedView);
    $("tripod-label-" + i).addEventListener("change", debouncedView);
  }
}

/* ------------------------------------------------------------------ */
/* system rotations                                                   */
/* ------------------------------------------------------------------ */
function addRotation(axis, angle) {
  const row = document.createElement("div");
  row.className = "rotation row wrap";
  const axisInput = document.createElement("input");
  axisInput.type = "text";
  axisInput.value = axis || "z";
  axisInput.size = 5;
  axisInput.spellcheck = false;
  axisInput.dataset.field = "axis";
  axisInput.title = "rotation axis: x, y, z, rd, td, nd or a vector such as 1,1,0";
  const angleInput = document.createElement("input");
  angleInput.type = "number";
  angleInput.step = "any";
  angleInput.value = angle || 0;
  angleInput.dataset.field = "angle";
  angleInput.title = "angle in degrees, right-handed about the axis";
  const unit = document.createElement("span");
  unit.className = "muted";
  unit.textContent = "\u00b0";
  const kill = document.createElement("button");
  kill.className = "small ghost kill";
  kill.textContent = "\u00d7";
  kill.title = "remove this rotation";
  kill.addEventListener("click", () => { row.remove(); debouncedColour(); });
  row.append("about ", axisInput, " by ", angleInput, unit, kill);
  $("rotations").append(row);
  attachVectorBoxes(axisInput);
  axisInput.addEventListener("change", debouncedColour);
  angleInput.addEventListener("change", debouncedColour);
  return row;
}

function rotationParams() {
  const out = [];
  for (const row of document.querySelectorAll("#rotations .rotation")) {
    const axis = row.querySelector('[data-field="axis"]').value.trim();
    const angle = parseFloat(row.querySelector('[data-field="angle"]').value);
    if (axis && Number.isFinite(angle) && angle !== 0) out.push([axis, angle]);
  }
  return out;
}

function setRotations(rotations) {
  $("rotations").replaceChildren();
  for (const [axis, angle] of rotations || []) addRotation(axis, angle);
}

/* ------------------------------------------------------------------ */
/* vectors by component                                               */
/* ------------------------------------------------------------------ */
/* Every direction box takes a name or a vector as text; this adds a small
 * "xyz" button beside it that opens three component boxes, for people who
 * think in components rather than in comma-separated strings. */
function attachVectorBoxes(input) {
  if (input.dataset.vectorBound) return;
  input.dataset.vectorBound = "1";
  const toggle = document.createElement("button");
  toggle.type = "button";
  toggle.className = "small ghost vec-toggle";
  toggle.textContent = "xyz";
  toggle.title = "enter the direction as x, y, z components";
  const boxes = document.createElement("span");
  boxes.className = "vec-boxes";
  boxes.hidden = true;
  const fields = ["x", "y", "z"].map((name) => {
    const field = document.createElement("input");
    field.type = "number";
    field.step = "any";
    field.placeholder = name;
    field.title = name + " component";
    return field;
  });
  boxes.append(...fields);
  const push = () => {
    const values = fields.map((f) => f.value.trim());
    if (!values.every(Boolean)) return;
    input.value = values.map((v) => String(parseFloat(v))).join(",");
    input.dispatchEvent(new Event("change", { bubbles: true }));
  };
  for (const field of fields) field.addEventListener("change", push);
  toggle.addEventListener("click", (event) => {
    event.preventDefault();
    boxes.hidden = !boxes.hidden;
    if (boxes.hidden) return;
    const parts = input.value.split(",").map(Number);
    if (parts.length === 3 && parts.every(Number.isFinite)) {
      fields.forEach((field, i) => { field.value = parts[i]; });
    }
    fields[0].focus();
  });
  input.after(toggle, boxes);
}

// Boundary filling changes every image the server draws, so it travels with
// each request instead of being part of the cached analysis settings.
function addFillParams(params) {
  if ($("fill-on") && $("fill-on").checked) {
    params.set("fill_radius", $("fill-radius").value || 6);
    params.set("fill_min_neighbours", $("fill-min").value || 3);
  }
  return params;
}

/* ------------------------------------------------------------------ */
/* colour scales for the density plots                                 */
/* ------------------------------------------------------------------ */
const CUSTOM_OPTION = "__upload__";

function fillColormapSelects() {
  const names = state.meta.colormaps || ["viridis"];
  for (const [id, preferred] of [["pole-cmap", "viridis"], ["density-cmap", "magma"]]) {
    const select = $(id);
    select.replaceChildren();
    for (const name of names) {
      const option = document.createElement("option");
      option.value = name;
      option.textContent = name;
      if (name === preferred) option.selected = true;
      select.append(option);
    }
    if (state.customColormap) {
      const option = document.createElement("option");
      option.value = "custom";
      option.textContent = state.customColormap;
      select.append(option);
    }
    const upload = document.createElement("option");
    upload.value = CUSTOM_OPTION;
    upload.textContent = "upload your own...";
    select.append(upload);
    select.dataset.previous = select.value;
  }
}

/* The select doubles as the upload trigger, so the dialog has to put the
 * previous choice back if nothing is uploaded after all. */
function bindColormapSelect(id, onChange) {
  const select = $(id);
  select.addEventListener("change", () => {
    if (select.value !== CUSTOM_OPTION) {
      select.dataset.previous = select.value;
      onChange();
      return;
    }
    select.value = select.dataset.previous || "viridis";
    openColormapDialog(id);
  });
}

function openColormapDialog(selectId) {
  state.colormapTarget = selectId;
  $("colormap-status").textContent = "";
  $("colormap-status").className = "muted note";
  $("colormap-dialog").showModal();
}

async function uploadColormap(file) {
  const note = $("colormap-status");
  note.textContent = "reading " + file.name + "...";
  note.className = "muted note";
  try {
    const buffer = new Uint8Array(await file.arrayBuffer());
    let binary = "";
    for (let i = 0; i < buffer.length; i += 1) binary += String.fromCharCode(buffer[i]);
    const outcome = await postJSON("/api/colormap/upload",
                                   { name: file.name, data: btoa(binary) });
    state.customColormap = outcome.name;
    const target = state.colormapTarget;
    fillColormapSelects();
    if (target) {
      $(target).value = "custom";
      $(target).dataset.previous = "custom";
    }
    note.textContent = `${outcome.name}: ${outcome.entries} colours, now in both menus`;
    $("colormap-dialog").close();
    refreshFigures();
  } catch (error) {
    note.textContent = error.message;
    note.className = "error-text note";
  }
}

function flatMapQuery(extra) {
  const params = new URLSearchParams({
    view: $("flat-view").value.trim() || "z",
    slab_width: $("flat-slab").value || 10,
    pixel_size: $("flat-pixel").value || 0.5,
    boundary_angle: $("flat-angle").value || 0,
    gen: state.generation,
  });
  if ($("flat-raw").checked) params.set("raw", 1);
  if ($("figures-from-selection").checked && state.selectionCount !== null) {
    params.set("selection", 1);
  }
  if (figuresSliced()) sliceParams(params);
  addFillParams(params);
  for (const [key, value] of Object.entries(extra || {})) params.set(key, value);
  return params;
}

async function refreshFlatMap() {
  if (state.generation < 0) return;
  const info = $("flat-info");
  info.textContent = "drawing...";
  const sliced = figuresSliced();
  state.flatDrawn = true;
  $("flat-view").disabled = sliced;
  $("flat-slab").disabled = sliced && parseFloat($("slice-width").value) > 0;
  try {
    const response = await loadImage("flatmap", "/api/figure/flatmap?" + flatMapQuery(),
                                     "flatmap-busy", "drawing the IPF map");
    if (!response) return;
    const grains = response.headers.get("X-Grain-Count");
    const size = response.headers.get("X-Map-Size");
    const center = response.headers.get("X-Slab-Center");
    $("download-flat").href = "/api/figure/flatmap?" + flatMapQuery({ download: 1 });
    $("download-flat-svg").href =
      "/api/figure/flatmap?" + flatMapQuery({ download: 1, format: "svg" });
    info.textContent = `${grains} grains, ${size} px` +
      (sliced ? `, section of the slice along ${$("slice-axis").value.trim()} centred at ${center} \u00c5`
              : "");
  } catch (error) {
    info.textContent = error.message;
  }
}

async function refreshView() {
  if (state.generation < 0) return;
  if (renderInFlight) { renderQueued = true; return; }
  renderInFlight = true;
  try {
    const response = await loadImage("view", "/api/render?" + viewQuery(),
                                     "view-busy", "rendering the 3D view");
    if (!response) return;
    $("view").classList.add("live");
    $("view-placeholder").hidden = true;
    $("download-view").href = "/api/render?" + viewQuery({ download: 1 });
    // A decoration OVITO would not draw is a note, not a failure: the view is
    // there, so say what is missing from it and leave it on screen.
    showViewError(response.headers.get("X-Render-Warning") || null, true);
  } catch (error) {
    setStatus("the 3D view failed", "error");
    showViewError(error.message);
  } finally {
    renderInFlight = false;
    if (renderQueued) { renderQueued = false; refreshView(); }
  }
}

/* The 3D view is drawn by OVITO on the server and arrives as a PNG, so a
 * failure here is a server-side renderer problem and the browser has nothing
 * to show for it.  Saying so in the empty frame beats a broken image. */
function showViewError(message, warningOnly) {
  const box = $("view-error");
  if (!message) { box.hidden = true; return; }
  box.replaceChildren();
  const title = document.createElement("strong");
  const detail = document.createElement("p");
  detail.textContent = message;
  const hint = document.createElement("p");
  hint.className = "muted";
  if (warningOnly) {
    // The image is fine, only a decoration is missing, so it stays on screen.
    title.textContent = "The view was drawn without one of its decorations.";
    hint.textContent = "Everything else in the view is as it should be. " +
      "Switching the triad off clears this message.";
  } else {
    // A stale image beside the explanation reads as if the view still worked.
    $("view").classList.remove("live");
    title.textContent = "The 3D view could not be drawn.";
    hint.textContent = "Run  ptmipf-ui --check  in the same environment to see what is " +
      "missing. The plots, the IPF map and the exports do not need a " +
      "renderer and still work.";
  }
  box.append(title, detail, hint);
  box.hidden = false;
  $("view-placeholder").hidden = true;
}

function bindViewer() {
  const img = $("view");
  let dragging = false, moved = false, lastX = 0, lastY = 0;
  img.addEventListener("pointerdown", (event) => {
    dragging = true; moved = false;
    lastX = event.clientX; lastY = event.clientY;
    img.setPointerCapture(event.pointerId);
    img.classList.add("dragging");
  });
  img.addEventListener("pointermove", (event) => {
    if (!dragging) return;
    const dx = event.clientX - lastX, dy = event.clientY - lastY;
    if (Math.abs(dx) + Math.abs(dy) > 2) moved = true;
    lastX = event.clientX; lastY = event.clientY;
    state.camera.az -= dx * 0.4;
    state.camera.el = Math.max(-89, Math.min(89, state.camera.el + dy * 0.4));
    refreshView();
  });
  img.addEventListener("pointerup", (event) => {
    dragging = false;
    img.classList.remove("dragging");
    if (!moved) pickAtom(event);
  });
  img.addEventListener("wheel", (event) => {
    event.preventDefault();
    state.camera.zoom = Math.max(0.1, Math.min(40,
      state.camera.zoom * (event.deltaY < 0 ? 1.15 : 1 / 1.15)));
    refreshView();
  }, { passive: false });
  $("reset-view").addEventListener("click", () => {
    state.camera = { az: -125, el: 20, zoom: 1.0 };
    refreshView();
  });
}

async function pickAtom(event) {
  const img = $("view");
  const rect = img.getBoundingClientRect();
  const [w, h] = viewSize();
  const body = {
    x: (event.clientX - rect.left) * (img.naturalWidth / rect.width),
    y: (event.clientY - rect.top) * (img.naturalHeight / rect.height),
    w: img.naturalWidth || w, h: img.naturalHeight || h,
    az: state.camera.az, el: state.camera.el, zoom: state.camera.zoom,
    hide_other: $("hide-other").checked,
    highlight: $("highlight-mode").value,
  };
  Object.assign(body, paramsObject(sliceParams(new URLSearchParams())));
  try {
    const outcome = await postJSON("/api/pick", body);
    showAtomInfo(outcome.atom);
  } catch (error) {
    setStatus(error.message, "error");
  }
}

function showAtomInfo(atom) {
  state.atomInfo = atom;
  const box = $("atom-info");
  if (!atom) { box.hidden = true; return; }
  box.hidden = false;
  box.replaceChildren();
  const head = document.createElement("div");
  const dot = document.createElement("span");
  dot.className = "dot chip";
  dot.style.background = rgbToHex(atom.color);
  dot.style.padding = "0 0.45rem";
  head.append(`atom ${atom.index} · ${atom.structure}` +
    (atom.type ? ` · type ${atom.type}` : "") + ` · rmsd ${atom.rmsd} `, dot);
  const pos = document.createElement("div");
  pos.className = "muted";
  pos.textContent = `at (${atom.position.join(", ")})`;
  const row = document.createElement("span");
  row.className = "row";
  const use = document.createElement("button");
  use.className = "small";
  use.textContent = "use as misorientation reference";
  use.addEventListener("click", () => useAsReference(atom.index));
  const close = document.createElement("button");
  close.className = "small ghost";
  close.textContent = "×";
  close.addEventListener("click", () => showAtomInfo(null));
  row.append(use, close);
  box.append(head, pos, row);
}

function useAsReference(index) {
  let target = document.querySelector('.criterion[data-kind="misorientation"] input[data-field="reference"]');
  if (!target) {
    addCriterion("misorientation");
    target = document.querySelector('.criterion[data-kind="misorientation"] input[data-field="reference"]');
  }
  if (target) target.value = String(index);
}

/* ------------------------------------------------------------------ */
/* figures                                                            */
/* ------------------------------------------------------------------ */
function figureQuery(extra) {
  const params = new URLSearchParams({ gen: state.generation });
  if ($("figures-from-selection").checked && state.selectionCount !== null) {
    params.set("selection", 1);
  }
  if (figuresSliced()) sliceParams(params);
  addFillParams(params);
  for (const [key, value] of Object.entries(extra || {})) if (value) params.set(key, value);
  return params;
}

function legendQuery() {
  return figureQuery({ structure: $("legend-structure").value });
}

function densityQuery() {
  return figureQuery({
    structure: $("pole-structure").value,
    cmap: $("density-cmap").value,
    smoothing: $("density-smoothing").value,
  });
}

function polesQuery() {
  const poles = $("poles").value.split(",").map((p) => p.trim()).filter(Boolean);
  if (!poles.length) return null;
  return figureQuery({
    poles: poles.join(","),
    mode: $("pole-mode").value,
    structure: $("pole-structure").value,
    c_over_a: $("c-over-a").value,
    cmap: $("pole-cmap").value,
    smoothing: $("pole-smoothing").value,
    up: $("pole-up").value.trim(),
    right: $("pole-right").value.trim(),
  });
}

function setDownloads(prefix, url) {
  $(prefix).href = url + "&download=1";
  $(prefix + "-svg").href = url + "&download=1&format=svg";
}

function refreshFigures() {
  if (state.generation < 0) return;
  const legend = "/api/figure/legend?" + legendQuery();
  setDownloads("download-legend", legend);
  loadImage("legend", legend, "legend-busy", "drawing the colour key")
    .catch((error) => setStatus("colour key: " + error.message, "error"));

  const density = "/api/figure/ipfdensity?" + densityQuery();
  setDownloads("download-density", density);
  loadImage("density", density, "density-busy", "drawing the IPF density")
    .catch((error) => setStatus("IPF density: " + error.message, "error"));

  const poles = polesQuery();
  if (poles) {
    const url = "/api/figure/poles?" + poles;
    setDownloads("download-poles", url);
    loadImage("polefig", url, "polefig-busy", "drawing the pole figures")
      .catch((error) => setStatus("pole figures: " + error.message, "error"));
  }
}

function refreshAll() { refreshView(); refreshFigures(); }

/* ------------------------------------------------------------------ */
/* selection builder                                                  */
/* ------------------------------------------------------------------ */
function structureOptions() {
  const names = state.analysed ? state.analysed.structures : ["fcc", "hcp", "bcc"];
  return names;
}

function makeField(labelText, input) {
  const label = document.createElement("label");
  label.className = "inline";
  label.append(labelText, input);
  return label;
}
function textInput(field, value, size) {
  const input = document.createElement("input");
  input.type = "text";
  input.dataset.field = field;
  input.value = value ?? "";
  if (size) input.size = size;
  input.spellcheck = false;
  return input;
}
function numberInput(field, value, step) {
  const input = document.createElement("input");
  input.type = "number";
  input.dataset.field = field;
  if (value !== null && value !== undefined) input.value = value;
  input.step = step || "any";
  return input;
}
function structureSelect(field) {
  const select = document.createElement("select");
  select.dataset.field = field;
  for (const name of structureOptions()) select.add(new Option(name, name));
  if (state.dominant) select.value = state.dominant;
  return select;
}

const CRITERION_LABELS = {
  structure: "structure", type: "particle type", rmsd: "RMSD range",
  region: "spatial slab", ipf: "orientation near direction",
  misorientation: "misorientation from reference",
};

function addCriterion(kind) {
  const box = document.createElement("div");
  box.className = "criterion";
  box.dataset.kind = kind;

  const head = document.createElement("div");
  head.className = "head";
  const invert = document.createElement("label");
  invert.className = "inline";
  const invertBox = document.createElement("input");
  invertBox.type = "checkbox";
  invertBox.dataset.field = "invert";
  invert.append(invertBox, "not");
  const kill = document.createElement("button");
  kill.className = "small ghost kill";
  kill.textContent = "×";
  kill.addEventListener("click", () => box.remove());
  head.append(CRITERION_LABELS[kind], invert, kill);
  box.append(head);

  const row = document.createElement("span");
  row.className = "row wrap";
  if (kind === "structure") {
    for (const name of structureOptions()) {
      const label = document.createElement("label");
      label.className = "inline";
      const check = document.createElement("input");
      check.type = "checkbox";
      check.dataset.structure = name;
      if (name === (state.dominant || "hcp")) check.checked = true;
      label.append(check, name);
      row.append(label);
    }
  } else if (kind === "type") {
    const select = document.createElement("select");
    select.dataset.field = "types";
    select.multiple = true;
    select.size = 3;
    for (const name of Object.values(state.typeNames || {})) select.add(new Option(name, name));
    row.append(makeField("types", select));
  } else if (kind === "rmsd") {
    row.append(makeField("min", numberInput("min", null, 0.01)),
               makeField("max", numberInput("max", 0.05, 0.01)));
  } else if (kind === "region") {
    row.append(makeField("axis", textInput("axis", "z", 4)),
               makeField("min", numberInput("min", null)),
               makeField("max", numberInput("max", null)));
  } else if (kind === "ipf") {
    row.append(makeField("crystal", textInput("crystal", "0001", 6)),
               makeField("sample", textInput("sample", "nd", 4)),
               makeField("tol°", numberInput("tolerance", 15, 1)),
               makeField("in", structureSelect("structure")));
  } else if (kind === "misorientation") {
    const ref = textInput("reference", "", 12);
    ref.placeholder = "atom # or x,y,z,w";
    ref.title = "an atom index (click an atom in the view) or a quaternion x,y,z,w";
    row.append(makeField("ref", ref),
               makeField("tol°", numberInput("tolerance", 5, 1)),
               makeField("in", structureSelect("structure")));
  }
  box.append(row);
  $("criteria").append(box);
}

function serializeCriteria() {
  const criteria = [];
  for (const box of document.querySelectorAll("#criteria .criterion")) {
    const kind = box.dataset.kind;
    const criterion = { kind };
    const invert = box.querySelector('input[data-field="invert"]');
    if (invert && invert.checked) criterion.invert = true;
    const value = (field) => {
      const el = box.querySelector(`[data-field="${field}"]`);
      return el && el.value !== "" ? el.value : null;
    };
    if (kind === "structure") {
      criterion.structures = [...box.querySelectorAll("input[data-structure]:checked")]
        .map((el) => el.dataset.structure);
      if (!criterion.structures.length) continue;
    } else if (kind === "type") {
      const select = box.querySelector('select[data-field="types"]');
      criterion.types = [...select.selectedOptions].map((option) => option.value);
      if (!criterion.types.length) continue;
    } else if (kind === "rmsd" || kind === "region") {
      if (kind === "region") criterion.axis = value("axis") || "z";
      const min = value("min"), max = value("max");
      if (min !== null) criterion.min = parseFloat(min);
      if (max !== null) criterion.max = parseFloat(max);
      if (min === null && max === null) continue;
    } else if (kind === "ipf") {
      criterion.crystal = value("crystal");
      criterion.sample = value("sample");
      criterion.tolerance = parseFloat(value("tolerance") || 10);
      criterion.structure = value("structure");
      if (!criterion.crystal || !criterion.sample) continue;
    } else if (kind === "misorientation") {
      const ref = value("reference");
      if (ref === null) continue;
      const parts = ref.split(",").map((t) => parseFloat(t)).filter((n) => !Number.isNaN(n));
      criterion.reference = parts.length === 4
        ? { quaternion: parts } : { atom: parseInt(ref, 10) };
      criterion.tolerance = parseFloat(value("tolerance") || 5);
      criterion.structure = value("structure");
    }
    criteria.push(criterion);
  }
  return criteria;
}

async function applySelection() {
  try {
    const outcome = await postJSON("/api/selection", {
      criteria: serializeCriteria(), mode: $("combine-mode").value,
    });
    state.selectionCount = outcome.count;
    updateSelectionCount();
    await refreshStatus();
  } catch (error) {
    setStatus(error.message, "error");
  }
}

function updateSelectionCount() {
  const count = state.selectionCount;
  $("selection-count").textContent =
    count === null ? "no selection active" : `${count.toLocaleString()} atoms selected`;
  for (const button of document.querySelectorAll('[data-export][data-selection="1"]')) {
    button.disabled = count === null;
  }
}

/* ------------------------------------------------------------------ */
/* file browser                                                       */
/* ------------------------------------------------------------------ */
async function openBrowser(path) {
  const dialog = $("browser-dialog");
  try {
    const listing = await api("/api/browse?path=" + encodeURIComponent(path || ""));
    $("browser-path").textContent = "/" + (listing.path || "");
    const items = [];
    if (!listing.at_root) {
      const li = document.createElement("li");
      li.textContent = "↑ ..";
      li.addEventListener("click", () =>
        openBrowser(listing.path.split("/").slice(0, -1).join("/")));
      items.push(li);
    }
    for (const entry of listing.entries) {
      const li = document.createElement("li");
      const name = document.createElement("span");
      name.textContent = (entry.dir ? "\u{1F4C1} " : "") + entry.name;
      li.append(name);
      if (!entry.dir) {
        const size = document.createElement("span");
        size.className = "size";
        size.textContent = entry.size > 1048576
          ? (entry.size / 1048576).toFixed(1) + " MB" : (entry.size / 1024).toFixed(0) + " kB";
        li.append(size);
      }
      const target = listing.path ? listing.path + "/" + entry.name : entry.name;
      li.addEventListener("click", () => {
        if (entry.dir) openBrowser(target);
        else { $("path").value = target; dialog.close(); detectSeries(target); }
      });
      items.push(li);
    }
    $("browser-list").replaceChildren(...items);
    if (!dialog.open) dialog.showModal();
  } catch (error) {
    setStatus(error.message, "error");
  }
}

/* ------------------------------------------------------------------ */
/* wiring                                                             */
/* ------------------------------------------------------------------ */
function populateStructureChecks(structures) {
  const makeCheck = (container, s, checked) => {
    const label = document.createElement("label");
    label.title = s.description;
    const check = document.createElement("input");
    check.type = "checkbox";
    check.value = s.name;
    check.checked = checked;
    label.append(check, s.name);
    container.append(label);
    return check;
  };
  for (const s of structures) {
    const idCheck = makeCheck($("structures"), s, s.default);
    const onlyCheck = makeCheck($("color-only"), s, s.default && s.colorable);
    onlyCheck.disabled = !s.colorable;
    idCheck.addEventListener("change", () => {
      onlyCheck.disabled = !idCheck.checked || !s.colorable;
      if (!idCheck.checked) onlyCheck.checked = false;
    });
    onlyCheck.addEventListener("change", debouncedColour);
  }
}

function populateStructureSelects(colorable, dominant) {
  for (const select of [$("legend-structure"), $("pole-structure")]) {
    const previous = select.value;
    select.replaceChildren(...colorable.map((name) => new Option(name, name)));
    select.value = colorable.includes(previous) ? previous : dominant;
  }
}

function populateTypeOptions(typeNames) {
  state.typeNames = typeNames;
}

const debouncedColour = debounce(() => runAnalysis(false), 400);
const debouncedView = debounce(refreshView, 250);
const debouncedFigures = debounce(refreshFigures, 400);

/* ------------------------------------------------------------------ */
/* orientations that are already in the file                          */
/* ------------------------------------------------------------------ */
/* A configuration another OVITO session has run PTM on already carries the
 * quaternions.  Mapping the columns skips PTM entirely, which is both faster
 * and the only way to colour a file whose orientations came from settings this
 * server does not know about.
 *
 * Nothing in a file states its quaternion convention, and the two common ones
 * differ by a transpose, which turns an IPF map into a plausible looking but
 * wrong one.  So the convention is asked for, never guessed, and the guess the
 * server offers is only for the column names. */
function columnControls() {
  return state.columns ? state.columns.controls : null;
}

function columnMapping() {
  const controls = columnControls();
  if (!$("use-columns").checked || !controls) return null;
  const quaternion = controls.single.value === "__four__"
    ? controls.quad.map((select) => select.value)
    : [controls.single.value];
  if (quaternion.some((name) => !name)) return null;
  const mapping = {
    quaternion,
    order: controls.order.value,
    conjugate: controls.conjugate.checked,
  };
  if (controls.structureType.value) {
    mapping.structure_type = controls.structureType.value;
  } else {
    mapping.structure = controls.structure.value;
  }
  if (controls.rmsd.value) mapping.rmsd = controls.rmsd.value;
  return mapping;
}

function labelledSelect(parent, text, options, selected, title) {
  const label = document.createElement("label");
  label.className = "inline";
  label.textContent = text;
  if (title) label.title = title;
  const select = document.createElement("select");
  for (const [value, name] of options) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = name;
    if (value === selected) option.selected = true;
    select.append(option);
  }
  label.append(select);
  parent.append(label);
  return select;
}

async function readColumns() {
  const path = $("path").value.trim();
  const note = $("columns-status");
  if (!path) { note.textContent = "choose a configuration file first"; return; }
  note.textContent = "reading...";
  note.className = "muted";
  try {
    const info = await api("/api/columns?path=" + encodeURIComponent(path) +
                           "&frame=" + (parseInt($("frame-index").value, 10) || 0));
    buildColumnMap(info);
    note.textContent = `${info.columns.length} columns, ` +
                       `${info.n_atoms.toLocaleString()} atoms`;
  } catch (error) {
    note.textContent = error.message;
    note.className = "error-text";
    $("column-map").hidden = true;
    state.columns = null;
  }
}

function buildColumnMap(info) {
  const box = $("column-map");
  box.replaceChildren();
  const scalars = info.columns.filter((c) => c.components === 1);
  const quads = info.columns.filter((c) => c.components === 4);
  const guess = info.guess || {};
  const guessed = guess.quaternion || [];

  const quaternionOptions = [
    ...quads.map((c) => [c.name, `${c.name} (4 components)`]),
    ["__four__", "four separate columns..."],
  ];
  const single = labelledSelect(box, "quaternion ", quaternionOptions,
    guessed.length === 1 ? guessed[0] : "__four__",
    "One four-component column, or four scalar columns");

  const quad = [];
  const quadBox = document.createElement("div");
  quadBox.className = "quad";
  const scalarOptions = [["", "..."], ...scalars.map((c) => [c.name, c.name])];
  for (let i = 0; i < 4; i += 1) {
    quad.push(labelledSelect(quadBox, "", scalarOptions, guessed[i] || ""));
  }
  box.append(quadBox);

  const order = labelledSelect(box, "component order ",
    [["xyzw", "x, y, z, w  (OVITO)"], ["wxyz", "w, x, y, z"]], "xyzw",
    "OVITO writes the scalar part last; most other tools write it first");

  const conjugateLabel = document.createElement("label");
  conjugateLabel.className = "inline";
  conjugateLabel.title = "Tick this if the file stores the sample to crystal rotation";
  const conjugate = document.createElement("input");
  conjugate.type = "checkbox";
  conjugateLabel.append(conjugate, " invert the sense (sample to crystal)");
  box.append(conjugateLabel);

  const structureType = labelledSelect(box, "structure column ",
    [["", "none, one phase for all"], ...scalars.map((c) => [c.name, c.name])],
    guess.structure_type || "",
    "A structure type column is read with OVITO's own PTM codes");
  const structure = labelledSelect(box, "phase ",
    (state.meta.structures || []).filter((s) => s.colorable).map((s) => [s.name, s.name]),
    state.dominant || "fcc");
  const rmsd = labelledSelect(box, "RMSD column ",
    [["", "none"], ...scalars.map((c) => [c.name, c.name])], guess.rmsd || "");

  const warning = document.createElement("p");
  warning.className = "warn";
  warning.textContent = "Check the result against a grain you know. The wrong component " +
    "order or sense gives a map that looks right and is not.";
  box.append(warning);

  state.columns = {
    info,
    controls: { single, quad, quadBox, order, conjugate, structureType, structure, rmsd },
  };
  const sync = () => {
    quadBox.hidden = single.value !== "__four__";
    structure.parentElement.hidden = Boolean(structureType.value);
  };
  single.addEventListener("change", sync);
  structureType.addEventListener("change", sync);
  sync();
  box.hidden = false;
}

/* ------------------------------------------------------------------ */
/* trajectory series                                                  */
/* ------------------------------------------------------------------ */
/* Once a frame has been analysed the server is asked which frames the file
 * belongs to: numbered siblings (dump_0, dump_20, dump_100) or the frames
 * inside it.  Stepping re-runs the analysis with the current settings; the
 * batch card renders any range of them with the same settings. */
const SERIES_OUTPUTS = [
  ["view", "3D IPF map", ["png", "gif", "mp4"]],
  ["ipfmap", "IPF map", ["png", "svg", "gif", "mp4"]],
  ["poles", "Pole figures", ["png", "svg", "gif", "mp4"]],
  ["density", "IPF density", ["png", "svg", "gif", "mp4"]],
  ["legend", "Colour key", ["png", "svg"]],
];

function updateSeriesFromPath() {
  const path = state.analysed ? state.analysed.path : $("path").value.trim();
  if (!path || path === state.seriesPath) return;
  state.seriesPath = path;
  detectSeries(path);
}

async function detectSeries(path) {
  try {
    state.series = await api("/api/series?path=" + encodeURIComponent(path));
  } catch (error) {
    state.series = null;
  }
  renderSeries();
}

function seriesCurrentIndex() {
  const series = state.series;
  if (!series || !series.items.length) return -1;
  const path = state.analysed ? state.analysed.path : $("path").value.trim();
  const frame = parseInt($("frame-index").value, 10) || 0;
  const index = series.items.findIndex((item) =>
    series.kind === "frames" ? item.frame_index === frame
                             : path.endsWith(item.path));
  return index < 0 ? series.current : index;
}

function renderSeries() {
  const series = state.series;
  const nav = $("series-nav");
  const card = $("series-card");
  const items = series ? series.items : [];
  if (!items.length) {
    nav.hidden = true;
    card.classList.add("inactive");
    $("series-card-info").textContent = state.generation < 0
      ? "analyse a frame first"
      : "this file is not part of a numbered series and holds a single frame";
    $("series-run").disabled = true;
    return;
  }
  nav.hidden = false;
  card.classList.remove("inactive");
  const current = seriesCurrentIndex();
  const options = () => items.map((item, i) => new Option(item.label, String(i)));
  $("series-select").replaceChildren(...options());
  $("series-select").value = String(current);
  $("series-prev").disabled = current <= 0;
  $("series-next").disabled = current >= items.length - 1;
  $("series-info").textContent = series.kind === "files"
    ? `${items.length} files, ${items[0].label} to ${items[items.length - 1].label}`
    : `${items.length} frames in this file`;
  const keepStart = $("series-start").value, keepStop = $("series-stop").value;
  $("series-start").replaceChildren(...options());
  $("series-stop").replaceChildren(...options());
  $("series-start").value = keepStart && Number(keepStart) < items.length ? keepStart : "0";
  $("series-stop").value = keepStop && Number(keepStop) < items.length
    ? keepStop : String(items.length - 1);
  $("series-card-info").textContent = `${items.length} frames`;
  if (!$("series-outdir").value) $("series-outdir").placeholder = series.stem + "_series";
  $("series-run").disabled = state.seriesRunning;
}

function gotoSeriesItem(index) {
  const series = state.series;
  if (!series || index < 0 || index >= series.items.length) return;
  const item = series.items[index];
  if (series.kind === "files") $("path").value = item.path;
  $("frame-index").value = item.frame_index;
  runAnalysis(true, state.analysed ? state.analysed.slab : null);
}

function buildSeriesOutputs() {
  const box = $("series-outputs");
  for (const [kind, label, formats] of SERIES_OUTPUTS) {
    const row = document.createElement("div");
    row.className = "row wrap output-row";
    const name = document.createElement("span");
    name.className = "output-name";
    name.textContent = label;
    row.append(name);
    for (const ext of formats) {
      const check = document.createElement("label");
      check.className = "inline";
      const input = document.createElement("input");
      input.type = "checkbox";
      input.value = `${kind}:${ext}`;
      input.checked = (kind === "view" && ext === "png") || (kind === "ipfmap" && ext === "gif");
      check.append(input, ext === "gif" || ext === "mp4" ? `${ext} movie` : `${ext} stills`);
      row.append(check);
    }
    box.append(row);
  }
}

async function startSeries() {
  const series = state.series;
  if (!series || !series.items.length) return;
  const outputs = [...document.querySelectorAll("#series-outputs input:checked")]
    .map((el) => el.value);
  const body = {
    path: state.analysed ? state.analysed.path : $("path").value.trim(),
    start: parseInt($("series-start").value, 10) || 0,
    stop: parseInt($("series-stop").value, 10) || 0,
    step: parseInt($("series-step").value, 10) || 1,
    outputs,
    seconds_per_frame: parseFloat($("series-seconds").value) || 0.5,
    label: $("series-stamp").checked,
    out_dir: $("series-outdir").value.trim() || null,
    view_query: paramsObject(viewQuery()),
    ipfmap_query: paramsObject(flatMapQuery()),
    poles_query: paramsObject(polesQuery() || figureQuery()),
    density_query: paramsObject(densityQuery()),
    legend_query: paramsObject(legendQuery()),
  };
  try {
    await postJSON("/api/series/render", body);
    $("series-files").replaceChildren();
    $("series-zip").hidden = true;
    pollSeries();
  } catch (error) {
    $("series-status").textContent = error.message;
    $("series-status").className = "error-text note";
  }
}

function pollSeries() {
  if (state.seriesTimer) return;
  state.seriesRunning = true;
  $("series-run").disabled = true;
  $("series-cancel").disabled = false;
  state.seriesTimer = setInterval(async () => {
    let status;
    try {
      status = await api("/api/series/status");
    } catch (error) {
      status = { state: "error", error: error.message, files: [] };
    }
    showSeries(status);
    if (status.state !== "running") {
      clearInterval(state.seriesTimer);
      state.seriesTimer = null;
      state.seriesRunning = false;
      $("series-run").disabled = false;
      $("series-cancel").disabled = true;
      $("header-progress").hidden = true;
      renderJobs();
    }
  }, 1000);
}

function showSeries(status) {
  const bar = $("series-progress");
  const note = $("series-status");
  const running = status.state === "running";
  bar.hidden = !running && !status.files.length;
  bar.value = status.progress || 0;
  const header = $("header-progress");
  if (running) {
    header.hidden = false;
    header.value = status.progress || 0;
    const left = status.seconds_per_item
      ? `, about ${Math.ceil(status.seconds_per_item * (status.n_items - status.item))} s left`
      : "";
    const text = `series ${status.item + 1}/${status.n_items} ${status.label}: ${status.stage}${left}`;
    $("status").textContent = text;
    $("status").className = "status busy";
    note.textContent = text;
    note.className = "muted note";
  } else if (status.state === "error") {
    note.textContent = "the series failed: " + status.error;
    note.className = "error-text note";
  } else if (status.state === "cancelled") {
    note.textContent = `cancelled after ${status.item} of ${status.n_items} frames`;
    note.className = "muted note";
  } else if (status.state === "done") {
    note.textContent = `${status.files.length} files written to ${status.out_dir} ` +
      `in ${status.elapsed} s`;
    note.className = "muted note";
  }
  const links = status.files.map((name) => {
    const link = document.createElement("a");
    link.href = "/api/series/file?path=" + encodeURIComponent(name) + "&download=1";
    link.textContent = name;
    link.download = name;
    link.className = "small button";
    return link;
  });
  $("series-files").replaceChildren(...links);
  $("series-zip").hidden = !status.files.length;
  $("series-zip").href = "/api/series/zip";
}

function bindSeries() {
  buildSeriesOutputs();
  $("series-prev").addEventListener("click", () => gotoSeriesItem(seriesCurrentIndex() - 1));
  $("series-next").addEventListener("click", () => gotoSeriesItem(seriesCurrentIndex() + 1));
  $("series-select").addEventListener("change", () =>
    gotoSeriesItem(parseInt($("series-select").value, 10)));
  $("series-run").addEventListener("click", startSeries);
  $("series-cancel").addEventListener("click", () => postJSON("/api/series/cancel", {}));
  renderSeries();
}

/* ------------------------------------------------------------------ */
/* the command line: copy, save, and read one back                     */
/* ------------------------------------------------------------------ */
function updateColorMapLink() {
  const link = $("download-colormap");
  // Without a result the endpoint has nothing to build a palette from, so the
  // link stays inert rather than downloading an error.
  if (state.generation < 0) { link.removeAttribute("href"); return; }
  const params = new URLSearchParams({
    directions: exportDirections().join(";"),
    gen: state.generation,
  });
  link.href = "/api/colormap?" + params;
}

function bindCommandDialog() {
  const dialog = $("cmd-dialog");

  $("cmd-button").addEventListener("click", async () => {
    try {
      const outcome = await postJSON("/api/command", uiOptions());
      state.command = outcome;
      $("cmd-text").textContent = outcome.command;
      $("cmd-note").textContent = outcome.note || "";
      $("cmd-note").hidden = !outcome.note;
      $("cmd-import-status").textContent = "";
      dialog.showModal();
    } catch (error) { setStatus(error.message, "error"); }
  });

  $("cmd-copy").addEventListener("click", () => copyCommand($("cmd-text").textContent));
  // Backslash continuations are a POSIX shell convention.  PowerShell and the
  // Windows command prompt break the command at the first line end instead, so
  // the pasted command has to be one line there.
  $("cmd-copy-line").addEventListener("click", () =>
    copyCommand((state.command && state.command.one_line) || $("cmd-text").textContent));

  $("cmd-save").addEventListener("click", () => {
    const text = $("cmd-text").textContent + "\n";
    const url = URL.createObjectURL(new Blob([text], { type: "text/plain" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = "ptmipf-command.txt";
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });

  $("cmd-load").addEventListener("click", () => $("cmd-file").click());
  $("cmd-file").addEventListener("change", async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    $("cmd-input").value = await file.text();
    event.target.value = "";
    $("cmd-import").open = true;
  });

  $("cmd-apply").addEventListener("click", async () => {
    const note = $("cmd-import-status");
    try {
      const settings = await postJSON("/api/command/parse",
                                      { command: $("cmd-input").value });
      applySettings(settings);
      note.textContent = "applied";
      note.className = "muted";
      dialog.close();
      if ($("cmd-apply-run").checked && $("path").value.trim()) {
        runAnalysis(true, state.pendingSlab || null);
      }
    } catch (error) {
      note.textContent = error.message;
      note.className = "error-text";
    }
  });

  $("cmd-close").addEventListener("click", () => dialog.close());
}

/* The renderer lives on the server, so ask it once whether there is one rather
 * than letting the first orbit come back as a broken image. */
async function checkEnvironment() {
  try {
    const report = await api("/api/diagnostics");
    if (report.ok) return;
    showViewError(report.checks.filter((check) => !check.ok)
      .map((check) => `${check.name}: ${check.detail || "not available"}`).join("\n"));
  } catch (error) {
    // The panel is a courtesy; a failed probe must never block the interface.
  }
}

async function copyCommand(text) {
  const note = $("cmd-import-status");
  try {
    await navigator.clipboard.writeText(text);
    note.textContent = "copied";
    note.className = "muted";
  } catch (error) {
    // A page served over plain http has no clipboard API in some browsers;
    // selecting the text is the honest fallback.
    const range = document.createRange();
    range.selectNodeContents($("cmd-text"));
    const selection = getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    note.textContent = "the browser blocked the clipboard; the command is selected, " +
      "press Ctrl-C";
    note.className = "error-text";
  }
}

/* Set the whole form from a parsed command line, so a session can be resumed
 * from the command that produced it. */
function applySettings(settings) {
  const analysis = settings.analysis || {};
  const colour = settings.colour || {};
  const ui = settings.ui || {};

  if (analysis.path) $("path").value = analysis.path;
  if (analysis.rmsd_cutoff !== null && analysis.rmsd_cutoff !== undefined) {
    $("rmsd").value = analysis.rmsd_cutoff;
  }
  $("frame-index").value = analysis.frame_index || 0;
  setChecks("#structures input", analysis.structures);
  setChecks("#color-only input",
            (colour.color_only && colour.color_only.length) ? colour.color_only : null);

  for (const name of ["rd", "td", "nd", "ed"]) {
    $("axis-" + name).value = (colour.axes && colour.axes[name]) || "";
  }
  if (colour.direction) $("direction").value = colour.direction;
  setRotations(colour.rotations || []);
  if (colour.other_color) {
    const parts = String(colour.other_color).split(",").map(Number);
    if (parts.length === 3 && parts.every((v) => Number.isFinite(v))) {
      $("other-color").value = rgbToHex(parts);
    }
  }

  $("fill-on").checked = Boolean(ui.fill_radius);
  if (ui.fill_radius) $("fill-radius").value = ui.fill_radius;
  if (ui.fill_min_neighbours) $("fill-min").value = ui.fill_min_neighbours;

  if (ui.poles && ui.poles.length) $("poles").value = ui.poles.join(",");
  if (ui.pole_mode) $("pole-mode").value = ui.pole_mode;
  if (ui.c_over_a) $("c-over-a").value = ui.c_over_a;

  $("hide-other").checked = Boolean(ui.hide_other);
  $("slice-on").checked = Boolean(ui.slice_axis);
  if (ui.slice_axis) $("slice-axis").value = ui.slice_axis;
  if (ui.slice_distance !== null && ui.slice_distance !== undefined) {
    $("slice-distance").value = ui.slice_distance;
    $("slice-distance").dataset.typed = "1";
  }
  $("slice-width").value = ui.slice_width || 0;
  state.pendingSlab = ui.ptm_slice && ui.slice_axis
    ? { axis: ui.slice_axis, distance: ui.slice_distance || 0, width: ui.slice_width || 0 }
    : null;
  $("tripod").checked = Boolean(ui.tripod);
  if (ui.tripod_axes && ui.tripod_axes.length) {
    const axes = ui.tripod_axes.map((a) => a.split("=")[0].trim().toLowerCase());
    if (axes.join(";") === "x;y;z") {
      $("tripod-mode").value = "cell";
    } else if (axes.join(";") !== "rd;td;nd") {
      $("tripod-mode").value = "custom";
      ui.tripod_axes.forEach((spec, i) => {
        const [axis, label] = spec.split("=");
        if (i < 3) {
          $("tripod-axis-" + (i + 1)).value = axis.trim();
          $("tripod-label-" + (i + 1)).value = (label || "").trim();
        }
      });
    }
    $("tripod-custom").hidden = $("tripod-mode").value !== "custom";
  }
  if (ui.export_directions && ui.export_directions.length) {
    $("export-directions").value = ui.export_directions.join("; ");
    updateColorMapLink();
  }
  if (ui.view) {
    $("flat-view").value = ui.view;
    const axis = { x: [0, 0], y: [90, 0], z: [-90, 89.9] }[ui.view.trim().toLowerCase()];
    if (axis) { state.camera.az = axis[0]; state.camera.el = axis[1]; }
  }
}

function setChecks(selector, wanted) {
  if (!wanted) return;
  const names = new Set(wanted);
  for (const input of document.querySelectorAll(selector)) {
    input.checked = names.has(input.value);
  }
}

async function init() {
  state.meta = await api("/api/meta");
  $("version").textContent = "v" + state.meta.version;
  populateStructureChecks(state.meta.structures);
  $("poles").value = state.meta.defaults.poles.join(",");
  $("c-over-a").value = state.meta.defaults.c_over_a.toFixed(3);
  $("other-color").value = rgbToHex(state.meta.defaults.other_color);
  if (!state.meta.selection_available) {
    $("selection-unavailable").hidden = false;
    $("selection-ui").hidden = true;
  }
  if (state.meta.initial_path) $("path").value = state.meta.initial_path;
  // The examples page hands a freshly built structure over this way.
  const wanted = new URLSearchParams(location.search).get("path");
  if (wanted) $("path").value = wanted;

  $("analyse").addEventListener("click", () => runAnalysis(true, null));
  $("analyse-slice").addEventListener("click", () => {
    const slab = sliceSlab();
    if (slab) runAnalysis(true, slab);
  });
  $("read-columns").addEventListener("click", readColumns);
  $("use-columns").addEventListener("change", () => {
    const on = $("use-columns").checked;
    if (on) {
      $("columns-section").open = true;
      if (!state.columns) readColumns();
    }
    // PTM's own settings do nothing once the orientations come from the file.
    $("rmsd").disabled = on;
  });
  $("browse").addEventListener("click", () => openBrowser(""));
  $("browser-close").addEventListener("click", () => $("browser-dialog").close());
  $("path").addEventListener("keydown", (e) => { if (e.key === "Enter") runAnalysis(true); });

  for (const id of ["direction", "axis-rd", "axis-td", "axis-nd", "axis-ed"]) {
    $(id).addEventListener("change", debouncedColour);
  }
  $("other-color").addEventListener("change", debouncedColour);
  for (const button of document.querySelectorAll("#direction-buttons button")) {
    button.addEventListener("click", () => {
      $("direction").value = button.dataset.dir;
      debouncedColour();
    });
  }

  bindViewer();
  bindTripod();
  for (const input of document.querySelectorAll("input[data-vector]")) attachVectorBoxes(input);
  $("add-rotation").addEventListener("click", () => addRotation("z", 0));
  $("clear-rotations").addEventListener("click", () => { setRotations([]); debouncedColour(); });
  $("hide-other").addEventListener("change", debouncedView);
  const debouncedSliced = debounce(refreshSliced, 300);
  const sliceNormalChanged = () => updateSliceBounds().then(debouncedSliced);
  $("slice-on").addEventListener("change", sliceNormalChanged);
  $("slice-axis").addEventListener("change", sliceNormalChanged);
  $("slice-frac").addEventListener("input", () => { syncSliceNumber(); debouncedSliced(); });
  $("slice-distance").addEventListener("input", () => {
    $("slice-distance").dataset.typed = "1";
    syncSliceSlider();
    debouncedSliced();
  });
  $("slice-width").addEventListener("input", debouncedSliced);
  $("slice-figures").addEventListener("change", () => {
    refreshFigures();
    if (state.flatDrawn) refreshFlatMap();
  });
  $("tripod").addEventListener("change", debouncedView);
  for (const id of ["fill-on", "fill-radius", "fill-min"]) {
    $(id).addEventListener("change", () => {
      refreshView();
      refreshFigures();
    });
  }
  $("flat-draw").addEventListener("click", refreshFlatMap);
  // Look straight down an axis, which turns a slab into an EBSD-style section.
  const axisViews = { "view-x": [0, 0], "view-y": [90, 0], "view-z": [-90, 89.9] };
  for (const [id, [az, el]] of Object.entries(axisViews)) {
    $(id).addEventListener("click", () => {
      state.camera.az = az;
      state.camera.el = el;
      refreshView();
    });
  }
  $("highlight-mode").addEventListener("change", debouncedView);

  for (const id of ["poles", "c-over-a", "pole-mode", "pole-structure",
                    "pole-smoothing", "density-smoothing", "pole-up", "pole-right"]) {
    $(id).addEventListener("change", debouncedFigures);
  }
  fillPolePresets("hexagonal");
  $("pole-preset").addEventListener("change", addPolePreset);
  bindSeries();
  fillColormapSelects();
  bindColormapSelect("pole-cmap", debouncedFigures);
  bindColormapSelect("density-cmap", debouncedFigures);
  $("colormap-file").addEventListener("change", (event) => {
    const file = event.target.files[0];
    event.target.value = "";
    if (file) uploadColormap(file);
  });
  $("colormap-close").addEventListener("click", () => $("colormap-dialog").close());
  $("legend-structure").addEventListener("change", debouncedFigures);
  $("figures-from-selection").addEventListener("change", refreshAll);

  $("add-criterion").addEventListener("click", () => addCriterion($("criterion-kind").value));
  $("apply-selection").addEventListener("click", applySelection);
  $("clear-selection").addEventListener("click", async () => {
    $("criteria").replaceChildren();
    await applySelection();
  });

  for (const button of document.querySelectorAll("[data-export]")) {
    button.addEventListener("click", () => {
      const params = new URLSearchParams({
        format: button.dataset.export,
        selection: button.dataset.selection,
        directions: exportDirections().join(";"),
      });
      window.location = "/api/export?" + params;
    });
  }
  $("export-directions").addEventListener("change", updateColorMapLink);
  updateColorMapLink();

  bindCommandDialog();

  $("theme-toggle").addEventListener("click", () => {
    const root = document.documentElement;
    const dark = root.dataset.theme
      ? root.dataset.theme === "dark"
      : matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    localStorage.setItem("ptmipf-theme", root.dataset.theme);
  });
  const savedTheme = localStorage.getItem("ptmipf-theme");
  if (savedTheme) document.documentElement.dataset.theme = savedTheme;

  updateSelectionCount();
  syncAnalyseSlice();
  checkEnvironment();
  // Pick up an analysis, or a series render, that survived a page reload.
  const status = await api("/api/status");
  if (status.state === "running") { state.running = true; pollUntilDone(); }
  else if (status.result) applyStatus(status);
  const series = await api("/api/series/status").catch(() => null);
  if (series && series.state === "running") pollSeries();
  else if (series && series.files && series.files.length) showSeries(series);
}

init().catch((error) => setStatus(error.message, "error"));
