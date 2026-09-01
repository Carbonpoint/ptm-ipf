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
  };
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
  };
}

function uiOptions() {
  const sliced = $("slice-on").checked && $("slice-axis").value.trim();
  return {
    poles: $("poles").value.split(",").map((p) => p.trim()).filter(Boolean),
    pole_mode: $("pole-mode").value,
    pole_structure: $("pole-structure").value || null,
    c_over_a: parseFloat($("c-over-a").value) || null,
    render_size: viewSize(),
    hide_other: $("hide-other").checked,
    slice_axis: sliced ? $("slice-axis").value.trim() : null,
    slice_width: sliced ? parseFloat($("slice-width").value) || null : null,
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
async function runAnalysis(full) {
  const analysis = full || !state.analysed ? analysisParams() : state.analysed;
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
      if (status.state === "error") { setStatus("analysis failed: " + status.error, "error"); return; }
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
  if (!status) { bar.hidden = true; bar.value = 0; return; }
  const fraction = typeof status.progress === "number" ? status.progress : 0;
  bar.hidden = false;
  bar.value = fraction;
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
  };
  state.selectionCount = status.selection ? status.selection.count : null;
  updateSummary(status.result);
  updateSelectionCount();
  $("cmd-button").disabled = false;
  if (status.generation !== state.generation) {
    state.generation = status.generation;
    updateColorMapLink();
    refreshAll();
  }
  setStatus(`${status.result.n_atoms.toLocaleString()} atoms · IPF ` +
            status.result.direction_label, "");
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
  if ($("slice-on").checked && $("slice-axis").value.trim()) {
    params.set("slice_axis", $("slice-axis").value.trim());
    params.set("slice_frac", ($("slice-frac").value / 100).toFixed(3));
    params.set("slice_width", $("slice-width").value || 0);
  }
  addFillParams(params);
  if ($("tripod") && $("tripod").checked) params.set("tripod", 1);
  for (const [key, value] of Object.entries(extra || {})) params.set(key, value);
  return params;
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
  addFillParams(params);
  for (const [key, value] of Object.entries(extra || {})) params.set(key, value);
  return params;
}

async function refreshFlatMap() {
  if (state.generation < 0) return;
  const info = $("flat-info");
  info.textContent = "drawing...";
  try {
    const response = await fetch("/api/figure/flatmap?" + flatMapQuery());
    if (!response.ok) throw new Error(await errorMessage(response));
    const grains = response.headers.get("X-Grain-Count");
    const size = response.headers.get("X-Map-Size");
    const blob = await response.blob();
    const img = $("flatmap");
    const old = img.src;
    img.src = URL.createObjectURL(blob);
    if (old && old.startsWith("blob:")) URL.revokeObjectURL(old);
    $("download-flat").href = "/api/figure/flatmap?" + flatMapQuery({ download: 1 });
    info.textContent = `${grains} grains, ${size} px`;
  } catch (error) {
    info.textContent = error.message;
  }
}

async function refreshView() {
  if (state.generation < 0) return;
  if (renderInFlight) { renderQueued = true; return; }
  renderInFlight = true;
  try {
    const response = await fetch("/api/render?" + viewQuery());
    if (!response.ok) throw new Error(await errorMessage(response));
    const blob = await response.blob();
    const img = $("view");
    const old = img.src;
    img.src = URL.createObjectURL(blob);
    img.classList.add("live");
    $("view-placeholder").hidden = true;
    if (old.startsWith("blob:")) URL.revokeObjectURL(old);
    $("download-view").href = "/api/render?" + viewQuery({ download: 1 });
    showViewError(null);
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
function showViewError(message) {
  const box = $("view-error");
  if (!message) { box.hidden = true; return; }
  // A stale image beside the explanation reads as if the view still worked.
  $("view").classList.remove("live");
  box.replaceChildren();
  const title = document.createElement("strong");
  title.textContent = "The 3D view could not be drawn.";
  const detail = document.createElement("p");
  detail.textContent = message;
  const hint = document.createElement("p");
  hint.className = "muted";
  hint.textContent = "Run  ptmipf-ui --check  in the same environment to see what is " +
    "missing. The plots, the flat orientation map and the exports do not need a " +
    "renderer and still work.";
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
  if ($("slice-on").checked && $("slice-axis").value.trim()) {
    body.slice_axis = $("slice-axis").value.trim();
    body.slice_frac = $("slice-frac").value / 100;
    body.slice_width = Number($("slice-width").value) || 0;
  }
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
  for (const [key, value] of Object.entries(extra || {})) if (value) params.set(key, value);
  return params;
}

function refreshFigures() {
  if (state.generation < 0) return;
  const legend = "/api/figure/legend?" + figureQuery({ structure: $("legend-structure").value });
  $("legend").src = legend;
  $("download-legend").href = legend + "&download=1";

  const density = "/api/figure/ipfdensity?" + figureQuery({ structure: $("pole-structure").value });
  $("density").src = density;
  $("download-density").href = density + "&download=1";

  const poles = $("poles").value.split(",").map((p) => p.trim()).filter(Boolean);
  if (poles.length) {
    const url = "/api/figure/poles?" + figureQuery({
      poles: poles.join(","),
      mode: $("pole-mode").value,
      structure: $("pole-structure").value,
      c_over_a: $("c-over-a").value,
    });
    $("polefig").src = url;
    $("download-poles").href = url + "&download=1";
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
        else { $("path").value = target; dialog.close(); }
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
      if ($("cmd-apply-run").checked && $("path").value.trim()) runAnalysis(true);
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
  if (ui.slice_width) $("slice-width").value = ui.slice_width;
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

  $("analyse").addEventListener("click", () => runAnalysis(true));
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
  $("hide-other").addEventListener("change", debouncedView);
  $("slice-on").addEventListener("change", debouncedView);
  $("slice-axis").addEventListener("change", debouncedView);
  $("slice-frac").addEventListener("input", debouncedView);
  $("slice-width").addEventListener("input", debouncedView);
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

  for (const id of ["poles", "c-over-a", "pole-mode", "pole-structure"]) {
    $(id).addEventListener("change", debouncedFigures);
  }
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
  checkEnvironment();
  // Pick up an analysis that survived a page reload.
  const status = await api("/api/status");
  if (status.state === "running") { state.running = true; pollUntilDone(); }
  else if (status.result) applyStatus(status);
}

init().catch((error) => setStatus(error.message, "error"));
