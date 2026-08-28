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
  return {
    poles: $("poles").value.split(",").map((p) => p.trim()).filter(Boolean),
    pole_mode: $("pole-mode").value,
    pole_structure: $("pole-structure").value || null,
    c_over_a: parseFloat($("c-over-a").value) || null,
    render_size: viewSize(),
    hide_other: $("hide-other").checked,
    slice_axis: $("slice-on").checked ? $("slice-axis").value.trim() : null,
  };
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
        setStatus(`${status.stage}… ${status.elapsed.toFixed(0)} s`, "busy");
        return;
      }
      clearInterval(timer);
      state.running = false;
      if (status.state === "error") { setStatus("analysis failed: " + status.error, "error"); return; }
      applyStatus(status);
    } catch (error) {
      clearInterval(timer);
      state.running = false;
      setStatus(error.message, "error");
    }
  }, 500);
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
  }
  for (const [key, value] of Object.entries(extra || {})) params.set(key, value);
  return params;
}

async function refreshView() {
  if (state.generation < 0) return;
  if (renderInFlight) { renderQueued = true; return; }
  renderInFlight = true;
  try {
    const response = await fetch("/api/render?" + viewQuery());
    if (!response.ok) throw new Error((await response.json()).error || "render failed");
    const blob = await response.blob();
    const img = $("view");
    const old = img.src;
    img.src = URL.createObjectURL(blob);
    img.classList.add("live");
    $("view-placeholder").hidden = true;
    if (old.startsWith("blob:")) URL.revokeObjectURL(old);
    $("download-view").href = "/api/render?" + viewQuery({ download: 1 });
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    renderInFlight = false;
    if (renderQueued) { renderQueued = false; refreshView(); }
  }
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

  $("analyse").addEventListener("click", () => runAnalysis(true));
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
      window.location = "/api/export?format=" + button.dataset.export +
        "&selection=" + button.dataset.selection;
    });
  }

  $("cmd-button").addEventListener("click", async () => {
    try {
      const outcome = await postJSON("/api/command", uiOptions());
      $("cmd-text").textContent = outcome.command;
      $("cmd-dialog").showModal();
    } catch (error) { setStatus(error.message, "error"); }
  });
  $("cmd-copy").addEventListener("click", () =>
    navigator.clipboard.writeText($("cmd-text").textContent));
  $("cmd-close").addEventListener("click", () => $("cmd-dialog").close());

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
  // Pick up an analysis that survived a page reload.
  const status = await api("/api/status");
  if (status.state === "running") { state.running = true; pollUntilDone(); }
  else if (status.result) applyStatus(status);
}

init().catch((error) => setStatus(error.message, "error"));
