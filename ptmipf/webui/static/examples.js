/* The examples page: build a polycrystal, fetch its potential, write the run.
 *
 * Everything expensive happens on the server; this side only collects the
 * settings, shows what a build would cost before it is asked for, and renders
 * what came back.
 */
"use strict";

const $ = (id) => document.getElementById(id);
const state = { meta: null, building: false };

async function api(path, options) {
  const response = await fetch(path, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || `the server answered ${response.status}`);
  return payload;
}

function setStatus(text, kind) {
  const el = $("status");
  el.textContent = text;
  el.className = "status" + (kind ? " " + kind : "");
}

function settings() {
  return {
    box: parseFloat($("box").value),
    n_grains: parseInt($("grains").value, 10),
    strain: parseFloat($("strain").value),
    strain_rate: parseFloat($("strain-rate").value),
    temperature: parseFloat($("temperature").value),
    seed: parseInt($("seed").value, 10),
    builder: $("builder").value,
  };
}

/* The same arithmetic the server uses, so a card can quote a cost before
 * anything is built.  Both are estimates; the one after the build is the one
 * with the real atom count in it. */
const RATE_FOUR_CORES = 2.0e6;   // atom-steps a second, measured, see lammps.py
const RATE_ONE_CORE = 6.0e5;

function estimate(entry) {
  const s = settings();
  const atoms = Math.round(0.97 * entry.atoms_per_cell * Math.pow(s.box / entry.a0, 3));
  const steps = Math.round(s.strain / s.strain_rate / 0.002) + 2000;
  const atomSteps = atoms * steps;
  return {
    atoms,
    atomSteps,
    low: atomSteps / RATE_FOUR_CORES / 60,
    high: atomSteps / RATE_ONE_CORE / 60,
  };
}

function renderCatalogue() {
  const box = $("catalogue");
  box.replaceChildren();
  for (const entry of state.meta.examples) {
    const cost = estimate(entry);
    const card = document.createElement("div");
    card.className = "example";

    const title = document.createElement("h3");
    title.textContent = `${entry.element} polycrystal, ${entry.structure}`;
    const cite = document.createElement("p");
    cite.className = "muted note";
    const link = document.createElement("a");
    link.href = entry.url;
    link.target = "_blank";
    link.rel = "noreferrer";
    link.textContent = entry.citation;
    cite.append("Potential: ", link);

    const cost_line = document.createElement("p");
    cost_line.className = "note";
    cost_line.textContent =
      `about ${cost.atoms.toLocaleString()} atoms, ` +
      `${(cost.atomSteps / 1e6).toFixed(0)} million atom-steps: ` +
      `roughly ${cost.low.toFixed(0)} to ${cost.high.toFixed(0)} minutes to run`;

    const button = document.createElement("button");
    button.className = "primary";
    button.textContent = `Build ${entry.element}`;
    button.addEventListener("click", () => build(entry));

    card.append(title, cite, cost_line, button);
    box.append(card);
  }
}

async function build(entry) {
  if (state.building) return;
  state.building = true;
  setStatus(`building the ${entry.element} example, downloading its potential…`, "busy");
  for (const button of document.querySelectorAll(".example button")) button.disabled = true;
  try {
    const report = await api("/api/examples/build", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ element: entry.element, ...settings() }),
    });
    renderResult(report);
    setStatus(`built ${report.n_atoms.toLocaleString()} atoms in ${report.relative}`, "");
  } catch (error) {
    setStatus(error.message, "error");
    renderFailure(error.message);
  } finally {
    state.building = false;
    for (const button of document.querySelectorAll(".example button")) button.disabled = false;
  }
}

function renderFailure(message) {
  $("result-card").hidden = false;
  const box = $("result");
  box.replaceChildren();
  const line = document.createElement("p");
  line.className = "error-text";
  line.textContent = message;
  box.append(line);
}

function row(table, name, value) {
  const tr = document.createElement("tr");
  const key = document.createElement("th");
  key.textContent = name;
  const cell = document.createElement("td");
  cell.append(value);
  tr.append(key, cell);
  table.append(tr);
}

function renderResult(report) {
  $("result-card").hidden = false;
  const box = $("result");
  box.replaceChildren();

  const where = document.createElement("p");
  where.className = "note";
  where.append("Written to ");
  const code = document.createElement("code");
  code.textContent = report.directory;
  where.append(code);
  box.append(where);

  const table = document.createElement("table");
  table.className = "kv";
  row(table, "structure",
      `${report.n_atoms.toLocaleString()} atoms, ${report.n_grains} grains averaging ` +
      `${report.grain_size} Å across, in a ${report.box} Å cube`);
  row(table, "density",
      `${(100 * report.density).toFixed(1)} % of a perfect single crystal, ` +
      `closest pair ${report.min_separation} Å`);
  row(table, "builder", report.builder === "atomsk"
      ? "atomsk"
      : "the built in fallback, a few percent less dense at the boundaries than atomsk");
  const cite = document.createElement("a");
  cite.href = report.potential.url;
  cite.target = "_blank";
  cite.rel = "noreferrer";
  cite.textContent = report.potential.citation;
  row(table, "potential", cite);
  row(table, "run",
      `${report.run.atom_steps.toLocaleString()} atom-steps: roughly ` +
      `${report.run.minutes_four_cores.toFixed(0)} to ` +
      `${report.run.minutes_one_core.toFixed(0)} minutes`);
  box.append(table);

  const steps = document.createElement("ol");
  steps.className = "steps";
  const run = document.createElement("li");
  run.append("Run it:");
  run.append(pre(`cd ${report.directory}\nlmp -in in.compression`));
  const look = document.createElement("li");
  look.append("Then open the last frame here, or from the terminal:");
  look.append(pre(
    `ptmipf compression.dump --structures ${report.structure} --direction z \\\n` +
    `    --legend key.png --render map.png --hide-other`));
  steps.append(run, look);
  box.append(steps);

  const open = document.createElement("a");
  open.className = "button primary";
  open.href = "/?path=" + encodeURIComponent(report.relative_xyz);
  open.textContent = "Open the as-built structure in the analysis page";
  box.append(open);

  const files = document.createElement("p");
  files.className = "muted note";
  files.textContent = "Files: " + report.files.join(", ");
  box.append(files);
}

function pre(text) {
  const block = document.createElement("pre");
  block.textContent = text;
  return block;
}

async function init() {
  const saved = localStorage.getItem("ptmipf-theme");
  if (saved) document.documentElement.dataset.theme = saved;
  $("theme-toggle").addEventListener("click", () => {
    const root = document.documentElement;
    const dark = root.dataset.theme
      ? root.dataset.theme === "dark"
      : matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    localStorage.setItem("ptmipf-theme", root.dataset.theme);
  });

  try {
    state.meta = await api("/api/examples");
  } catch (error) {
    setStatus(error.message, "error");
    return;
  }
  if (!state.meta.atomsk) {
    const note = $("atomsk-note");
    note.textContent = state.meta.atomsk_help;
    note.hidden = false;
    $("builder").value = "voronoi";
  }
  renderCatalogue();
  for (const id of ["box", "grains", "strain", "strain-rate", "temperature", "seed"]) {
    $(id).addEventListener("input", renderCatalogue);
  }
  setStatus(`serving ${state.meta.root}`, "");
}

init().catch((error) => setStatus(error.message, "error"));
