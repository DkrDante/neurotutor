// Guided data-collection runner. It walks a participant through the protocol in
// protocol.js, and for every block posts block_start / block_end markers (plus
// the self-report ratings) to the server's recorder — so the training labels
// are a by-product of running the session, never hand-annotated afterwards.

const { buildProtocol, totalSeconds } = window.NeuroProtocol;
const $ = (id) => document.getElementById(id);
function el(tag, props = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "text") node.textContent = v; else if (k === "class") node.className = v; else node.setAttribute(k, v);
  }
  for (const c of children) node.append(c);
  return node;
}
async function api(path, method = "GET", body) {
  const res = await fetch(path, {
    method, headers: body ? { "Content-Type": "application/json" } : undefined, body: body ? JSON.stringify(body) : undefined,
  });
  let data = null;
  try { data = await res.json(); } catch (_) { /* empty */ }
  if (!res.ok) throw new Error((data && data.detail) || `${res.status} ${res.statusText}`);
  return data;
}
const marker = (kind, data = {}) => api("/api/record/marker", "POST", { kind, data }).catch((e) => console.warn("marker failed", kind, e));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const fmt = (s) => `${Math.floor(Math.max(0, s) / 60)}:${String(Math.floor(Math.max(0, s) % 60)).padStart(2, "0")}`;

const run = { active: false, abort: false, skip: false, step: null };

// ------------------------------------------------------------------- setup
let hub = null;
function readOptions() {
  return {
    participant: $("participant").value.trim(), skill: parseFloat($("skill").value) || 1000,
    blockSeconds: parseFloat($("block-seconds").value) || 240, quick: $("quick").checked,
  };
}
function renderPreview() {
  const steps = buildProtocol(readOptions());
  $("protocol-preview").replaceChildren(...steps.map((s) => el("li", {}, [
    el("strong", { text: s.type === "task" ? `${s.condition.replace("_", " ")} ` : `${s.title} ` }),
    el("span", { text: `${fmt(s.seconds)}${s.band ? ` · puzzles rated ${s.band[0]}–${s.band[1]}` : ""}${s.dual ? " · + digit task" : ""}` }),
  ])));
  $("protocol-preview").append(el("li", { class: "mini-label", text: `Total ≈ ${Math.round(totalSeconds(steps) / 60)} min including ratings and rests.` }));
}
function renderSetupStatus() {
  const pills = [];
  const pill = (text, cls) => pills.push(el("span", { class: `setup-pill ${cls}`, text }));
  if (!hub || !hub.connected) {
    pill("No EEG source connected — open the live monitor and connect", "bad");
  } else {
    pill(hub.kind === "lsl" ? "Headset streaming" : "Simulator (dry run only — no real microvolts)", hub.kind === "lsl" ? "good" : "warn");
    const names = (hub.channel_names || []).join(", ");
    pill(`${hub.num_channels} channels${names ? ` (${names})` : ""} @ ${hub.sample_rate} Hz`, hub.num_channels > 0 ? "good" : "bad");
    if (hub.tutor_model_channels && hub.tutor_model_channels !== hub.num_channels) {
      pill(`Tutor model expects ${hub.tutor_model_channels} ch — recording is unaffected (puzzles run without predictions)`, "warn");
    }
    if (hub.kind === "replay") pill("A replay is connected — connect the headset (or simulator) to record a new session", "warn");
    const q = hub.quality || [];
    const bad = q.filter((c) => c.status === "flat" || c.status === "noisy").length;
    if (hub.kind === "lsl") pill(bad === 0 ? "Signal quality OK on all channels" : `${bad} channel(s) with poor contact — fix before starting`, bad === 0 ? "good" : "warn");
    if (hub.calibrating) pill(`Learning baseline ${Math.round((hub.calibration_progress || 0) * 100)}%`, "warn");
    if (hub.recording) pill("Already recording — stop it on /live first", "bad");
  }
  $("setup-status").replaceChildren(...pills);
  const ready = hub && hub.connected && hub.num_channels > 0 && !hub.recording && $("participant").value.trim();
  $("btn-start").disabled = run.active || !ready;
}
async function pollHub() {
  try { hub = await api("/api/eeg/status"); } catch (_) { hub = null; }
  if (!run.active) renderSetupStatus();
}
try { $("participant").value = localStorage.getItem("neurotutor_participant") || ""; } catch (_) { /* ignore */ }
for (const id of ["participant", "skill", "block-seconds", "quick"]) $(id).addEventListener("input", () => { renderPreview(); renderSetupStatus(); });

// --------------------------------------------------------------- primitives
function beep(freq = 880, ms = 300) {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const osc = ctx.createOscillator(); const gain = ctx.createGain();
    osc.frequency.value = freq; gain.gain.value = 0.15; osc.connect(gain); gain.connect(ctx.destination);
    osc.start(); osc.stop(ctx.currentTime + ms / 1000); setTimeout(() => ctx.close(), ms + 100);
  } catch (_) { /* audio is optional */ }
}

function setInstruction(title, text, sub = "") {
  const box = $("instruction");
  box.replaceChildren(el("span", { class: "big", text: title }), document.createTextNode(text));
  if (sub) box.append(el("span", { class: "sub", text: sub }));
}

// Resolves when the time is up, or early if the operator ends the block / session.
function countdown(seconds) {
  return new Promise((resolve) => {
    const end = performance.now() + seconds * 1000;
    const tick = () => {
      const left = (end - performance.now()) / 1000;
      $("timer").textContent = fmt(Math.ceil(left));
      $("timer-fill").style.width = `${Math.min(100, ((seconds - left) / seconds) * 100)}%`;
      if (left <= 0 || run.skip || run.abort) { clearInterval(timer); run.skip = false; resolve(left <= 0 ? "done" : "skipped"); }
    };
    const timer = setInterval(tick, 200);
    tick();
  });
}

function show(which) {
  $("stage").hidden = which !== "stage";
  $("fixation").hidden = which !== "fixation";
  $("ratings").hidden = which !== "ratings";
  $("rest").hidden = which !== "rest";
}

// ----------------------------------------------------------------- digit task
const NBACK_INTERVAL_MS = 2500;
const NBACK_SHOW_MS = 1200;
const NBACK_TARGET_PROB = 0.3;
const nback = { timer: null, prev: null, current: null, hits: 0, misses: 0, falseAlarms: 0 };

function nbackScore() {
  $("nback-score").textContent = `hits ${nback.hits} · misses ${nback.misses} · false alarms ${nback.falseAlarms}`;
}
function nbackStimulus() {
  if (nback.current && nback.current.isTarget && !nback.current.responded) { nback.misses += 1; marker("nback_response", { responded: false, correct: false, miss: true }); }
  let digit;
  if (nback.prev !== null && Math.random() < NBACK_TARGET_PROB) digit = nback.prev;
  else { do { digit = 1 + Math.floor(Math.random() * 9); } while (digit === nback.prev); }
  nback.current = { digit, isTarget: digit === nback.prev, responded: false };
  nback.prev = digit;
  $("nback-digit").textContent = digit;
  setTimeout(() => { if (nback.current && nback.current.digit === digit) $("nback-digit").textContent = " "; }, NBACK_SHOW_MS);
  marker("nback_stimulus", { digit, is_target: nback.current.isTarget });
  nbackScore();
}
function nbackKey(event) {
  if (!$("nback").hidden && (event.key === "m" || event.key === "M")) {
    event.preventDefault();
    const cur = nback.current;
    if (cur && !cur.responded) {
      cur.responded = true;
      if (cur.isTarget) { nback.hits += 1; marker("nback_response", { responded: true, correct: true }); }
      else { nback.falseAlarms += 1; marker("nback_response", { responded: true, correct: false }); }
    } else { nback.falseAlarms += 1; }
    nbackScore();
  }
}
window.addEventListener("keydown", nbackKey);
function startNback() {
  Object.assign(nback, { prev: null, current: null, hits: 0, misses: 0, falseAlarms: 0 });
  $("nback").hidden = false; nbackScore();
  nbackStimulus();
  nback.timer = setInterval(nbackStimulus, NBACK_INTERVAL_MS);
}
function stopNback() {
  clearInterval(nback.timer); nback.timer = null; nback.current = null;
  $("nback").hidden = true;
}
$("tutor-frame").addEventListener("load", () => {
  // Keystrokes inside the embedded tutor don't bubble to this window.
  try { $("tutor-frame").contentWindow.addEventListener("keydown", nbackKey); } catch (_) { /* cross-origin never happens here */ }
});

// ----------------------------------------------------------------- step types
async function runBaseline(step) {
  show("fixation");
  $("fixation").querySelector("span").textContent = step.condition === "Baseline_EC" ? "●" : "+";
  await marker("block_start", { block: step.id, condition: step.condition, block_kind: "baseline", seconds: step.seconds });
  if (step.condition === "Baseline_EO" && hub && hub.kind === "lsl") {
    // The live view's own normalisation baseline is learned from this eyes-open rest.
    api("/api/eeg/recalibrate", "POST", { seconds: step.seconds }).catch(() => {});
  }
  const result = await countdown(step.seconds);
  await marker("block_end", { block: step.id, condition: step.condition, aborted: result === "skipped" });
  if (step.condition === "Baseline_EC") beep();
}

function tutorUrl(step, participant) {
  const params = new URLSearchParams({
    embed: "1", source: "hub", mode: "puzzle", policy: "fixed",
    target_min: step.band[0], target_max: step.band[1], learner_id: `collect-${participant}`,
  });
  return `/?${params.toString()}`;
}

async function runTask(step, participant) {
  show("stage");
  $("tutor-frame").src = tutorUrl(step, participant);
  if (step.dual) startNback();
  await marker("block_start", {
    block: step.block, condition: step.condition, block_kind: "task", band: step.band, dual: step.dual, seconds: step.seconds,
  });
  beep(660, 150);
  const result = await countdown(step.seconds);
  await marker("block_end", { block: step.block, condition: step.condition, aborted: result === "skipped" });
  if (step.dual) {
    await marker("nback_summary", { block: step.block, hits: nback.hits, misses: nback.misses, false_alarms: nback.falseAlarms });
    stopNback();
  }
  $("tutor-frame").src = "about:blank";
  beep(440, 250);
}

const RATING_ITEMS = [
  ["mental_demand", "Mental demand", "How mentally demanding was this block?"],
  ["effort", "Effort", "How hard did you have to work?"],
  ["frustration", "Frustration / stress", "How irritated, stressed or rushed did you feel?"],
  ["confusion", "Confusion", "I often didn't understand what was going on."],
  ["fatigue", "Fatigue / sleepiness", "How tired did you feel?"],
  ["concentration", "Concentration", "How well could you concentrate?"],
];
function buildRatingForm() {
  $("rating-fields").replaceChildren(...RATING_ITEMS.map(([key, label, sub]) => {
    const out = el("output", { text: "4" });
    const input = el("input", { type: "range", min: "1", max: "7", step: "1", value: "4", name: key });
    input.addEventListener("input", () => { out.textContent = input.value; });
    return el("div", { class: "rating-item" }, [
      el("label", {}, [document.createTextNode(label), el("span", { class: "sub", text: sub })]), input, out,
    ]);
  }));
}
function askRatings(step) {
  buildRatingForm();
  show("ratings");
  setInstruction("How was that block?", "Answer for the block you just finished.");
  return new Promise((resolve) => {
    $("ratings").onsubmit = async (event) => {
      event.preventDefault();
      const ratings = {};
      $("rating-fields").querySelectorAll("input").forEach((i) => { ratings[i.name] = parseInt(i.value, 10); });
      await marker("rating", { block: step.block, condition: step.condition, ratings });
      $("ratings").onsubmit = null;
      resolve(ratings);
    };
  });
}

async function runRest(step) {
  show("rest");
  $("rest").textContent = "Rest — look away from the screen and relax.";
  setInstruction("Rest", "Take a breath. The next block starts automatically.");
  await marker("block_start", { block: `${step.id}-rest`, condition: "Rest", block_kind: "rest", seconds: step.restSeconds });
  await countdown(step.restSeconds);
  await marker("block_end", { block: `${step.id}-rest`, condition: "Rest" });
}

// ------------------------------------------------------------------- session
async function startSession() {
  const opts = readOptions();
  if (!opts.participant) { $("setup-msg").textContent = "Enter a participant id."; return; }
  try { localStorage.setItem("neurotutor_participant", opts.participant); } catch (_) { /* ignore */ }
  const steps = buildProtocol(opts);
  $("btn-start").disabled = true;
  try {
    await api("/api/record/start", "POST", { participant: opts.participant, label: opts.quick ? "dryrun" : "protocol" });
  } catch (err) { $("setup-msg").textContent = err.message; $("setup-msg").className = "live-msg error"; renderSetupStatus(); return; }
  await marker("protocol", { version: 1, options: opts, steps });

  run.active = true; run.abort = false; run.skip = false;
  $("setup").hidden = true; $("runner").hidden = false;
  window.onbeforeunload = () => "A recording is in progress.";

  try {
    for (let i = 0; i < steps.length && !run.abort; i++) {
      const step = steps[i];
      run.step = step;
      $("step-kind").textContent = step.type === "task" ? "puzzles" : "rest";
      $("step-title").textContent = step.title;
      $("step-progress").textContent = `step ${i + 1} of ${steps.length}`;
      setInstruction(step.title, step.text, `Duration ${fmt(step.seconds)}`);
      if (step.type === "baseline") { await runBaseline(step); }
      else {
        await runTask(step, opts.participant);
        if (step.rated && !run.abort) { await askRatings(step); if (!run.abort) await runRest(step); }
      }
    }
  } finally {
    await finishSession();
  }
}

async function finishSession() {
  stopNback();
  $("tutor-frame").src = "about:blank";
  run.active = false;
  window.onbeforeunload = null;
  let summary = "";
  let directory = "";
  try {
    const res = await api("/api/record/stop", "POST");
    directory = res.directory;
    summary = `${run.abort ? "Stopped early. " : ""}Saved ${res.samples.toLocaleString()} EEG samples and ${res.attempts} chess attempts to ${res.directory}.`;
  } catch (err) { summary = `Could not stop the recording cleanly: ${err.message}`; }
  $("runner").hidden = true; $("done").hidden = false;
  $("done-msg").textContent = summary;
  $("done-cmd").textContent = directory ? `python -m data_gen.from_recordings ${directory}` : "";
}

$("btn-start").addEventListener("click", startSession);
$("btn-skip").addEventListener("click", () => { run.skip = true; });
$("btn-abort").addEventListener("click", () => { if (confirm("Stop the session? The recording so far is kept.")) { run.abort = true; run.skip = true; } });
$("btn-note").addEventListener("click", async () => {
  await marker("note", { note: "disturbance" });
  const b = $("btn-note"); const old = b.textContent; b.textContent = "Marked ✓"; setTimeout(() => { b.textContent = old; }, 1200);
});

renderPreview();
pollHub();
setInterval(pollHub, 2000);
