// Guided data-collection runner. It walks a participant through the protocol in
// protocol.js, and for every block posts block_start / block_end markers (plus
// the self-report ratings) to the server's recorder — so the training labels
// are a by-product of running the session, never hand-annotated afterwards.

const { buildProtocol, totalSeconds, RATING_QUESTIONS } = window.NeuroProtocol;
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
    el("div", {}, [
      el("b", { text: s.type === "task" ? (s.rated ? s.title : s.title) : s.title }),
      el("br"),
      el("span", { text: `${fmt(s.seconds)}${s.dual ? " · with digit task" : ""}${s.rated ? " · then 3 quick questions" : ""}` }),
    ]),
  ])));
  $("plan-total").textContent = `about ${Math.round(totalSeconds(steps) / 60)} min in total`;
}
function renderSetupStatus() {
  const rows = [];
  const row = (text, cls) => rows.push(el("li", { class: cls, text }));
  if (!hub || !hub.connected) {
    row("No headset connected. Open Live signal and connect it first.", "bad");
  } else {
    row(hub.kind === "lsl" ? "Headset streaming" : "Simulator, so this is a test run only (no real signal)", hub.kind === "lsl" ? "good" : "warn");
    const names = (hub.channel_names || []).join(", ");
    row(`${hub.num_channels} channels${names ? `: ${names}` : ""}`, hub.num_channels > 0 ? "good" : "bad");
    const q = hub.quality || [];
    const bad = q.filter((c) => c.status === "flat" || c.status === "noisy").map((c, i) => (hub.channel_names || [])[q.indexOf(c)] || `Ch${i + 1}`);
    if (hub.kind === "lsl") row(bad.length === 0 ? "Signal looks good on every channel" : `Poor contact on ${bad.join(", ")}. Fix before starting.`, bad.length === 0 ? "good" : "warn");
    if (hub.calibrating) row(`Learning the resting baseline: ${Math.round((hub.calibration_progress || 0) * 100)}%. Keep still.`, "warn");
    if (hub.tutor_model_channels && hub.tutor_model_channels !== hub.num_channels) {
      row("The tutor model has a different channel count. Recording is not affected.", "warn");
    }
    if (hub.kind === "replay") row("A replay is connected. Connect the headset to record a new session.", "warn");
    if (hub.recording) row("A recording is already running. Stop it on Live signal first.", "bad");
  }
  $("setup-status").replaceChildren(...rows);
  const ready = hub && hub.connected && hub.num_channels > 0 && !hub.recording && $("participant").value.trim();
  $("btn-start").disabled = run.active || !ready;
}
// The run bar's per-channel chips (green / orange / grey) so contact loss is visible mid-session.
function renderSignal() {
  const box = $("sig");
  if (!hub || !hub.connected) { box.replaceChildren(el("span", { class: "sig-chip bad", text: "no signal" })); return; }
  const names = (hub.channel_names || []).length ? hub.channel_names : Array.from({ length: hub.num_channels || 0 }, (_, i) => `Ch${i + 1}`);
  const q = hub.quality || [];
  box.replaceChildren(...names.map((n, i) => el("span", { class: `sig-chip ${(q[i] && q[i].status) || "no_data"}`, text: n })));
  if (hub.seconds_since_data != null && hub.seconds_since_data > 3) box.append(el("span", { class: "sig-chip noisy", text: "stream stalled" }));
}
async function pollHub() {
  try { hub = await api("/api/eeg/status"); } catch (_) { hub = null; }
  if (run.active) renderSignal(); else renderSetupStatus();
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

const SCREENS = ["intro", "stage", "fixation", "ratings", "rest"];
function show(which) { for (const id of SCREENS) $(id).hidden = id !== which; }
function setSkippable(on) { $("btn-skip").disabled = !on; }

// Resolves when the time is up, or early if the operator ends the block / session.
function countdown(seconds) {
  run.skip = false;
  setSkippable(true);
  return new Promise((resolve) => {
    // Wall clock, not performance.now(): that one stops while the laptop sleeps, but the recorder's markers don't.
    const end = Date.now() + seconds * 1000;
    const tick = () => {
      const left = (end - Date.now()) / 1000;
      $("timer").textContent = fmt(Math.ceil(left));
      $("timer-fill").style.width = `${Math.min(100, ((seconds - left) / seconds) * 100)}%`;
      if (left <= 0 || run.skip || run.abort) {
        clearInterval(timer); setSkippable(false); const skipped = run.skip; run.skip = false;
        resolve(left <= 0 && !skipped ? "done" : "skipped");
      }
    };
    const timer = setInterval(tick, 200);
    tick();
  });
}

// Before each puzzle block: what's about to happen, with a short automatic countdown (Space or the
// button starts it sooner). The EEG keeps recording, but no block marker is written until it starts.
function runIntro(step, seconds) {
  show("intro");
  $("timer").textContent = "–"; $("timer-fill").style.width = "0%";
  $("intro-kind").textContent = step.dual ? "Puzzles + digit task" : "Puzzles";
  $("intro-title").textContent = step.title;
  $("intro-text").textContent = step.text;
  return new Promise((resolve) => {
    let left = seconds;
    const done = () => { clearInterval(timer); window.removeEventListener("keydown", onKey); $("btn-go").onclick = null; resolve(); };
    const onKey = (e) => { if (e.code === "Space") { e.preventDefault(); done(); } };
    const timer = setInterval(() => {
      left -= 1;
      $("intro-count").textContent = left > 0 ? `starts in ${left} s` : "";
      if (left <= 0 || run.abort) done();
    }, 1000);
    $("intro-count").textContent = `starts in ${left} s`;
    $("btn-go").onclick = done;
    window.addEventListener("keydown", onKey);
  });
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
  $("fix-mark").textContent = step.condition === "Baseline_EC" ? "●" : "+";
  $("fix-text").textContent = step.text;
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

// Three one-tap questions (definitions live in protocol.js so the dataset builder stays in step).
function buildRatingForm(answers, onChange) {
  $("rating-fields").replaceChildren(...RATING_QUESTIONS.map((q) => {
    const group = el("div", { class: "q-options", role: "radiogroup", "aria-label": q.prompt, style: `--n:${q.options.length}` });
    for (const option of q.options) {
      const button = el("button", { type: "button", class: "q-opt", role: "radio", "aria-checked": "false" }, [
        el("span", { class: "l", text: option.label }), ...(option.hint ? [el("span", { class: "h", text: option.hint })] : []),
      ]);
      button.addEventListener("click", () => {
        answers[q.key] = option.value;
        group.querySelectorAll(".q-opt").forEach((b) => b.setAttribute("aria-checked", String(b === button)));
        onChange();
      });
      group.append(button);
    }
    return el("div", {}, [el("p", { class: "q-prompt", text: q.prompt }), group]);
  }));
}
function askRatings(step) {
  const answers = {};
  const complete = () => RATING_QUESTIONS.every((q) => answers[q.key] !== undefined);
  buildRatingForm(answers, () => {
    $("btn-ratings").disabled = !complete();
    $("ratings-hint").textContent = complete() ? "Thanks. Press Continue." : "Tap one answer in each row.";
  });
  $("btn-ratings").disabled = true;
  $("ratings-hint").textContent = "Tap one answer in each row.";
  show("ratings");
  $("timer").textContent = "–"; $("timer-fill").style.width = "0%";
  return new Promise((resolve) => {
    $("ratings-form").onsubmit = async (event) => {
      event.preventDefault();
      if (!complete()) return;
      $("ratings-form").onsubmit = null;
      await marker("rating", { block: step.block, condition: step.condition, ratings: { ...answers } });
      resolve(answers);
    };
    // The operator may stop the session while the questions are on screen.
    const watch = setInterval(() => { if (run.abort) { clearInterval(watch); $("ratings-form").onsubmit = null; resolve(null); } }, 300);
  });
}

async function runRest(step) {
  show("rest");
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
  $("run-head").hidden = false; document.body.classList.add("running"); setSkippable(false); renderSignal();
  window.onbeforeunload = () => "A recording is in progress.";

  try {
    for (let i = 0; i < steps.length && !run.abort; i++) {
      const step = steps[i];
      run.step = step;
      $("step-kind").textContent = step.type === "task" ? "puzzles" : "rest";
      $("step-title").textContent = step.title;
      $("step-progress").textContent = `${i + 1} / ${steps.length}`;
      if (step.type === "baseline") { await runBaseline(step); }
      else {
        await runIntro(step, opts.quick ? 3 : 6);
        if (run.abort) break;
        await runTask(step, opts.participant);
        if (step.rated && !run.abort) { const r = await askRatings(step); if (r && !run.abort) await runRest(step); }
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
  $("runner").hidden = true; $("run-head").hidden = true; document.body.classList.remove("running"); $("done").hidden = false;
  $("done-msg").textContent = summary;
  $("done-cmd").textContent = directory ? `python -m data_gen.from_recordings ${directory}` : "";
}

$("btn-start").addEventListener("click", startSession);
$("btn-skip").addEventListener("click", () => { run.skip = true; });
$("btn-abort").addEventListener("click", () => { if (confirm("Stop the session? The recording so far is kept.")) { run.abort = true; run.skip = true; } });
$("btn-copy").addEventListener("click", () => {
  navigator.clipboard.writeText($("done-cmd").textContent).then(() => { $("btn-copy").textContent = "Copied ✓"; }).catch(() => {});
});
$("btn-note").addEventListener("click", async () => {
  await marker("note", { note: "disturbance" });
  const b = $("btn-note"); const old = b.textContent; b.textContent = "Marked ✓"; setTimeout(() => { b.textContent = old; }, 1200);
});

renderPreview();
pollHub();
setInterval(pollHub, 2000);
