// Live EEG monitor. Everything drawn here comes from /ws/live (the shared EEG hub):
// the streamed samples, a 1 Hz status/quality/band-power/state feed, and the
// tutor's own per-move updates. No synthetic decoration.

const STATES = ["Focused", "Overloaded", "Confused", "Fatigued", "Engaged"];
const STATE_COLORS = {
  Focused: "#4a7c2f", Overloaded: "#b91c1c", Confused: "#c2410c", Fatigued: "#6d28d9", Engaged: "#3f7fbf",
};
const CH_COLORS = ["#4a7c2f", "#b58863", "#3f7fbf", "#b91c1c", "#c2410c", "#6d28d9", "#0f766e", "#a16207"];
const BAND_COLORS = { theta: "#6d28d9", alpha: "#4a7c2f", beta: "#c2410c" };
const BAND_LABELS = { theta: "Theta 4–8", alpha: "Alpha 8–13", beta: "Beta 13–30" };

const WAVE_SECONDS = 4;
const HISTORY_SECONDS = 120;
const TIMELINE_SECONDS = 180;

const $ = (id) => document.getElementById(id);
function el(tag, props = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "text") node.textContent = v;
    else if (k === "class") node.className = v;
    else node.setAttribute(k, v);
  }
  for (const c of children) node.append(c);
  return node;
}
async function api(path, method = "GET", body) {
  const res = await fetch(path, {
    method, headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  let data = null;
  try { data = await res.json(); } catch (_) { /* empty body */ }
  if (!res.ok) throw new Error((data && data.detail) || `${res.status} ${res.statusText}`);
  return data;
}
function setMsg(id, text, kind = "") {
  const node = $(id);
  node.textContent = text;
  node.className = `live-msg ${kind}`.trim();
}
function fmtDuration(s) {
  s = Math.max(0, Math.round(s));
  const m = Math.floor(s / 60);
  return m ? `${m}m ${String(s % 60).padStart(2, "0")}s` : `${s}s`;
}

// ------------------------------------------------------------------ canvases
function fitCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, Math.floor(canvas.clientWidth * dpr));
  const h = Math.max(1, Math.floor(canvas.clientHeight * dpr));
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  return { ctx: canvas.getContext("2d"), w, h, dpr };
}

// ------------------------------------------------------------------ waveform
let numChannels = 8;
let sampleRate = 128;
let waveBuffers = [];
let waveGain = 1;
let wavePaused = false;
let qualityByChannel = [];
let channelNames = [];          // real electrode labels from the stream (e.g. Fz, Pz, ...) when known
const chName = (i) => channelNames[i] || `Ch${i + 1}`;

function resetWave(channels, rate) {
  numChannels = channels;
  sampleRate = rate;
  const len = Math.round(WAVE_SECONDS * rate);
  waveBuffers = Array.from({ length: channels }, () => new Float32Array(len));
  waveFill = 0;
}
let waveFill = 0;
function pushSamples(rows) {
  if (!rows.length) return;
  if (!waveBuffers.length || rows[0].length !== numChannels) resetWave(rows[0].length, sampleRate);
  const len = waveBuffers[0].length;
  const n = Math.min(rows.length, len);
  for (let ch = 0; ch < numChannels; ch++) {
    const buf = waveBuffers[ch];
    buf.copyWithin(0, n);
    for (let i = 0; i < n; i++) buf[len - n + i] = rows[rows.length - n + i][ch];
  }
  waveFill = Math.min(len, waveFill + n);
}
function drawWave() {
  const canvas = $("wave");
  const { ctx, w, h, dpr } = fitCanvas(canvas);
  ctx.clearRect(0, 0, w, h);
  const channels = waveBuffers.length;
  if (!channels) {
    ctx.fillStyle = "#9a948c"; ctx.font = `${14 * dpr}px Inter, sans-serif`;
    ctx.fillText("Waiting for data — connect a source above.", 16 * dpr, 28 * dpr);
    return;
  }
  const band = h / channels;
  const len = waveBuffers[0].length;
  const scale = band * 0.32 * waveGain; // ±1.5 normalised units ≈ most of a band
  ctx.lineWidth = 1 * dpr;
  for (let ch = 0; ch < channels; ch++) {
    const base = band * ch + band / 2;
    ctx.strokeStyle = "#ece8e1";
    ctx.beginPath(); ctx.moveTo(0, base); ctx.lineTo(w, base); ctx.stroke();
    const status = (qualityByChannel[ch] || {}).status;
    ctx.strokeStyle = status === "flat" ? "#adb5bd" : CH_COLORS[ch % CH_COLORS.length];
    ctx.beginPath();
    const buf = waveBuffers[ch];
    const start = len - waveFill;
    for (let i = start; i < len; i++) {
      const x = (i / (len - 1)) * w;
      const y = Math.max(band * ch + 2, Math.min(band * (ch + 1) - 2, base - buf[i] * scale));
      if (i === start) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.fillStyle = CH_COLORS[ch % CH_COLORS.length];
    ctx.font = `${11 * dpr}px Inter, sans-serif`;
    ctx.fillText(chName(ch), 6 * dpr, band * ch + 14 * dpr);
  }
}
function animate() {
  if (!wavePaused) drawWave();
  requestAnimationFrame(animate);
}

// -------------------------------------------------------------------- status
let lastStatus = null;
let sourceSynced = false;
function applyStatus(s) {
  lastStatus = s;
  if (!sourceSynced && s.connected && ["lsl", "sim", "replay"].includes(s.kind)) {
    $("src-kind").value = s.kind; updateSourceForm(); sourceSynced = true;
  }
  const connected = s.connected;
  $("h-status").textContent = connected ? (s.kind === "lsl" ? "Streaming (LSL)" : "Simulator") : (s.error ? "Source crashed" : "Not connected");
  if (s.error) setMsg("connect-msg", `Source stopped: ${s.error}`, "error");
  $("h-rate").textContent = connected
    ? (s.native_rate ? `${s.native_rate} → ${s.sample_rate} Hz` : `${s.sample_rate} Hz`) : "—";
  $("h-ch").textContent = connected ? String(s.num_channels) : "—";
  const age = s.seconds_since_data;
  $("h-age").textContent = !connected ? "—" : age == null ? "no data yet" : age < 1 ? "now" : `${age.toFixed(1)} s ago`;
  $("h-age").style.color = connected && age != null && age > 2 ? "#b3261e" : "";
  $("h-up").textContent = connected ? fmtDuration(s.uptime) : "—";
  $("h-rec").textContent = s.recording ? `● ${fmtDuration(s.recording.seconds)}` : "off";
  $("h-rec").style.color = s.recording ? "#b3261e" : "";

  // calibration
  $("calib-wrap").hidden = !(connected && s.calibrating);
  $("calib-fill").style.width = `${Math.round((s.calibration_progress || 0) * 100)}%`;

  // quality chips
  channelNames = s.channel_names || [];
  if (s.unit_warning) setMsg("connect-msg", s.unit_warning, "warn");
  qualityByChannel = s.quality || [];
  const grid = $("quality-grid");
  grid.replaceChildren(...qualityByChannel.map((q, i) =>
    el("div", { class: `q-chip ${q.status}` }, [
      el("strong", { text: chName(i) }),
      el("span", { text: (q.status === "simulated" || q.status === "replay") ? q.status : q.status === "no_data" ? "no data" : `${q.rms.toFixed(1)} µV · ${q.status}` }),
    ])));

  // model banner
  const lm = s.live_model;
  const banner = $("model-banner");
  if (lm) {
    banner.hidden = false;
    banner.className = `model-banner${lm.preliminary ? "" : " real"}`;
    const mismatch = lm.available && connected && lm.num_channels !== s.num_channels;
    if (mismatch) banner.className = "model-banner";
    banner.textContent = !lm.available
      ? "No EEG model checkpoint found — train one (see README)."
      : mismatch
        ? `No state estimate yet: the live model expects ${lm.num_channels} channels but this stream has ${s.num_channels}. ` +
          "Record sessions with /collect, then train one for this headset (python -m model.train_real … --eeg-only)."
        : lm.preliminary ? `Preliminary — ${lm.label}. Not yet validated on real recordings.` : lm.label;
  }

  // replay: show the TRUE recorded condition next to the model's estimate
  const info = $("replay-info");
  if (s.replay) {
    info.hidden = false;
    const pct = Math.round(s.replay.progress * 100);
    info.replaceChildren(
      el("span", { text: `Replaying ${s.replay.participant || "recording"} ×${s.replay.speed} · recorded condition now:` }),
      el("strong", { text: (s.replay.condition || "—").replace(/_/g, " ") }),
      el("div", { class: "bar" }, [(() => { const f = el("div", { class: "bar-fill" }); f.style.width = `${pct}%`; return f; })()]),
      el("span", { text: `${pct}%${s.replay.laps ? ` · lap ${s.replay.laps + 1}` : ""}` }),
    );
  } else { info.hidden = true; }

  // recording buttons
  $("btn-rec-start").disabled = !connected || !!s.recording;
  $("btn-rec-stop").disabled = !s.recording;
  $("btn-recalibrate").disabled = !connected || s.kind !== "lsl";
  $("h-status").textContent = connected ? ({ lsl: "Streaming (LSL)", sim: "Simulator", replay: "Replay" }[s.kind] || s.kind) : "Not connected";
  $("btn-disconnect").disabled = !connected;
  markerButtons.forEach((b) => { b.disabled = !connected; });
}

// --------------------------------------------------------------------- bands
const bandHistory = []; // {t, theta, alpha, beta}
function applyBands(b) {
  const names = b.bands;
  const bars = $("band-bars");
  bars.replaceChildren(...names.map((name, i) => {
    const pct = b.mean_relative[i] * 100;
    const fill = el("div", { class: "bar-fill" });
    fill.style.width = `${pct}%`; fill.style.background = BAND_COLORS[name];
    return el("div", { class: "band-row" }, [
      el("span", { text: BAND_LABELS[name] || name }),
      el("div", { class: "bar" }, [fill]),
      el("span", { text: `${pct.toFixed(0)}%` }),
    ]);
  }));

  const now = Date.now() / 1000;
  const entry = { t: now };
  names.forEach((n, i) => { entry[n] = b.mean_relative[i]; });
  bandHistory.push(entry);
  while (bandHistory.length && now - bandHistory[0].t > HISTORY_SECONDS) bandHistory.shift();
  drawBandHistory(names);

  const table = $("band-table");
  const cells = [el("span", { class: "h", text: "" }), ...names.map((n) => el("span", { class: "h", text: n.slice(0, 1).toUpperCase() + n.slice(1) }))];
  b.relative.forEach((row, ch) => {
    cells.push(el("span", { class: "h", text: chName(ch) }));
    row.forEach((v, i) => {
      const cell = el("span", { class: "cell", text: `${(v * 100).toFixed(0)}%` });
      cell.style.background = `${BAND_COLORS[names[i]]}${Math.round(18 + v * 60).toString(16).padStart(2, "0")}`;
      cells.push(cell);
    });
  });
  table.replaceChildren(...cells);
}
function drawBandHistory(names) {
  const { ctx, w, h, dpr } = fitCanvas($("band-history"));
  ctx.clearRect(0, 0, w, h);
  if (bandHistory.length < 2) return;
  const t1 = bandHistory[bandHistory.length - 1].t;
  ctx.lineWidth = 1.5 * dpr;
  for (const name of names) {
    ctx.strokeStyle = BAND_COLORS[name];
    ctx.beginPath();
    bandHistory.forEach((p, i) => {
      const x = w - ((t1 - p.t) / HISTORY_SECONDS) * w;
      const y = h - 4 * dpr - p[name] * (h - 8 * dpr);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }
}

// --------------------------------------------------------------------- state
const stateTimeline = []; // {t, state}
function applyState(s) {
  const badge = $("state-badge");
  badge.textContent = s.state;
  badge.style.background = STATE_COLORS[s.state] || "#adb5bd";
  $("state-conf").textContent = `${Math.round(s.confidence * 100)}% confidence · window ${s.history_seconds.toFixed(0)} s`;
  const bars = $("prob-bars");
  bars.replaceChildren(...Object.keys(s.probs).map((name) => {
    const fill = el("div", { class: "bar-fill" });
    fill.style.width = `${s.probs[name] * 100}%`; fill.style.background = STATE_COLORS[name] || "#888";
    return el("div", { class: "prob-row" }, [
      el("span", { text: name }), el("div", { class: "bar" }, [fill]), el("span", { text: `${(s.probs[name] * 100).toFixed(0)}%` }),
    ]);
  }));
  const now = Date.now() / 1000;
  stateTimeline.push({ t: now, state: s.state, confidence: s.confidence });
  while (stateTimeline.length && now - stateTimeline[0].t > TIMELINE_SECONDS) stateTimeline.shift();
  drawTimeline();
}
function drawTimeline() {
  const { ctx, w, h } = fitCanvas($("state-timeline"));
  ctx.clearRect(0, 0, w, h);
  if (!stateTimeline.length) return;
  const t1 = stateTimeline[stateTimeline.length - 1].t;
  const blockW = Math.max(2, w / TIMELINE_SECONDS);
  for (const p of stateTimeline) {
    const x = w - ((t1 - p.t) / TIMELINE_SECONDS) * w - blockW;
    ctx.globalAlpha = 0.35 + 0.65 * p.confidence;
    ctx.fillStyle = STATE_COLORS[p.state] || "#888";
    ctx.fillRect(x, 0, blockW + 0.5, h);
  }
  ctx.globalAlpha = 1;
}
function buildLegend() {
  $("state-legend").replaceChildren(...STATES.map((s) => {
    const node = el("span", { text: s });
    node.style.setProperty("--c", STATE_COLORS[s]);
    return node;
  }));
}

// -------------------------------------------------------------------- tutor
function applyTutor(u) {
  const act = u.action || {};
  const delta = act.difficulty_delta != null ? `${act.difficulty_delta >= 0 ? "+" : ""}${act.difficulty_delta}` : "";
  $("tutor-now").textContent =
    `${u.correct ? "✓ Correct" : "✗ Wrong"} · predicted ${u.predicted_state} (${Math.round(u.confidence * 100)}%) · ` +
    `difficulty ${Math.round(u.difficulty)} (${delta})${act.show_hint ? " · hint" : ""}${u.proactive_intervention ? " · proactive nudge" : ""}`;
  const li = el("li", { class: u.correct ? "ok" : "bad" }, [
    el("span", { text: `${new Date(u.t * 1000).toLocaleTimeString()} · ${u.predicted_state}` }),
    el("span", { text: `${u.correct ? "correct" : "wrong"} · d=${Math.round(u.difficulty)}` }),
  ]);
  const log = $("tutor-log");
  log.prepend(li);
  while (log.children.length > 30) log.lastChild.remove();
}

// ----------------------------------------------------------------- websocket
let ws = null;
let retry = 0;
function connectWs() {
  ws = new WebSocket(`ws://${location.host}/ws/live`);
  ws.onopen = () => { retry = 0; $("ws-dot").className = "connection-dot live"; $("ws-label").textContent = "Live"; };
  ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    if (m.type === "eeg_chunk") { if (!wavePaused) pushSamples(m.samples); }
    else if (m.type === "status") {
      if (m.connected && (m.num_channels !== numChannels || m.sample_rate !== sampleRate || !waveBuffers.length)) {
        resetWave(m.num_channels, m.sample_rate);
      }
      applyStatus(m);
    }
    else if (m.type === "bands") applyBands(m);
    else if (m.type === "state") applyState(m);
    else if (m.type === "tutor_update") applyTutor(m);
  };
  ws.onclose = () => {
    $("ws-dot").className = "connection-dot error"; $("ws-label").textContent = "Reconnecting…";
    retry += 1;
    setTimeout(connectWs, Math.min(1000 * 2 ** (retry - 1), 8000));
  };
}

// ------------------------------------------------------------------ controls
function updateSourceForm() {
  const kind = $("src-kind").value;
  for (const id of ["lbl-stream-name", "lbl-channel-names", "lbl-channels", "lbl-notch", "lbl-calib"]) $(id).hidden = kind !== "lsl";
  $("lbl-simstate").hidden = kind !== "sim";
  $("lbl-recording").hidden = kind !== "replay";
  $("adv-lsl").hidden = kind !== "lsl";
  $("lbl-speed").hidden = kind !== "replay";
}
$("src-kind").addEventListener("change", updateSourceForm);

$("btn-connect").addEventListener("click", async () => {
  const kind = $("src-kind").value;
  const body = { source: kind, sim_state: $("sim-state").value };
  if (kind === "replay") {
    body.recording = $("rep-recording").value;
    body.speed = parseFloat($("rep-speed").value) || 1;
    if (!body.recording) { setMsg("connect-msg", "No saved recordings yet.", "error"); return; }
  }
  if (kind === "lsl") {
    body.stream_name = $("src-name").value.trim() || null;
    const names = $("src-channel-names").value.trim();
    if (names) body.channel_names = names.split(",").map((x) => x.trim()).filter(Boolean);
    const ch = $("src-channels").value.trim();
    if (ch) body.channels = ch.split(",").map((x) => parseInt(x, 10)).filter((x) => !Number.isNaN(x));
    const notch = $("src-notch").value;
    body.notch_hz = notch === "" ? null : parseFloat(notch);
    body.calibration_seconds = parseFloat($("src-calib").value) || 30;
  }
  $("btn-connect").disabled = true;
  setMsg("connect-msg", kind === "lsl" ? "Looking for an LSL stream (up to 5 s)…" : "Starting simulator…");
  try {
    const res = await api("/api/eeg/connect", "POST", body);
    waveBuffers = []; bandHistory.length = 0; stateTimeline.length = 0;
    setMsg("connect-msg", res.warnings.length ? res.warnings.join(" ") : "Connected.", res.warnings.length ? "warn" : "ok");
  } catch (err) {
    setMsg("connect-msg", err.message, "error");
  } finally {
    $("btn-connect").disabled = false;
  }
});
$("btn-disconnect").addEventListener("click", async () => {
  try { await api("/api/eeg/disconnect", "POST"); setMsg("connect-msg", "Disconnected."); }
  catch (err) { setMsg("connect-msg", err.message, "error"); }
});
$("btn-recalibrate").addEventListener("click", async () => {
  try {
    await api("/api/eeg/recalibrate", "POST", { seconds: parseFloat($("src-calib").value) || 30 });
    setMsg("connect-msg", "Learning a new resting baseline — ask the participant to sit still, eyes open.", "ok");
  } catch (err) { setMsg("connect-msg", err.message, "error"); }
});
$("sim-state").addEventListener("change", async () => {
  if (lastStatus && lastStatus.connected && lastStatus.kind === "sim") {
    try { await api("/api/eeg/sim-state", "POST", { state: $("sim-state").value }); } catch (err) { setMsg("connect-msg", err.message, "error"); }
  }
});
$("wave-gain").addEventListener("input", (e) => { waveGain = parseFloat(e.target.value); });
$("wave-pause").addEventListener("click", () => {
  wavePaused = !wavePaused;
  $("wave-pause").textContent = wavePaused ? "Resume" : "Pause";
});

// recording
const PARTICIPANT_KEY = "neurotutor_participant";
try { $("rec-participant").value = localStorage.getItem(PARTICIPANT_KEY) || ""; } catch (_) { /* storage unavailable */ }
$("btn-rec-start").addEventListener("click", async () => {
  const participant = $("rec-participant").value.trim();
  if (!participant) { setMsg("record-msg", "Enter a participant id first.", "error"); return; }
  try { localStorage.setItem(PARTICIPANT_KEY, participant); } catch (_) { /* ignore */ }
  try {
    const res = await api("/api/record/start", "POST", { participant, label: $("rec-label").value.trim() });
    setMsg("record-msg", `Recording to ${res.directory}`, "ok");
  } catch (err) { setMsg("record-msg", err.message, "error"); }
});
$("btn-rec-stop").addEventListener("click", async () => {
  try {
    const res = await api("/api/record/stop", "POST");
    setMsg("record-msg", `Saved ${res.samples.toLocaleString()} samples and ${res.attempts} chess attempts to ${res.directory}`, "ok");
    loadRecordings();
  } catch (err) { setMsg("record-msg", err.message, "error"); }
});

const markerButtons = [];
function buildMarkerButtons() {
  const row = $("marker-row");
  const defs = [
    ["Baseline", { kind: "block_start", data: { condition: "Baseline", block: "manual" } }],
    ["Focused", { kind: "block_start", data: { condition: "Focused", block: "manual" } }],
    ["Struggling", { kind: "block_start", data: { condition: "Struggling", block: "manual" } }],
    ["Fatigued", { kind: "block_start", data: { condition: "Fatigued", block: "manual" } }],
    ["End block", { kind: "block_end", data: { block: "manual" } }],
  ];
  for (const [label, payload] of defs) {
    const btn = el("button", { class: "btn btn-secondary", type: "button", text: label });
    btn.addEventListener("click", async () => {
      try {
        await api("/api/record/marker", "POST", payload);
        setMsg("record-msg", `Marker: ${label}`, "ok");
      } catch (err) { setMsg("record-msg", err.message, "error"); }
    });
    markerButtons.push(btn);
    row.append(btn);
  }
}

async function loadRecordings() {
  try {
    const { recordings } = await api("/api/record/list");
    $("rep-recording").replaceChildren(...recordings.filter((r) => r.ended_at).map((r) =>
      el("option", { value: r.directory, text: `${r.participant} · ${new Date(r.started_at * 1000).toLocaleString()} · ${fmtDuration(r.ended_at - r.started_at)}` })));
    const list = $("rec-list");
    list.replaceChildren(...recordings.slice(0, 12).map((r) => el("li", {}, [
      el("span", { text: `${r.participant} · ${r.started_at ? new Date(r.started_at * 1000).toLocaleString() : ""}` }),
      el("span", { text: r.ended_at ? `${fmtDuration(r.ended_at - r.started_at)}` : "unfinished" }),
    ])));
    if (!recordings.length) list.replaceChildren(el("li", { text: "None yet." }));
  } catch (_) { /* list is a convenience */ }
}

// ---------------------------------------------------------------------- init
for (const s of STATES) $("sim-state").append(el("option", { value: s, text: s }));
buildLegend();
buildMarkerButtons();
updateSourceForm();
loadRecordings();
connectWs();
requestAnimationFrame(animate);
window.addEventListener("resize", () => { drawBandHistory(["theta", "alpha", "beta"]); drawTimeline(); });
