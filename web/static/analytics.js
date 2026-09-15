const SVG_NS = "http://www.w3.org/2000/svg";
const KNOWN_STATES = ["Focused", "Overloaded", "Confused", "Fatigued", "Engaged"];
const STATE_COLORS = {
  Focused: "#1d4ed8", Overloaded: "#b91c1c", Confused: "#c2410c",
  Fatigued: "#6d28d9", Engaged: "#15803d",
};
const INITIAL_DIFFICULTY = 1000.0; // must match web/session.py's TutorSession default
const DIFFICULTY_FLOOR = 400.0; // must match web/session.py's max(400.0, ...) floor

function svgEl(tag, attrs) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
}

function formatTimestamp(unixSeconds) {
  return new Date(unixSeconds * 1000).toLocaleString();
}

function statCard(value, label) {
  const card = document.createElement("div");
  card.className = "stat-card";
  const valueEl = document.createElement("div");
  valueEl.className = "stat-card-value";
  valueEl.textContent = value;
  const labelEl = document.createElement("div");
  labelEl.className = "stat-card-label";
  labelEl.textContent = label;
  card.append(valueEl, labelEl);
  return card;
}

async function loadAggregate() {
  try {
    const response = await fetch("/analytics/aggregate");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    renderAggregate("puzzle", data.puzzle);
    renderAggregate("game", data.game);
  } catch (err) {
    for (const mode of ["puzzle", "game"]) {
      document.getElementById(`aggregate-status-${mode}`).textContent = `Could not load cohort overview: ${err.message}`;
    }
  }
}

// Puzzle-mode and full-game-mode cohorts are rendered into separately-suffixed
// element ids (…-puzzle / …-game) by this same function — never combined,
// since full-game mode repurposes puzzle_rating to mean the live
// adaptive-difficulty number, a different signal from a real puzzle rating.
function renderAggregate(mode, data) {
  const statusEl = document.getElementById(`aggregate-status-${mode}`);
  if (data.total_sessions === 0) {
    statusEl.textContent = mode === "game"
      ? "No full-game sessions recorded yet."
      : "No sessions with recorded attempts yet — play a few puzzles first.";
  } else {
    statusEl.textContent = `${data.total_sessions} session${data.total_sessions === 1 ? "" : "s"}, ${data.total_attempts} attempts, aggregated live from storage.db.`;
  }

  const grid = document.getElementById(`aggregate-stat-grid-${mode}`);
  grid.innerHTML = "";
  grid.append(
    statCard(data.total_sessions, "Sessions"),
    statCard(data.total_attempts, "Total attempts"),
    statCard(`${(data.overall_accuracy * 100).toFixed(0)}%`, "Overall accuracy"),
    statCard(`${data.avg_latency.toFixed(3)}s`, "Avg sense-to-adapt latency"),
    statCard(
      data.engagement_trend_avg !== null
        ? `${data.engagement_trend_avg >= 0 ? "+" : ""}${(data.engagement_trend_avg * 100).toFixed(0)}pp`
        : "N/A",
      "Engagement trend (2nd half − 1st half accuracy)",
    ),
  );

  renderAggregateStateChart(mode, data.state_distribution);

  const stateBody = document.getElementById(`aggregate-state-body-${mode}`);
  stateBody.innerHTML = "";
  for (const row of data.state_distribution) {
    const tr = document.createElement("tr");
    for (const value of [
      row.state, row.count, `${(row.accuracy * 100).toFixed(0)}%`, row.avg_confidence.toFixed(2),
    ]) {
      const td = document.createElement("td");
      td.textContent = value;
      tr.appendChild(td);
    }
    stateBody.appendChild(tr);
  }

  const comparison = data.policy_comparison;
  renderPolicyComparisonChart(mode, comparison);

  const comparisonGrid = document.getElementById(`policy-comparison-grid-${mode}`);
  comparisonGrid.innerHTML = "";
  comparisonGrid.append(
    statCard(comparison.total_attempts, "Attempts traced"),
    statCard(comparison.divergent_attempts, "Diverged from rule-based"),
    statCard(
      comparison.divergence_rate !== null ? `${(comparison.divergence_rate * 100).toFixed(0)}%` : "N/A",
      "Divergence rate",
    ),
    statCard(
      comparison.avg_absolute_delta_difference !== null ? comparison.avg_absolute_delta_difference.toFixed(1) : "N/A",
      "Avg |actual − rule-based| delta",
    ),
  );
}

// Same bar-chart shape as the per-session state-chart below, but summed
// across EVERY session's attempts instead of one session's — the cohort-wide
// counterpart the per-session chart alone can't show.
function renderAggregateStateChart(mode, stateDistribution) {
  const svg = document.getElementById(`aggregate-state-chart-${mode}`);
  const width = 640, height = 220, padding = { left: 40, right: 20, top: 16, bottom: 40 };
  const { plotWidth, plotHeight } = chartFrame(svg, width, height, padding);
  if (stateDistribution.length === 0) return;

  const maxCount = Math.max(1, ...stateDistribution.map((row) => row.count));
  const barWidth = plotWidth / stateDistribution.length;

  stateDistribution.forEach((row, i) => {
    const barHeight = (row.count / maxCount) * plotHeight;
    const x = padding.left + i * barWidth + barWidth * 0.15;
    const y = padding.top + plotHeight - barHeight;
    svg.appendChild(svgEl("rect", {
      x, y, width: barWidth * 0.7, height: barHeight, fill: STATE_COLORS[row.state] || "#7a726a",
    }));
    svg.appendChild(svgEl("text", {
      x: x + barWidth * 0.35, y: padding.top + plotHeight + 16, class: "chart-axis-label", "text-anchor": "middle",
    })).textContent = row.state;
    svg.appendChild(svgEl("text", {
      x: x + barWidth * 0.35, y: y - 4, class: "chart-axis-label", "text-anchor": "middle",
    })).textContent = `${row.count} (${(row.accuracy * 100).toFixed(0)}%)`;
  });
}

// A single stacked bar — matching (rule-based) vs. diverged (RL policy
// actually chose something different) — the explicit visual counterpart to
// the divergence-rate number in the stat grid below it.
function renderPolicyComparisonChart(mode, comparison) {
  const svg = document.getElementById(`policy-comparison-chart-${mode}`);
  svg.innerHTML = "";
  const width = 640, height = 90, padding = { left: 20, right: 20, top: 10 };
  const barWidth = width - padding.left - padding.right;
  const barHeight = 32;
  const y = padding.top + 10;

  if (!comparison.total_attempts) {
    svg.appendChild(svgEl("text", { x: width / 2, y: height / 2, class: "chart-axis-label", "text-anchor": "middle" }))
      .textContent = "No attempts recorded yet.";
    return;
  }

  const divergedFraction = comparison.divergent_attempts / comparison.total_attempts;
  const matchedWidth = barWidth * (1 - divergedFraction);
  const divergedWidth = barWidth * divergedFraction;

  svg.appendChild(svgEl("rect", { x: padding.left, y, width: matchedWidth, height: barHeight, fill: "#eef2ea" }));
  svg.appendChild(svgEl("rect", {
    x: padding.left + matchedWidth, y, width: divergedWidth, height: barHeight, fill: "#ede9fe",
  }));
  svg.appendChild(svgEl("rect", {
    x: padding.left, y, width: barWidth, height: barHeight, fill: "none", stroke: "#c9c4bb",
  }));

  svg.appendChild(svgEl("text", {
    x: padding.left + matchedWidth / 2, y: y + barHeight / 2 + 4, class: "chart-legend",
    fill: "var(--accent-dark)", "text-anchor": "middle",
  })).textContent = matchedWidth > 90 ? `Matched rule-based (${((1 - divergedFraction) * 100).toFixed(0)}%)` : "";
  svg.appendChild(svgEl("text", {
    x: padding.left + matchedWidth + divergedWidth / 2, y: y + barHeight / 2 + 4, class: "chart-legend",
    fill: "#6d28d9", "text-anchor": "middle",
  })).textContent = divergedWidth > 90 ? `Diverged / RL policy (${(divergedFraction * 100).toFixed(0)}%)` : "";
}

async function loadSessions() {
  const statusEl = document.getElementById("sessions-status");
  const listEl = document.getElementById("session-list");
  try {
    const response = await fetch("/sessions");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const sessions = await response.json();
    if (sessions.length === 0) {
      statusEl.textContent = "No sessions with recorded attempts yet — play a few puzzles first.";
      return;
    }
    statusEl.textContent = `${sessions.length} session${sessions.length === 1 ? "" : "s"}`;
    listEl.innerHTML = "";
    for (const session of sessions) {
      const li = document.createElement("li");
      li.className = "session-item";
      li.dataset.sessionId = session.session_id;
      const title = document.createElement("div");
      title.className = "session-item-title";
      title.textContent = formatTimestamp(session.started_at);
      const modeBadge = document.createElement("span");
      modeBadge.className = "session-item-mode";
      modeBadge.textContent = session.mode === "game" ? "Full game" : "Puzzle";
      title.appendChild(modeBadge);
      const stats = document.createElement("div");
      stats.className = "session-item-stats";
      stats.textContent =
        `${session.num_attempts} attempts · ${(session.accuracy_rate * 100).toFixed(0)}% accuracy · ` +
        `${session.avg_latency.toFixed(3)}s avg sense-to-adapt latency`;
      li.append(title, stats);
      li.addEventListener("click", () => selectSession(session.session_id, li));
      listEl.appendChild(li);
    }
  } catch (err) {
    statusEl.textContent = `Could not load sessions: ${err.message}`;
  }
}

async function selectSession(sessionId, listItemEl) {
  document.querySelectorAll(".session-item.selected").forEach((el) => el.classList.remove("selected"));
  listItemEl.classList.add("selected");
  document.getElementById("empty-state").hidden = true;

  const detailSection = document.getElementById("detail-section");
  try {
    const [analyticsRes, coachRes] = await Promise.all([
      fetch(`/session/${sessionId}/analytics`),
      fetch(`/session/${sessionId}/coach`),
    ]);
    if (!analyticsRes.ok) throw new Error(`HTTP ${analyticsRes.status}`);
    const data = await analyticsRes.json();
    const coach = coachRes.ok ? await coachRes.json() : null;
    renderDetail(sessionId, data, coach);
    detailSection.hidden = false;
  } catch (err) {
    detailSection.hidden = true;
    document.getElementById("empty-state").hidden = false;
    document.getElementById("empty-state").textContent = `Could not load session analytics: ${err.message}`;
  }
}

function renderDetail(sessionId, data, coach) {
  document.getElementById("detail-session-id").textContent = sessionId;
  document.getElementById("session-export-link").href = `/session/${sessionId}/export`;
  renderStatGrid(data.summary, data.attempts);
  renderCoaching(coach);
  renderAccuracyChart(data.attempts);
  renderDifficultyChart(data.attempts);
  renderStateChart(data.attempts);
  renderCalculations(data.calculations);
}

function renderCoaching(coach) {
  const sourceEl = document.getElementById("coaching-source");
  const narrativeEl = document.getElementById("coaching-narrative");
  const tipsEl = document.getElementById("coaching-tips");
  tipsEl.innerHTML = "";

  if (!coach) {
    sourceEl.textContent = "";
    narrativeEl.hidden = true;
    const li = document.createElement("li");
    li.textContent = "Coaching feedback could not be loaded for this session.";
    tipsEl.appendChild(li);
    return;
  }

  sourceEl.textContent = coach.narrative ? "— written by Claude from the stats below" : "— rule-based, from the stats below";
  if (coach.narrative) {
    narrativeEl.textContent = coach.narrative;
    narrativeEl.hidden = false;
  } else {
    narrativeEl.hidden = true;
  }
  for (const tip of coach.tips || []) {
    const li = document.createElement("li");
    li.textContent = tip;
    tipsEl.appendChild(li);
  }
}

function renderStatGrid(summary, attempts) {
  const hintRate = attempts.length ? attempts.filter((a) => a.show_hint).length / attempts.length : 0;
  const avgEvalLoss = attempts.length ? attempts.reduce((sum, a) => sum + a.eval_loss, 0) / attempts.length : 0;
  const stats = [
    { label: "Attempts", value: summary.num_attempts },
    { label: "Accuracy", value: `${(summary.accuracy_rate * 100).toFixed(0)}%` },
    { label: "Avg sense-to-adapt latency", value: `${summary.avg_latency.toFixed(3)}s` },
    { label: "Avg eval loss (centipawns)", value: avgEvalLoss.toFixed(0) },
    { label: "Attempts shown a hint", value: `${(hintRate * 100).toFixed(0)}%` },
  ];
  const grid = document.getElementById("stat-grid");
  grid.innerHTML = "";
  for (const { label, value } of stats) {
    const card = document.createElement("div");
    card.className = "stat-card";
    const valueEl = document.createElement("div");
    valueEl.className = "stat-card-value";
    valueEl.textContent = value;
    const labelEl = document.createElement("div");
    labelEl.className = "stat-card-label";
    labelEl.textContent = label;
    card.append(valueEl, labelEl);
    grid.appendChild(card);
  }
}

// Shared axis/frame drawing for the line charts below.
function chartFrame(svg, width, height, padding) {
  svg.innerHTML = "";
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  svg.appendChild(svgEl("line", {
    x1: padding.left, y1: height - padding.bottom, x2: width - padding.right, y2: height - padding.bottom,
    stroke: "#c9c4bb", "stroke-width": 1,
  }));
  svg.appendChild(svgEl("line", {
    x1: padding.left, y1: padding.top, x2: padding.left, y2: height - padding.bottom,
    stroke: "#c9c4bb", "stroke-width": 1,
  }));
  return { plotWidth, plotHeight };
}

function linePath(points) {
  return points.map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");
}

function renderAccuracyChart(attempts) {
  const svg = document.getElementById("accuracy-chart");
  const width = 640, height = 220, padding = { left: 40, right: 20, top: 16, bottom: 28 };
  const { plotWidth, plotHeight } = chartFrame(svg, width, height, padding);
  if (attempts.length === 0) return;

  // Cumulative accuracy after each attempt — a real running calculation
  // (correct_so_far / attempts_so_far), not a smoothed/fabricated curve.
  let correctSoFar = 0;
  const accuracyPoints = attempts.map((a, i) => {
    if (a.correct) correctSoFar += 1;
    const x = padding.left + (i / Math.max(1, attempts.length - 1)) * plotWidth;
    const y = padding.top + (1 - correctSoFar / (i + 1)) * plotHeight;
    return { x, y };
  });
  const confidencePoints = attempts.map((a, i) => ({
    x: padding.left + (i / Math.max(1, attempts.length - 1)) * plotWidth,
    y: padding.top + (1 - a.confidence) * plotHeight,
  }));

  svg.appendChild(svgEl("path", { d: linePath(accuracyPoints), fill: "none", stroke: "#4a7c2f", "stroke-width": 2 }));
  svg.appendChild(svgEl("path", { d: linePath(confidencePoints), fill: "none", stroke: "#3f7fbf", "stroke-width": 2, "stroke-dasharray": "4,3" }));

  svg.appendChild(svgEl("text", { x: padding.left, y: 12, class: "chart-legend", fill: "#4a7c2f" })).textContent = "Cumulative accuracy";
  svg.appendChild(svgEl("text", { x: padding.left + 170, y: 12, class: "chart-legend", fill: "#3f7fbf" })).textContent = "Model confidence";
  svg.appendChild(svgEl("text", { x: 4, y: padding.top + 4, class: "chart-axis-label" })).textContent = "100%";
  svg.appendChild(svgEl("text", { x: 4, y: height - padding.bottom, class: "chart-axis-label" })).textContent = "0%";
}

function renderDifficultyChart(attempts) {
  const svg = document.getElementById("difficulty-chart");
  const width = 640, height = 180, padding = { left: 50, right: 20, top: 16, bottom: 28 };
  const { plotWidth, plotHeight } = chartFrame(svg, width, height, padding);
  if (attempts.length === 0) return;

  // Reconstructs the same running calculation web/session.py's TutorSession
  // performs live: difficulty = max(400, difficulty + delta), starting at 1000.
  let difficulty = INITIAL_DIFFICULTY;
  const series = attempts.map((a) => {
    difficulty = Math.max(DIFFICULTY_FLOOR, difficulty + a.difficulty_delta);
    return difficulty;
  });
  const minD = Math.min(DIFFICULTY_FLOOR, ...series);
  const maxD = Math.max(INITIAL_DIFFICULTY, ...series);
  const points = series.map((d, i) => ({
    x: padding.left + (i / Math.max(1, series.length - 1)) * plotWidth,
    y: padding.top + (1 - (d - minD) / Math.max(1, maxD - minD)) * plotHeight,
  }));
  svg.appendChild(svgEl("path", { d: linePath(points), fill: "none", stroke: "#b58863", "stroke-width": 2 }));
  svg.appendChild(svgEl("text", { x: 4, y: padding.top + 4, class: "chart-axis-label" })).textContent = maxD.toFixed(0);
  svg.appendChild(svgEl("text", { x: 4, y: height - padding.bottom, class: "chart-axis-label" })).textContent = minD.toFixed(0);
}

function renderStateChart(attempts) {
  const svg = document.getElementById("state-chart");
  const width = 640, height = 220, padding = { left: 40, right: 20, top: 16, bottom: 40 };
  const { plotWidth, plotHeight } = chartFrame(svg, width, height, padding);

  const counts = Object.fromEntries(KNOWN_STATES.map((s) => [s, 0]));
  for (const a of attempts) if (counts[a.predicted_state] !== undefined) counts[a.predicted_state] += 1;
  const maxCount = Math.max(1, ...Object.values(counts));
  const barWidth = plotWidth / KNOWN_STATES.length;

  KNOWN_STATES.forEach((state, i) => {
    const count = counts[state];
    const barHeight = (count / maxCount) * plotHeight;
    const x = padding.left + i * barWidth + barWidth * 0.15;
    const y = padding.top + plotHeight - barHeight;
    svg.appendChild(svgEl("rect", {
      x, y, width: barWidth * 0.7, height: barHeight, fill: STATE_COLORS[state] || "#7a726a",
    }));
    svg.appendChild(svgEl("text", {
      x: x + barWidth * 0.35, y: padding.top + plotHeight + 16, class: "chart-axis-label", "text-anchor": "middle",
    })).textContent = state;
    svg.appendChild(svgEl("text", {
      x: x + barWidth * 0.35, y: y - 4, class: "chart-axis-label", "text-anchor": "middle",
    })).textContent = String(count);
  });
}

function renderCalculations(calculations) {
  const listEl = document.getElementById("calculations-list");
  listEl.innerHTML = "";
  for (const { title, formula, explanation } of calculations) {
    const li = document.createElement("li");
    const titleEl = document.createElement("strong");
    titleEl.textContent = title;
    const formulaEl = document.createElement("code");
    formulaEl.textContent = formula;
    const explanationEl = document.createElement("p");
    explanationEl.textContent = explanation;
    li.append(titleEl, formulaEl, explanationEl);
    listEl.appendChild(li);
  }
}

const appHeaderEl = document.querySelector(".app-header");
if (appHeaderEl) {
  window.addEventListener("scroll", () => {
    appHeaderEl.classList.toggle("scrolled", window.scrollY > 4);
  }, { passive: true });
}

loadSessions();
loadAggregate();
