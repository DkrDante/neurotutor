function formatTimestamp(unixSeconds) {
  return new Date(unixSeconds * 1000).toLocaleString();
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
    const response = await fetch(`/session/${sessionId}/calculations`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    renderDetail(sessionId, data);
    detailSection.hidden = false;
  } catch (err) {
    detailSection.hidden = true;
    document.getElementById("empty-state").hidden = false;
    document.getElementById("empty-state").textContent = `Could not load calculations: ${err.message}`;
  }
}

function renderDetail(sessionId, data) {
  document.getElementById("detail-session-id").textContent = sessionId;
  renderFormulaList(data.formulas);
  renderAggregateList(data.aggregate);
  renderTraceTable(data.attempts);
}

function renderFormulaList(formulas) {
  const listEl = document.getElementById("formula-list");
  listEl.innerHTML = "";
  for (const { title, formula, explanation } of formulas) {
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

function renderAggregateList(aggregate) {
  const listEl = document.getElementById("aggregate-list");
  listEl.innerHTML = "";
  if (aggregate.length === 0) {
    const li = document.createElement("li");
    li.className = "calc-trace-empty";
    li.textContent = "No attempts recorded in this session yet — nothing to calculate.";
    listEl.appendChild(li);
    return;
  }
  for (const { title, formula, substitution, result } of aggregate) {
    const li = document.createElement("li");
    const titleEl = document.createElement("strong");
    titleEl.textContent = title;
    const formulaEl = document.createElement("code");
    formulaEl.textContent = formula;
    const subEl = document.createElement("div");
    subEl.className = "calc-substitution";
    subEl.textContent = substitution;
    const resultEl = document.createElement("div");
    resultEl.className = "calc-result";
    resultEl.textContent = result;
    li.append(titleEl, formulaEl, subEl, resultEl);
    listEl.appendChild(li);
  }
}

function renderTraceTable(attempts) {
  const bodyEl = document.getElementById("trace-body");
  bodyEl.innerHTML = "";
  if (attempts.length === 0) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 7;
    cell.textContent = "No attempts recorded in this session yet.";
    cell.className = "calc-table-empty";
    row.appendChild(cell);
    bodyEl.appendChild(row);
    return;
  }

  for (const attempt of attempts) {
    const row = document.createElement("tr");

    const indexCell = document.createElement("td");
    indexCell.textContent = `${attempt.index + 1}`;

    const resultCell = document.createElement("td");
    const resultBadge = document.createElement("span");
    resultBadge.className = `calc-pill ${attempt.correct ? "calc-pill-correct" : "calc-pill-wrong"}`;
    resultBadge.textContent = attempt.correct ? "Correct" : "Wrong";
    resultCell.appendChild(resultBadge);

    const stateCell = document.createElement("td");
    stateCell.textContent = `${attempt.predicted_state} (${(attempt.confidence * 100).toFixed(0)}%)`;

    const evalCell = document.createElement("td");
    evalCell.textContent = `${attempt.eval_loss.toFixed(0)}cp`;

    const walkCell = document.createElement("td");
    walkCell.className = "calc-mono";
    walkCell.textContent =
      `${attempt.difficulty_before.toFixed(0)} ${attempt.actual_delta >= 0 ? "+" : ""}${attempt.actual_delta.toFixed(0)} = ${attempt.difficulty_after.toFixed(0)}`;

    const ruleCell = document.createElement("td");
    const recon = attempt.rule_based_reconstruction;
    const ruleBadge = document.createElement("span");
    ruleBadge.className = `calc-pill ${recon.matches_actual ? "calc-pill-match" : "calc-pill-diverge"}`;
    ruleBadge.textContent = recon.matches_actual ? "Rule-based" : "RL policy";
    ruleBadge.title =
      `Reconstructed: correctness ${recon.correctness_delta >= 0 ? "+" : ""}${recon.correctness_delta.toFixed(0)}` +
      (recon.confidence_trusted
        ? ` + state ${recon.state_delta_applied >= 0 ? "+" : ""}${recon.state_delta_applied.toFixed(0)} (confidence trusted)`
        : " (confidence below 0.4 threshold — state ignored)") +
      ` = ${recon.predicted_delta >= 0 ? "+" : ""}${recon.predicted_delta.toFixed(0)}, vs actual ${attempt.actual_delta >= 0 ? "+" : ""}${attempt.actual_delta.toFixed(0)}`;
    ruleCell.appendChild(ruleBadge);

    const accCell = document.createElement("td");
    accCell.textContent = `${(attempt.running_accuracy * 100).toFixed(0)}%`;

    row.append(indexCell, resultCell, stateCell, evalCell, walkCell, ruleCell, accCell);
    bodyEl.appendChild(row);
  }
}

loadSessions();
