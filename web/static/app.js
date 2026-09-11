// Solid glyphs only (both colors use the "black" unicode variants) — see style.css's
// .piece-white / .piece-black for how the two are told apart visually. Unicode's
// separate "white" glyphs render as hollow outlines in most fonts, which doesn't
// take a fill color well; using one solid glyph set per piece type is more reliable.
const PIECE_GLYPH = { k: "♚", q: "♛", r: "♜", b: "♝", n: "♞", p: "♟" };

const STATE_HINT_MESSAGES = {
  Overloaded: "You look overloaded — slow down and re-scan the whole board before moving.",
  Confused: "Look for checks, captures, and threats one at a time.",
  Fatigued: "Fatigue detected — a short breather might help before the next puzzle.",
};
const DEFAULT_HINT_MESSAGE = "Take your time and re-check the position.";
const KNOWN_STATES = ["Focused", "Overloaded", "Confused", "Fatigued", "Engaged"];

function parseFen(fen) {
  const board = [];
  const rows = fen.split(" ")[0].split("/");
  for (const row of rows) {
    const cells = [];
    for (const ch of row) {
      if (/\d/.test(ch)) {
        for (let i = 0; i < parseInt(ch, 10); i++) cells.push(null);
      } else {
        cells.push(ch);
      }
    }
    board.push(cells);
  }
  return board; // board[0] = rank 8 ... board[7] = rank 1
}

function squareName(rankIdx, fileIdx) {
  const files = "abcdefgh";
  return `${files[fileIdx]}${8 - rankIdx}`;
}

function squareToIndices(square) {
  const files = "abcdefgh";
  const fileIdx = files.indexOf(square[0]);
  const rankIdx = 8 - parseInt(square[1], 10);
  return [rankIdx, fileIdx];
}

let selectedSquare = null;
let legalTargets = [];
let attemptStartMs = null;
let ws = null;
let sessionId = null;
let lastServerError = null;
let currentAttemptToken = null;
let boardLocked = false;
let currentFen = null;
let currentBoard = null;
let boardFlipped = false;
let lastMove = null; // {from, to} of the most recently submitted move, for highlighting
let solverColor = "white"; // which side the human plays THIS puzzle — puzzles aren't all White-to-move

function pieceAt(square) {
  if (!currentBoard) return null;
  const [rankIdx, fileIdx] = squareToIndices(square);
  return currentBoard[rankIdx][fileIdx];
}

function isOwnPiece(piece) {
  if (!piece) return false;
  const isUppercase = piece === piece.toUpperCase();
  return solverColor === "white" ? isUppercase : !isUppercase;
}

function clearSelection() {
  selectedSquare = null;
  legalTargets = [];
}

function drawBoard() {
  currentBoard = parseFen(currentFen);
  const boardEl = document.getElementById("board");
  boardEl.innerHTML = "";
  for (let displayRow = 0; displayRow < 8; displayRow++) {
    for (let displayCol = 0; displayCol < 8; displayCol++) {
      const r = boardFlipped ? 7 - displayRow : displayRow;
      const f = boardFlipped ? 7 - displayCol : displayCol;
      const square = document.createElement("div");
      const name = squareName(r, f);
      const classes = ["square", (r + f) % 2 === 0 ? "light" : "dark"];
      if (name === selectedSquare) classes.push("selected");
      if (lastMove && (name === lastMove.from || name === lastMove.to)) classes.push("last-move");
      const piece = currentBoard[r][f];
      const isTarget = legalTargets.includes(name);
      if (isTarget) {
        classes.push("legal-target");
        if (piece) classes.push("capture");
      }
      square.className = classes.join(" ");
      square.dataset.square = name;
      if (piece) {
        const glyph = document.createElement("span");
        glyph.className = "piece " + (isOwnPiece(piece) ? "piece-white" : "piece-black");
        glyph.textContent = PIECE_GLYPH[piece.toLowerCase()];
        square.appendChild(glyph);
      }
      square.addEventListener("click", () => onSquareClick(name));
      boardEl.appendChild(square);
    }
  }
  drawCoordinateLabels();
}

function drawCoordinateLabels() {
  const files = boardFlipped ? "hgfedcba" : "abcdefgh";
  const ranks = boardFlipped ? "12345678" : "87654321";
  const filesEl = document.getElementById("files");
  const ranksEl = document.getElementById("ranks");
  filesEl.innerHTML = "";
  ranksEl.innerHTML = "";
  for (const ch of files) {
    const span = document.createElement("span");
    span.textContent = ch;
    filesEl.appendChild(span);
  }
  for (const ch of ranks) {
    const span = document.createElement("span");
    span.textContent = ch;
    ranksEl.appendChild(span);
  }
}

function renderBoard(fen) {
  currentFen = fen;
  lastMove = null;
  clearSelection();
  drawBoard();
  attemptStartMs = performance.now();
  boardLocked = false;
}

function toggleFlip() {
  boardFlipped = !boardFlipped;
  if (currentFen) drawBoard();
}

function setCallout(el, text, variant) {
  el.hidden = !text;
  el.className = "callout" + (variant ? ` callout-${variant}` : "");
  el.textContent = text || "";
}

function selectPiece(square) {
  selectedSquare = square;
  legalTargets = [];
  drawBoard();
  ws.send(JSON.stringify({ type: "hint", square }));
}

function onSquareClick(square) {
  if (boardLocked) return; // Waiting on the server (pacing delay / next puzzle).

  if (!selectedSquare) {
    const piece = pieceAt(square);
    if (isOwnPiece(piece)) selectPiece(square);
    return;
  }

  if (square === selectedSquare) {
    clearSelection();
    drawBoard();
    return;
  }

  if (legalTargets.includes(square)) {
    let moveUci = selectedSquare + square;
    const piece = pieceAt(selectedSquare);
    const destRank = square[1];
    const isPromotion =
      (piece === "P" && destRank === "8") || (piece === "p" && destRank === "1");
    if (isPromotion) {
      moveUci += "q"; // Auto-queen; no promotion-choice UI.
    }
    const timeToMove = (performance.now() - attemptStartMs) / 1000.0;
    setCallout(document.getElementById("feedback"), "", null);
    ws.send(JSON.stringify({
      move_uci: moveUci, time_to_move: timeToMove, attempt_token: currentAttemptToken,
    }));
    lastMove = { from: selectedSquare, to: square };
    clearSelection();
    boardLocked = true; // Locked until the next "puzzle" message arrives.
    drawBoard();
    return;
  }

  const otherPiece = pieceAt(square);
  if (isOwnPiece(otherPiece)) {
    selectPiece(square);
    return;
  }

  // Not a legal destination for the selected piece: ask the server why, without
  // scoring anything or losing the current selection/highlights.
  const attemptedMove = selectedSquare + square;
  ws.send(JSON.stringify({ type: "check_move", move_uci: attemptedMove }));
}

async function showSummary() {
  const summaryEl = document.getElementById("summary");
  if (!sessionId) {
    summaryEl.textContent = "No session yet.";
    return;
  }
  try {
    const response = await fetch(`/session/${sessionId}/summary`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const s = await response.json();
    summaryEl.textContent =
      `Attempts: ${s.num_attempts} · Accuracy: ${(s.accuracy_rate * 100).toFixed(0)}% · ` +
      `Avg latency: ${s.avg_latency.toFixed(3)}s · States: ${s.state_trend.join(", ") || "-"}`;
  } catch (err) {
    summaryEl.textContent = `Could not load summary: ${err.message}`;
  }
}

// ---------------------------------------------------------------------------
// Network Activity: a live visualization of the actual fusion model's forward
// pass, driven entirely by real values from server "update" messages'
// network_activity field (model.inference.StatePredictor.predict_with_internals) —
// nothing here is a canned animation. Diagram built once on first data (the GCN
// channel-adjacency weights are a fixed trained parameter, not per-prediction),
// then only node/edge visual attributes update afterward.
// ---------------------------------------------------------------------------

const NUM_EEG_NODES = 8;
const SVG_NS = "http://www.w3.org/2000/svg";
const NET = {
  eegCenter: { x: 90, y: 150 }, eegRadius: 60,
  eegEmbed: { x: 210, y: 150 },
  behaviorBars: { x0: 270, y: 90, w: 14, gap: 10, maxHeight: 50, baseline: 150 },
  behaviorEmbed: { x: 340, y: 150 },
  fusion: { x: 430, y: 150 },
  lstm: { x: 510, y: 150 },
  outputs: { x: 610, y0: 40, spacing: 55 },
};
const VIEWBOX_WIDTH = 800; // must stay wide enough that output-node labels (x + ~80px) don't get clipped

let networkBuilt = false;

function svgEl(tag, attrs) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
}

function eegNodePosition(i) {
  const angle = (-90 + i * (360 / NUM_EEG_NODES)) * (Math.PI / 180);
  return {
    x: NET.eegCenter.x + NET.eegRadius * Math.cos(angle),
    y: NET.eegCenter.y + NET.eegRadius * Math.sin(angle),
  };
}

function buildNetworkDiagram(activity) {
  const svg = document.getElementById("network-svg");
  svg.setAttribute("viewBox", `0 0 ${VIEWBOX_WIDTH} 300`);
  svg.innerHTML = "";

  const edgesGroup = svgEl("g", { id: "gcn-edges" });
  svg.appendChild(edgesGroup);
  // The adjacency matrix is a fixed, learned parameter (same every prediction) —
  // draw it once now. Only edges clearly above a uniform baseline are shown, so
  // the diagram reads as "what the model actually learned to connect" rather
  // than a dense, unreadable mesh of all 28 channel pairs.
  const threshold = 1 / NUM_EEG_NODES;
  const adjacency = activity.gcn_adjacency;
  for (let i = 0; i < NUM_EEG_NODES; i++) {
    for (let j = i + 1; j < NUM_EEG_NODES; j++) {
      const weight = Math.max(adjacency[i][j], adjacency[j][i]);
      if (weight <= threshold) continue;
      const a = eegNodePosition(i), b = eegNodePosition(j);
      edgesGroup.appendChild(svgEl("line", {
        class: "net-edge", x1: a.x, y1: a.y, x2: b.x, y2: b.y,
        "stroke-width": 1 + weight * 3, "stroke-opacity": Math.min(1, weight * 2.5),
      }));
    }
  }

  const nodesGroup = svgEl("g", { id: "gcn-nodes" });
  svg.appendChild(nodesGroup);
  for (let i = 0; i < NUM_EEG_NODES; i++) {
    const pos = eegNodePosition(i);
    nodesGroup.appendChild(svgEl("circle", {
      id: `gcn-node-${i}`, class: "net-node net-node-eeg", cx: pos.x, cy: pos.y, r: 10,
    }));
  }
  svg.appendChild(svgEl("text", {
    class: "net-label", x: NET.eegCenter.x, y: NET.eegCenter.y - NET.eegRadius - 14,
  })).textContent = "EEG channels";

  const flowEdge = (from, to) => svg.appendChild(svgEl("line", {
    class: "net-edge net-edge-flow", x1: from.x, y1: from.y, x2: to.x, y2: to.y,
    "stroke-width": 1.5, "stroke-opacity": 0.5,
  }));
  flowEdge({ x: NET.eegCenter.x + NET.eegRadius, y: NET.eegCenter.y }, NET.eegEmbed);
  flowEdge(NET.eegEmbed, NET.fusion);
  flowEdge(NET.behaviorEmbed, NET.fusion);
  flowEdge(NET.fusion, NET.lstm);

  svg.appendChild(svgEl("circle", {
    id: "eeg-embed-node", class: "net-node net-node-embed", cx: NET.eegEmbed.x, cy: NET.eegEmbed.y, r: 18,
  }));
  svg.appendChild(svgEl("text", { class: "net-label", x: NET.eegEmbed.x, y: NET.eegEmbed.y + 32 })).textContent = "EEG embed";

  const barsGroup = svgEl("g", { id: "behavior-bars" });
  svg.appendChild(barsGroup);
  // "correct / time-to-move / eval-loss / puzzle-rating" (the same 4 features
  // data_gen.generate_dataset.behavior_to_vector produces) — a hover title per bar
  // instead of always-on text labels, which collided at this width.
  const behaviorLabels = ["correct", "time to move", "eval loss", "puzzle rating"];
  const barsSpan = 4 * NET.behaviorBars.w + 3 * NET.behaviorBars.gap;
  for (let i = 0; i < 4; i++) {
    const x = NET.behaviorBars.x0 + i * (NET.behaviorBars.w + NET.behaviorBars.gap);
    const bar = svgEl("rect", {
      id: `behavior-bar-${i}`, class: "net-behavior-bar",
      x, y: NET.behaviorBars.baseline, width: NET.behaviorBars.w, height: 0,
    });
    bar.appendChild(svgEl("title", {})).textContent = behaviorLabels[i];
    barsGroup.appendChild(bar);
  }
  flowEdge({ x: NET.behaviorBars.x0 + barsSpan / 2, y: NET.behaviorBars.y }, NET.behaviorEmbed);
  svg.appendChild(svgEl("circle", {
    id: "behavior-embed-node", class: "net-node net-node-embed", cx: NET.behaviorEmbed.x, cy: NET.behaviorEmbed.y, r: 18,
  }));
  svg.appendChild(svgEl("text", { class: "net-label", x: NET.behaviorEmbed.x, y: NET.behaviorEmbed.y + 32 })).textContent = "Behavior";

  svg.appendChild(svgEl("circle", {
    id: "fusion-node", class: "net-node net-node-fusion", cx: NET.fusion.x, cy: NET.fusion.y, r: 20,
  }));
  svg.appendChild(svgEl("text", { class: "net-label", x: NET.fusion.x, y: NET.fusion.y + 34 })).textContent = "Fusion";

  svg.appendChild(svgEl("circle", {
    id: "lstm-node", class: "net-node net-node-lstm", cx: NET.lstm.x, cy: NET.lstm.y, r: 20,
  }));
  svg.appendChild(svgEl("text", { class: "net-label", x: NET.lstm.x, y: NET.lstm.y + 34 })).textContent = "LSTM";

  KNOWN_STATES.forEach((state, i) => {
    const y = NET.outputs.y0 + i * NET.outputs.spacing;
    flowEdge(NET.lstm, { x: NET.outputs.x, y });
    svg.appendChild(svgEl("circle", {
      id: `output-node-${state}`, class: "net-node net-node-output", cx: NET.outputs.x, cy: y, r: 12,
    }));
    svg.appendChild(svgEl("text", {
      // +30, not +20: the circle's radius can grow up to 20 (see updateNetworkDiagram's
      // 8 + p*12 for p=1), which would otherwise sit under the label at a high-confidence
      // prediction. Anchor is set via inline `style`, not the `text-anchor` attribute:
      // the .net-label CSS class's own text-anchor:middle otherwise wins the cascade over
      // a same-specificity presentation attribute, silently re-centering the label on x.
      class: "net-label", x: NET.outputs.x + 30, y: y + 4, style: "text-anchor: start",
    })).textContent = state;
  });

  networkBuilt = true;
}

// Real, unbounded magnitudes (vector norms) squashed into a [0,1]-ish display
// range for opacity/sizing — the scale factor is chosen for visual range, not
// a calibrated unit; relative differences between predictions are what matter.
function squash(value, scale) {
  return Math.tanh(Math.abs(value) / scale);
}

function updateNetworkDiagram(activity) {
  if (!activity) return;
  const svg = document.getElementById("network-svg");
  if (!networkBuilt) buildNetworkDiagram(activity);

  for (let i = 0; i < NUM_EEG_NODES; i++) {
    const node = document.getElementById(`gcn-node-${i}`);
    const level = squash(activity.gcn2_node_activity[i], 2);
    node.setAttribute("r", 7 + level * 8);
    node.setAttribute("fill-opacity", 0.35 + level * 0.65);
  }

  const eegLevel = squash(activity.eeg_embedding_norm, 2);
  const eegEmbedNode = document.getElementById("eeg-embed-node");
  eegEmbedNode.setAttribute("fill-opacity", 0.35 + eegLevel * 0.65);

  activity.behavior_activity.forEach((value, i) => {
    const bar = document.getElementById(`behavior-bar-${i}`);
    const clamped = Math.max(0, Math.min(1, value));
    const height = clamped * NET.behaviorBars.maxHeight;
    bar.setAttribute("y", NET.behaviorBars.baseline - height);
    bar.setAttribute("height", height);
  });
  const behaviorLevel = squash(activity.behavior_embedding_norm, 2);
  document.getElementById("behavior-embed-node").setAttribute("fill-opacity", 0.35 + behaviorLevel * 0.65);

  const lstmLevel = squash(activity.lstm_hidden_norm, 3);
  document.getElementById("lstm-node").setAttribute("fill-opacity", 0.35 + lstmLevel * 0.65);
  document.getElementById("fusion-node").setAttribute("fill-opacity", 0.35 + (eegLevel + behaviorLevel) / 2 * 0.65);

  KNOWN_STATES.forEach((state) => {
    const node = document.getElementById(`output-node-${state}`);
    const p = activity.probs[state] || 0;
    node.setAttribute("r", 8 + p * 12); // max r=20, well clear of the label at +30 (see buildNetworkDiagram)
    node.setAttribute("fill-opacity", 0.3 + p * 0.7);
    node.classList.toggle("predicted", state === activity.state);
  });

  svg.classList.remove("pulse");
  void svg.offsetWidth; // force reflow so re-adding the class restarts the CSS animation
  svg.classList.add("pulse");
}

// ---------------------------------------------------------------------------
// Live EEG waveform: a real scrolling multi-channel strip chart fed by the
// server's background stream_eeg() task — the actual streamed signal, not a
// decorative loop.
// ---------------------------------------------------------------------------

const EEG_WINDOW = 160; // samples retained per channel (~1.25s at the simulator's 128Hz)
const eegBuffers = Array.from({ length: NUM_EEG_NODES }, () => []);
const EEG_CHANNEL_COLORS = ["#4a7c2f", "#b58863", "#3f7fbf", "#b91c1c", "#c2410c", "#6d28d9", "#0f766e", "#a16207"];

function onEegChunk(msg) {
  for (const row of msg.samples) {
    for (let ch = 0; ch < NUM_EEG_NODES; ch++) {
      const buf = eegBuffers[ch];
      buf.push(row[ch]);
      if (buf.length > EEG_WINDOW) buf.shift();
    }
  }
  drawEegWaveform();
}

function drawEegWaveform() {
  const canvas = document.getElementById("eeg-canvas");
  const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  const bandHeight = h / NUM_EEG_NODES;

  for (let ch = 0; ch < NUM_EEG_NODES; ch++) {
    const buf = eegBuffers[ch];
    if (buf.length < 2) continue;
    const baseline = bandHeight * ch + bandHeight / 2;
    ctx.beginPath();
    ctx.strokeStyle = EEG_CHANNEL_COLORS[ch];
    ctx.lineWidth = 1;
    for (let i = 0; i < buf.length; i++) {
      const x = (i / (EEG_WINDOW - 1)) * w;
      const y = baseline - buf[i] * (bandHeight * 0.4);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
}

function setState(stateName) {
  const badge = document.getElementById("state-badge");
  badge.textContent = stateName;
  badge.className = "badge " + (KNOWN_STATES.includes(stateName) ? `badge-${stateName}` : "badge-neutral");
}

function setConfidence(confidence) {
  document.getElementById("confidence-value").textContent = confidence.toFixed(2);
  document.getElementById("confidence-fill").style.width = `${Math.round(confidence * 100)}%`;
}

function connect() {
  ws = new WebSocket(`ws://${window.location.host}/ws/session`);
  const feedbackEl = document.getElementById("feedback");
  const hintBox = document.getElementById("hint-box");

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.session_id) sessionId = msg.session_id;

    if (msg.type === "eeg_chunk") {
      onEegChunk(msg);
      return; // high-frequency; skip the session_id bookkeeping below entirely
    }

    if (msg.type === "puzzle") {
      currentAttemptToken = msg.attempt_token;
      if (msg.solver_color && msg.solver_color !== solverColor) {
        solverColor = msg.solver_color;
        boardFlipped = solverColor === "black"; // orient the board to the human's side
      }
      renderBoard(msg.fen);
      setCallout(feedbackEl, "", null);
    } else if (msg.type === "hint") {
      if (msg.square === selectedSquare) {
        legalTargets = msg.targets;
        drawBoard();
      }
    } else if (msg.type === "invalid_move") {
      legalTargets = msg.targets || legalTargets;
      setCallout(feedbackEl, `Invalid move — ${msg.reason}`, "invalid");
      boardLocked = false; // A rejected move must not leave the board stuck.
      drawBoard();
    } else if (msg.type === "update") {
      setState(msg.predicted_state);
      setConfidence(msg.confidence);
      document.getElementById("difficulty").textContent = msg.difficulty.toFixed(0);
      updateNetworkDiagram(msg.network_activity);
      if (msg.action.show_hint) {
        hintBox.hidden = false;
        document.getElementById("hint-text").textContent =
          STATE_HINT_MESSAGES[msg.predicted_state] || DEFAULT_HINT_MESSAGE;
      } else {
        hintBox.hidden = true;
      }
      if (msg.status === "solved") {
        setCallout(feedbackEl, "Correct! Puzzle solved — next one incoming.", "correct");
      } else if (msg.status === "continue") {
        setCallout(feedbackEl, `Correct! Opponent plays ${msg.opponent_move} — keep going.`, "continue");
      } else {
        setCallout(feedbackEl, "There's a better move — try again.", "incorrect");
      }
    } else if (msg.type === "error") {
      lastServerError = msg.message;
      setCallout(feedbackEl, msg.message, "error");
      boardLocked = false;
    }
  };
  ws.onerror = () => {
    setCallout(feedbackEl, "Connection error.", "error");
  };
  ws.onclose = () => {
    setCallout(
      feedbackEl,
      lastServerError ? `${lastServerError} (connection closed)` : "Connection closed.",
      "error",
    );
  };
}

document.getElementById("summary-button").addEventListener("click", showSummary);
document.getElementById("flip-button").addEventListener("click", toggleFlip);
connect();
