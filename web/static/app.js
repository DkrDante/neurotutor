// Solid glyphs only (both colors use the "black" unicode variants) — see style.css's
// .piece-white / .piece-black for how the two are told apart visually. Unicode's
// separate "white" glyphs render as hollow outlines in most fonts, which doesn't
// take a fill color well; using one solid glyph set per piece type is more reliable.
const PIECE_GLYPH = { k: "♚", q: "♛", r: "♜", b: "♝", n: "♞", p: "♟" };
const PIECE_NAME_FOR_A11Y = { k: "king", q: "queen", r: "rook", b: "bishop", n: "knight", p: "pawn" };

function describeSquareForA11y(square, piece) {
  if (!piece) return `${square}, empty`;
  const color = piece === piece.toUpperCase() ? "white" : "black";
  return `${square}, ${color} ${PIECE_NAME_FOR_A11Y[piece.toLowerCase()]}`;
}

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
let reconnectAttempts = 0;
let reconnectTimer = null;
let loggedInUsername = null; // null = signed out, using localStorage-only gamification state
let sessionId = null;
let lastServerError = null;
let currentAttemptToken = null;
let boardLocked = false;
let currentFen = null;
let currentBoard = null;
let boardFlipped = false;
let lastMove = null; // {from, to} of the most recently submitted move, for highlighting
let solverColor = "white"; // which side the human plays THIS puzzle — puzzles aren't all White-to-move
let hintSquare = null; // from-square of a requested puzzle_hint, cleared on the next puzzle/retry
let pendingOptimisticMove = null; // {from, to, movingPiece, capturedPiece} — undone if the server says "retry"
let boardFocusIndex = 0; // roving-tabindex position among the 64 squares, in current display order

// --square-size (style.css) shrinks below 820px viewport width so the board
// fits on a phone screen — measuring the actually-rendered square rather than
// assuming a fixed pixel size keeps the slide-in animation's offset correct
// at every breakpoint instead of just the desktop default.
function currentSquareSizePx() {
  const sample = document.querySelector("#board .square");
  return sample ? sample.getBoundingClientRect().width : 76;
}

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

// Display-space (screen row/col, accounting for board orientation) position of a
// square — used only for the slide animation's pixel-offset math below.
function displayPosition(square) {
  const [r, f] = squareToIndices(square);
  return { row: boardFlipped ? 7 - r : r, col: boardFlipped ? 7 - f : f };
}

// Re-renders the board from whatever `currentBoard` currently holds, WITHOUT
// re-parsing `currentFen` — lets an optimistic local move (applyMoveLocally)
// show up immediately, ahead of the server's authoritative confirmation.
function renderCurrentBoard() {
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
      if (name === hintSquare) classes.push("hint-square");
      const piece = currentBoard[r][f];
      const isTarget = legalTargets.includes(name);
      if (isTarget) {
        classes.push("legal-target");
        if (piece) classes.push("capture");
      }
      square.className = classes.join(" ");
      square.dataset.square = name;
      // Roving tabindex: only one square is ever a Tab stop (boardFocusIndex),
      // so a sighted keyboard user tabbing past the board isn't forced to
      // cycle through all 64 squares to reach the controls after it. Arrow
      // keys (see the delegated #board keydown listener) move focus within.
      const flatIndex = displayRow * 8 + displayCol;
      square.tabIndex = flatIndex === boardFocusIndex ? 0 : -1;
      square.setAttribute("role", "button");
      square.setAttribute("aria-label", describeSquareForA11y(name, piece));
      if (name === selectedSquare) square.setAttribute("aria-pressed", "true");
      if (piece) {
        const glyph = document.createElement("span");
        glyph.className = "piece " + (isOwnPiece(piece) ? "piece-white" : "piece-black");
        glyph.textContent = PIECE_GLYPH[piece.toLowerCase()];
        square.appendChild(glyph);
      }
      square.addEventListener("click", () => {
        boardFocusIndex = flatIndex;
        onSquareClick(name);
      });
      boardEl.appendChild(square);
    }
  }
  drawCoordinateLabels();
  renderCapturedTrays();
}

function drawBoard() {
  currentBoard = parseFen(currentFen);
  renderCurrentBoard();
}

// Slides the piece now sitting on `square` in from `fromSquare`'s old screen
// position — call right after a render so the piece just landed on `square`.
function animateSlideIn(square, fromSquare) {
  const el = document.querySelector(`.square[data-square="${square}"] .piece`);
  if (!el) return;
  const from = displayPosition(fromSquare);
  const to = displayPosition(square);
  const squareSize = currentSquareSizePx();
  const dx = (from.col - to.col) * squareSize;
  const dy = (from.row - to.row) * squareSize;
  el.style.transition = "none";
  el.style.transform = `translate(${dx}px, ${dy}px)`;
  void el.offsetWidth; // force reflow so the transition below actually animates
  el.style.transition = "transform 0.22s ease";
  el.style.transform = "translate(0, 0)";
}

// Mutates currentBoard directly (no re-parse) to reflect a move immediately,
// ahead of the server's confirmation — returns what was there before, so a
// wrong-move revert (see the "retry" handling in ws.onmessage) can restore it
// exactly.
function applyMoveLocally(fromSquare, toSquare, promoType) {
  const [fr, ff] = squareToIndices(fromSquare);
  const [tr, tf] = squareToIndices(toSquare);
  const movingPiece = currentBoard[fr][ff];
  const capturedPiece = currentBoard[tr][tf];
  currentBoard[fr][ff] = null;
  let finalPiece = movingPiece;
  if (promoType && movingPiece) {
    finalPiece = movingPiece === movingPiece.toUpperCase() ? promoType.toUpperCase() : promoType.toLowerCase();
  }
  currentBoard[tr][tf] = finalPiece;

  // Castling is encoded in UCI as just the king's own from/to (e.g. "e1g1") —
  // the server-side python-chess board relocates the rook too, so the local
  // optimistic copy needs to do the same or it'd sit wrong until the next
  // authoritative redraw corrects it.
  if (movingPiece && movingPiece.toLowerCase() === "k" && fr === tr && Math.abs(tf - ff) === 2) {
    const kingside = tf > ff;
    const rookFromFile = kingside ? 7 : 0;
    const rookToFile = kingside ? tf - 1 : tf + 1;
    currentBoard[fr][rookToFile] = currentBoard[fr][rookFromFile];
    currentBoard[fr][rookFromFile] = null;
  }
  return { movingPiece, capturedPiece };
}

// UCI moves are 4 chars ("e2e4") or 5 with a promotion letter ("e7e8q") —
// this is the single place that parses that shape for the optimistic-move path.
function applyUciMoveLocally(uci) {
  const from = uci.slice(0, 2);
  const to = uci.slice(2, 4);
  const promoType = uci.length === 5 ? uci[4] : null;
  const { movingPiece, capturedPiece } = applyMoveLocally(from, to, promoType);
  return { from, to, movingPiece, capturedPiece };
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
  clearSelection();
  drawBoard();
  attemptStartMs = performance.now();
  boardLocked = false;
}

function requestHint() {
  if (!ws || ws.readyState !== WebSocket.OPEN || boardLocked) return;
  hintUsedThisPuzzle = true;
  ws.send(JSON.stringify({ type: "puzzle_hint" }));
}

function toggleFlip() {
  boardFlipped = !boardFlipped;
  // Re-layout whatever currentBoard already holds rather than re-parsing
  // currentFen — orientation never depends on the authoritative FEN, and this
  // way flipping mid-move can't discard an in-flight optimistic move.
  if (currentBoard) renderCurrentBoard();
}

function setCallout(el, text, variant) {
  el.hidden = !text;
  el.className = "callout" + (variant ? ` callout-${variant}` : "");
  el.textContent = text || "";
  if (text) bounceMascot();
}

// Nudges the mascot avatar next to the feedback/hint speech bubble whenever
// it has something new to "say" — the same restart-a-CSS-animation trick as
// popCallout() above, just applied to the avatar instead of the message box.
function bounceMascot() {
  const mascot = document.getElementById("mascot-avatar");
  if (!mascot) return;
  mascot.classList.remove("talking");
  void mascot.offsetWidth;
  mascot.classList.add("talking");
}

// Like setCallout, but for a wrong move that came back with a move quality
// score/label (server.py's chess_task.move_explainer.score_move) and a
// factual description of what the move did — shown as a small colored
// score badge above the (already tier-adjusted-for-difficulty) explanation,
// instead of just plain text.
function setCalloutWithMoveQuality(el, text, variant, moveScore, moveLabel, moveDescription, proactiveIntervention) {
  el.hidden = false;
  el.className = "callout" + (variant ? ` callout-${variant}` : "");
  el.innerHTML = "";
  if (moveLabel != null && moveScore != null) {
    const badge = document.createElement("span");
    badge.className = `move-quality-badge move-quality-${moveLabel.toLowerCase()}`;
    badge.textContent = `${moveLabel} · ${moveScore}/100`;
    el.appendChild(badge);
  }
  // The proactive loop (adaptive.proactive) acted on a TREND across recent
  // attempts — not just this one — so it gets its own line, distinct from
  // the ordinary per-move feedback below it.
  if (proactiveIntervention) {
    const note = document.createElement("span");
    note.className = "callout-proactive-note";
    note.textContent = "Noticed you might be finding this tough, so we've eased the difficulty and revealed the solution.";
    el.appendChild(note);
  }
  const body = document.createElement("span");
  body.className = "callout-body";
  body.textContent = [moveDescription, text].filter(Boolean).join(" ");
  el.appendChild(body);
  bounceMascot();
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
    lastSubmittedTimeToMove = timeToMove;
    setCallout(document.getElementById("feedback"), "", null);
    hintSquare = null;
    document.getElementById("puzzle-hint-box").hidden = true;
    ws.send(JSON.stringify({
      move_uci: moveUci, time_to_move: timeToMove, attempt_token: currentAttemptToken,
    }));

    const { from, to, movingPiece, capturedPiece } = applyUciMoveLocally(moveUci);
    pendingOptimisticMove = { from, to, movingPiece, capturedPiece };
    lastMove = { from, to };
    clearSelection();
    boardLocked = true; // Locked until the server confirms or reverts this move.
    renderCurrentBoard();
    animateSlideIn(to, from);
    playSound(capturedPiece ? "capture" : "move");
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
  summaryEl.textContent = "Loading…";
  try {
    // /coach is fetched alongside /summary rather than replacing it — the raw
    // attempts/accuracy numbers still matter, coaching is additive context on
    // top of them, not instead of them.
    const [summaryRes, coachRes] = await Promise.all([
      fetch(`/session/${sessionId}/summary`),
      fetch(`/session/${sessionId}/coach`),
    ]);
    if (!summaryRes.ok) throw new Error(`HTTP ${summaryRes.status}`);
    const s = await summaryRes.json();
    const coach = coachRes.ok ? await coachRes.json() : null;

    summaryEl.innerHTML = "";
    const statsLine = document.createElement("p");
    statsLine.className = "summary-stats-line";
    statsLine.textContent =
      `${s.num_attempts} attempts · ${(s.accuracy_rate * 100).toFixed(0)}% accuracy · ` +
      `${s.avg_latency.toFixed(3)}s avg sense-to-adapt latency`;
    summaryEl.appendChild(statsLine);

    if (coach) {
      if (coach.narrative) {
        // Claude's take on the same real statistics below — only present when
        // ANTHROPIC_API_KEY is configured server-side (coaching.narrator.ai_narrative).
        const narrativeEl = document.createElement("p");
        narrativeEl.className = "summary-narrative";
        narrativeEl.textContent = coach.narrative;
        summaryEl.appendChild(narrativeEl);
      }
      if (coach.tips && coach.tips.length) {
        const tipsHeading = document.createElement("p");
        tipsHeading.className = "mini-label";
        tipsHeading.textContent = "Where to improve";
        summaryEl.appendChild(tipsHeading);
        const tipsList = document.createElement("ul");
        tipsList.className = "coaching-tips";
        for (const tip of coach.tips) {
          const li = document.createElement("li");
          li.textContent = tip;
          tipsList.appendChild(li);
        }
        summaryEl.appendChild(tipsList);
      }
      // Closes the loop from "here's your weak spot" to actually practicing
      // it: reconnects with ?target_min=/?target_max= so the NEXT session's
      // puzzles are drawn specifically from that rating band, instead of the
      // ordinary adaptive loop just moving one difficulty number up or down.
      if (coach.targeted_practice) {
        const targetedBtn = document.createElement("button");
        targetedBtn.type = "button";
        targetedBtn.className = "btn btn-primary summary-targeted-btn";
        targetedBtn.textContent = `Practice your weak spot (${coach.targeted_practice.label}) →`;
        targetedBtn.addEventListener("click", () => {
          startTargetedPractice(coach.targeted_practice.rating_min, coach.targeted_practice.rating_max);
        });
        summaryEl.appendChild(targetedBtn);
      }
    }

    const link = document.createElement("a");
    link.href = "/analytics";
    link.className = "nav-link summary-analytics-link";
    link.textContent = "Full analytics & charts →";
    summaryEl.appendChild(link);
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

// Real, unbounded magnitudes (vector norms/activations) squashed into a
// [0,1]-ish display range for opacity/sizing — the scale factor is chosen for
// visual range, not a calibrated unit; relative differences are what matter.
function squash(value, scale) {
  return Math.tanh(Math.abs(value) / scale);
}

function svgEl(tag, attrs) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
}

// Every node gets a native <title> child so hovering it shows a tooltip (no
// extra CSS/JS needed for the tooltip itself — the browser renders it).
function withTitle(el, text) {
  el.appendChild(svgEl("title", {})).textContent = text;
  return el;
}

// ---------------------------------------------------------------------------
// Click-to-inspect: every node and weighted connection below is bound to a
// getDetails() closure that reads live values off `lastNetworkActivity` (the
// most recent real "update" message) at CLICK time, not at build time — so
// the numbers shown are always this move's, not whatever they were when the
// diagram was first built. Weights themselves ARE fixed once training is
// done (that's what "trained" means — nothing here claims otherwise); what
// changes every move is which real activations are flowing through them,
// and that's exactly what re-clicking the same node after a new move shows.
// ---------------------------------------------------------------------------

let lastNetworkActivity = null;

function fmtSigned(n, digits = 4) {
  return `${n >= 0 ? "+" : ""}${n.toFixed(digits)}`;
}

function showInspector(details) {
  document.getElementById("net-inspector-kicker").textContent = details.kicker || "";
  document.getElementById("net-inspector-title").textContent = details.title;
  document.getElementById("net-inspector-subtitle").textContent = details.subtitle || "";
  const listEl = document.getElementById("net-inspector-rows");
  listEl.innerHTML = "";
  for (const row of details.rows || []) {
    const li = document.createElement("li");
    const label = document.createElement("span");
    label.className = "net-inspector-row-label";
    label.textContent = row.label;
    const value = document.createElement("span");
    value.className = "net-inspector-row-value";
    value.textContent = row.value;
    li.append(label, value);
    listEl.appendChild(li);
  }
  document.getElementById("net-inspector").hidden = false;
}

// Click/keyboard-activatable — every node and weighted edge in the three
// panels gets this, so the whole diagram is explorable by keyboard too.
function bindInspectable(el, getDetails) {
  el.classList.add("net-inspectable");
  el.setAttribute("tabindex", "0");
  el.setAttribute("role", "button");
  const activate = (event) => { event.preventDefault(); showInspector(getDetails()); };
  el.addEventListener("click", activate);
  el.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") activate(event);
  });
}

// Formats a real incoming/outgoing weight list for the inspector, ranked by
// |weight| and capped so a 32- or 256-connection layer doesn't dump an
// unreadable wall of numbers — but always states the true total so nothing
// is silently hidden, just not all listed.
function weightRows(weights, sourceLabels, maxShown = 8) {
  const withLabels = weights.map((w, i) => ({ label: sourceLabels[i], weight: w }));
  const sorted = [...withLabels].sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight));
  const rows = sorted.slice(0, maxShown).map((c) => ({ label: c.label, value: fmtSigned(c.weight) }));
  if (sorted.length > maxShown) rows.push({ label: `…and ${sorted.length - maxShown} more (ranked by |weight|)`, value: "" });
  return rows;
}

document.getElementById("net-inspector-close").addEventListener("click", () => {
  document.getElementById("net-inspector").hidden = true;
});

// A small dot that glides along a flow edge (native SVG <animateMotion>) —
// but only when triggerFlowParticles() fires it (see updateNetworkDiagram),
// once per real move. Sitting idle and looping forever regardless of whether
// anything was actually happening read as decorative, not "the network
// reacting to your input" — begin="indefinite" plus a manual .beginElement()
// call ties every glide to an actual attempt.
let flowParticleAnimations = [];

function addFlowParticle(svg, from, to) {
  // The circle's own position is the path's start point; the path itself is
  // expressed relative to that (starting at 0,0) so the two don't double up —
  // animateMotion translates an element BY the path's coordinates on top of
  // wherever it already is, it doesn't replace the element's position outright.
  const particle = svgEl("circle", { r: 2.5, class: "flow-particle", cx: from.x, cy: from.y });
  const motion = document.createElementNS(SVG_NS, "animateMotion");
  motion.setAttribute("dur", "0.7s");
  motion.setAttribute("begin", "indefinite");
  motion.setAttribute("path", `M0,0 L${to.x - from.x},${to.y - from.y}`);
  particle.appendChild(motion);
  svg.appendChild(particle);
  flowParticleAnimations.push(motion);
}

function triggerFlowParticles() {
  for (const motion of flowParticleAnimations) {
    try {
      motion.beginElement();
    } catch (err) {
      // SMIL's beginElement() isn't supported in every browser — the dots just
      // stay put at their start point there, which is a safe, static fallback.
    }
  }
}

function flowEdgeIn(svg, from, to) {
  const line = svg.appendChild(svgEl("line", {
    class: "net-edge net-edge-flow", x1: from.x, y1: from.y, x2: to.x, y2: to.y,
    "stroke-width": 1.5, "stroke-opacity": 0.5,
  }));
  addFlowParticle(svg, from, to);
  return line;
}

// For the dense weight-edge meshes (too many edges to animate all of them
// without turning into visual noise), only the strongest real connections get
// a travelling particle — still real data (the top |weight| edges), just a
// bounded number of them.
function addTopParticles(svg, edges, count) {
  const strongest = [...edges].sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight)).slice(0, count);
  for (const e of strongest) {
    addFlowParticle(svg, { x: e.el.getAttribute("x1"), y: e.el.getAttribute("y1") },
      { x: e.el.getAttribute("x2"), y: e.el.getAttribute("y2") });
  }
}

let panelsBuilt = false;

// ---------------------------------------------------------------------------
// Panel 1: the behavior branch as an ACTUAL layered neural network (no EEG
// involved) — input layer (the 4 raw behavior features) fully connected to a
// 16-unit hidden layer, fully connected to the 16-dim output embedding.
//
// Each edge's base intensity is the real learned |weight| from
// model.encoders.BehaviorMLPEncoder (model/inference.py's behavior_w1/w2 — a
// fixed, trained parameter, so this component of the line never changes).
// On top of that, every move re-weights each edge by how active its SOURCE
// neuron actually is right now (a real per-forward-pass activation), so the
// diagram visibly "lights up" a different subset of connections after each
// move instead of sitting static between weight-only lines.
// ---------------------------------------------------------------------------

const BEHAVIOR_NN = {
  input: { x: 30, count: 4, y0: 90, spacing: 40 },
  hidden: { x: 125, count: 16, y0: 10, spacing: 18 },
  output: { x: 220, count: 16, y0: 10, spacing: 18 },
};
const BEHAVIOR_LABELS = ["correct", "time to move", "eval loss", "puzzle rating"];

function layerNodeY(layer, i) {
  return layer.y0 + i * layer.spacing;
}

let behaviorEdgesIH = []; // {el, i, j, weight} — input neuron i -> hidden neuron j
let behaviorEdgesHO = []; // {el, i, j, weight} — hidden neuron i -> output neuron j

function buildWeightEdges(svg, fromLayer, toLayer, weights, describeEdge) {
  // weights[j][i]: weight from input-layer node i to output-layer node j
  // (torch.nn.Linear stores weight as (out_features, in_features)).
  const edges = [];
  for (let j = 0; j < toLayer.count; j++) {
    for (let i = 0; i < fromLayer.count; i++) {
      const weight = weights[j][i];
      const el = svg.appendChild(svgEl("line", {
        class: "net-edge net-edge-weight",
        x1: fromLayer.x, y1: layerNodeY(fromLayer, i),
        x2: toLayer.x, y2: layerNodeY(toLayer, j),
        "stroke-width": 0.6, "stroke-opacity": 0.03,
      }));
      if (describeEdge) bindInspectable(el, () => describeEdge(i, j, weight));
      edges.push({ el, i, j, weight });
    }
  }
  return edges;
}

// Combines each edge's fixed |weight| with its source neuron's current
// activation into one "how much signal is flowing through this connection
// right now" score, then normalizes across the layer so the strongest
// currently-active paths stand out.
function updateWeightEdges(edges, sourceLevels) {
  let maxSignal = 1e-6;
  const signals = edges.map((e) => Math.abs(e.weight) * (0.15 + 0.85 * sourceLevels[e.i]));
  signals.forEach((s) => { maxSignal = Math.max(maxSignal, s); });
  edges.forEach((e, idx) => {
    const norm = signals[idx] / maxSignal;
    e.el.setAttribute("stroke-opacity", (0.03 + 0.92 * norm).toFixed(3));
    e.el.setAttribute("stroke-width", (0.4 + 1.8 * norm).toFixed(2));
  });
}

const BEHAVIOR_HIDDEN_LABELS = Array.from({ length: 16 }, (_, i) => `Hidden ${i + 1}`);
const BEHAVIOR_OUTPUT_LABELS = Array.from({ length: 16 }, (_, i) => `Output dim ${i + 1}`);

function buildBehaviorNN(activity) {
  const svg = document.getElementById("behavior-nn-svg");
  svg.innerHTML = "";

  behaviorEdgesIH = buildWeightEdges(svg, BEHAVIOR_NN.input, BEHAVIOR_NN.hidden, activity.behavior_w1, (i, j, weight) => ({
    kicker: "Connection — behavior neural network",
    title: `${BEHAVIOR_LABELS[i]} → Hidden ${j + 1}`,
    subtitle: "A fixed, trained weight — lit up right now by how active its source input currently is.",
    rows: [
      { label: "Weight (fixed)", value: fmtSigned(weight) },
      { label: `Current value of "${BEHAVIOR_LABELS[i]}"`, value: fmtSigned(lastNetworkActivity.behavior_activity[i]) },
    ],
  }));
  behaviorEdgesHO = buildWeightEdges(svg, BEHAVIOR_NN.hidden, BEHAVIOR_NN.output, activity.behavior_w2, (i, j, weight) => ({
    kicker: "Connection — behavior neural network",
    title: `Hidden ${i + 1} → Output dim ${j + 1}`,
    subtitle: "A fixed, trained weight — lit up right now by how active its source hidden neuron currently is.",
    rows: [
      { label: "Weight (fixed)", value: fmtSigned(weight) },
      { label: "Current hidden activation", value: fmtSigned(lastNetworkActivity.behavior_hidden_activity[i]) },
    ],
  }));
  addTopParticles(svg, behaviorEdgesIH, 4);
  addTopParticles(svg, behaviorEdgesHO, 4);

  for (let i = 0; i < BEHAVIOR_NN.input.count; i++) {
    const node = svgEl("circle", {
      id: `behavior-input-${i}`, class: "net-node net-node-behavior-input",
      cx: BEHAVIOR_NN.input.x, cy: layerNodeY(BEHAVIOR_NN.input, i), r: 9,
    });
    svg.appendChild(withTitle(node, `Input neuron: ${BEHAVIOR_LABELS[i]}`));
    bindInspectable(node, () => ({
      kicker: "Input — behavior neural network",
      title: BEHAVIOR_LABELS[i],
      subtitle: "A raw feature from this attempt, fed directly into the network (no weight on this side).",
      rows: [{ label: "Current value", value: fmtSigned(lastNetworkActivity.behavior_activity[i]) }],
    }));
  }
  for (let i = 0; i < BEHAVIOR_NN.hidden.count; i++) {
    const node = svgEl("circle", {
      id: `behavior-hidden-${i}`, class: "net-node net-node-behavior-hidden",
      cx: BEHAVIOR_NN.hidden.x, cy: layerNodeY(BEHAVIOR_NN.hidden, i), r: 4,
    });
    svg.appendChild(withTitle(node, `Hidden neuron ${i + 1} of 16 (ReLU activation)`));
    bindInspectable(node, () => ({
      kicker: "Hidden neuron — behavior neural network",
      title: `Hidden ${i + 1} of 16`,
      subtitle: "ReLU activation. Incoming weights, ranked by |weight|:",
      rows: [
        { label: "Current activation", value: fmtSigned(lastNetworkActivity.behavior_hidden_activity[i]) },
        ...weightRows(lastNetworkActivity.behavior_w1[i], BEHAVIOR_LABELS),
      ],
    }));
  }
  for (let i = 0; i < BEHAVIOR_NN.output.count; i++) {
    const node = svgEl("circle", {
      id: `behavior-output-${i}`, class: "net-node net-node-behavior-output",
      cx: BEHAVIOR_NN.output.x, cy: layerNodeY(BEHAVIOR_NN.output, i), r: 4,
    });
    svg.appendChild(withTitle(node, `Output dimension ${i + 1} of 16 — part of the behavior embedding`));
    bindInspectable(node, () => ({
      kicker: "Output — behavior neural network",
      title: `Output dim ${i + 1} of 16`,
      subtitle: "Part of the behavior embedding fed into fusion. Incoming weights, ranked by |weight|:",
      rows: [
        { label: "Current activation", value: fmtSigned(lastNetworkActivity.behavior_output_activity[i]) },
        ...weightRows(lastNetworkActivity.behavior_w2[i], BEHAVIOR_HIDDEN_LABELS),
      ],
    }));
  }

  svg.appendChild(svgEl("text", { class: "net-label", x: BEHAVIOR_NN.input.x, y: 285 })).textContent = "Input (4)";
  svg.appendChild(svgEl("text", { class: "net-label", x: BEHAVIOR_NN.hidden.x, y: 285 })).textContent = "Hidden (16)";
  svg.appendChild(svgEl("text", { class: "net-label", x: BEHAVIOR_NN.output.x, y: 285 })).textContent = "Output (16)";
}

function updateBehaviorNN(activity) {
  const inputLevels = activity.behavior_activity.map((v) => Math.max(0, Math.min(1, Math.abs(v))));
  const hiddenLevels = activity.behavior_hidden_activity.map((v) => squash(v, 2));
  const outputLevels = activity.behavior_output_activity.map((v) => squash(v, 2));

  inputLevels.forEach((level, i) => {
    document.getElementById(`behavior-input-${i}`).setAttribute("fill-opacity", 0.35 + level * 0.65);
  });
  hiddenLevels.forEach((level, i) => {
    document.getElementById(`behavior-hidden-${i}`).setAttribute("fill-opacity", 0.25 + level * 0.75);
  });
  outputLevels.forEach((level, i) => {
    document.getElementById(`behavior-output-${i}`).setAttribute("fill-opacity", 0.25 + level * 0.75);
  });

  updateWeightEdges(behaviorEdgesIH, inputLevels);
  updateWeightEdges(behaviorEdgesHO, hiddenLevels);
}

// ---------------------------------------------------------------------------
// Panel 2: the EEG branch as its own graph — BOTH GCN layers' 8 channel nodes
// with the model's real learned (softmax) adjacency as edges (previously only
// layer 1's structure was ever drawn, leaving layer 2 entirely invisible),
// plus the pooled EEG embedding and a second, real "EEG-only" classifier
// (model/ablation.py's eeg_only variant) reading directly off that embedding
// — no behavior data anywhere in this panel.
// ---------------------------------------------------------------------------

const EEG_GRAPH = {
  layer1: { center: { x: 75, y: 85 }, radius: 55 },
  layer2: { center: { x: 75, y: 245 }, radius: 55 },
  embed: { x: 195, y: 245 },
  eegOnly: { x: 195, y: 145 },
};

function eegNodePosition(layer, i) {
  const angle = (-90 + i * (360 / NUM_EEG_NODES)) * (Math.PI / 180);
  return {
    x: layer.center.x + layer.radius * Math.cos(angle),
    y: layer.center.y + layer.radius * Math.sin(angle),
  };
}

let eegLayer1Edges = []; // {el, i, j, weight} — fixed learned adjacency, redrawn per move by current channel activity
let eegLayer2Edges = [];

// A perfectly flat/uniform adjacency (as gcn2's actually is — see
// model/MODEL_EVIDENCE.md) should still read as "channels are connected,
// just not differentiated" rather than "nothing is connected" — an
// all-edges-hidden diagram looks broken/incomplete, not honest. So every
// edge is always drawn; only its INTENSITY is scaled by how far the real
// weight deviates from the uniform (1/n) baseline, with a guaranteed
// minimum floor so a flat matrix still shows a faint but visible mesh.
const EEG_ADJACENCY_BASELINE = 1 / NUM_EEG_NODES;
const EEG_MAX_DEVIATION = 1 - EEG_ADJACENCY_BASELINE;

function adjacencyDeviationStrength(weight) {
  return Math.min(1, Math.abs(weight - EEG_ADJACENCY_BASELINE) / EEG_MAX_DEVIATION);
}

// Builds one GCN layer's 8-channel circle graph — shared by both layers.
function buildGcnLayerCircle(svg, layer, adjacency, idPrefix, labelText, activityKey) {
  const edgesGroup = svgEl("g", { id: `${idPrefix}-edges` });
  svg.appendChild(edgesGroup);
  const edges = [];
  for (let i = 0; i < NUM_EEG_NODES; i++) {
    for (let j = i + 1; j < NUM_EEG_NODES; j++) {
      const weight = (adjacency[i][j] + adjacency[j][i]) / 2;
      const a = eegNodePosition(layer, i), b = eegNodePosition(layer, j);
      const el = edgesGroup.appendChild(svgEl("line", {
        class: "net-edge", x1: a.x, y1: a.y, x2: b.x, y2: b.y, "stroke-width": 1,
      }));
      bindInspectable(el, () => ({
        kicker: `Connection — ${labelText}`,
        title: `Channel ${i + 1} ↔ Channel ${j + 1}`,
        subtitle: `Softmax-normalized learned adjacency (uniform/no-structure baseline = ${EEG_ADJACENCY_BASELINE.toFixed(4)}).`,
        rows: [
          { label: "Adjacency weight (fixed)", value: weight.toFixed(4) },
          { label: `Ch${i + 1} current activation`, value: lastNetworkActivity[activityKey][i].toFixed(4) },
          { label: `Ch${j + 1} current activation`, value: lastNetworkActivity[activityKey][j].toFixed(4) },
        ],
      }));
      edges.push({ el, i, j, weight });
    }
  }
  addTopParticles(svg, edges, 3);

  for (let i = 0; i < NUM_EEG_NODES; i++) {
    const pos = eegNodePosition(layer, i);
    const node = svgEl("circle", {
      id: `${idPrefix}-node-${i}`, class: "net-node net-node-eeg", cx: pos.x, cy: pos.y, r: 10,
    });
    svg.appendChild(withTitle(node, `EEG channel ${i + 1} of 8, ${labelText}`));
    bindInspectable(node, () => ({
      kicker: `Channel node — ${labelText}`,
      title: `EEG channel ${i + 1} of 8`,
      subtitle: "Adjacency to every other channel in this layer, ranked by |weight|:",
      rows: [
        { label: "Current activation magnitude", value: lastNetworkActivity[activityKey][i].toFixed(4) },
        ...weightRows(
          adjacency[i].filter((_, k) => k !== i),
          adjacency[i].map((_, k) => `Ch${k + 1}`).filter((_, k) => k !== i),
        ),
      ],
    }));
  }
  svg.appendChild(svgEl("text", {
    class: "net-label", x: layer.center.x, y: layer.center.y - layer.radius - 14,
  })).textContent = labelText;

  return edges;
}

function updateGcnLayerCircle(idPrefix, edges, channelLevels) {
  for (let i = 0; i < NUM_EEG_NODES; i++) {
    const node = document.getElementById(`${idPrefix}-node-${i}`);
    node.setAttribute("r", 7 + channelLevels[i] * 8);
    node.setAttribute("fill-opacity", 0.35 + channelLevels[i] * 0.65);
  }
  // Same "fixed structure × current activation" treatment as the behavior
  // NN's edges: the connection itself never changes, but how much of it is
  // lit up right now depends on how active both channels it joins currently
  // are. The structural half is scaled by deviation from the uniform
  // baseline (not the raw weight) with a floor, so a genuinely flat layer
  // (gcn2) still shows a faint, visible mesh instead of no edges at all.
  edges.forEach(({ el, i, j, weight }) => {
    const structuralStrength = adjacencyDeviationStrength(weight);
    const activityLevel = (channelLevels[i] + channelLevels[j]) / 2;
    const intensity = (0.15 + 0.85 * structuralStrength) * (0.3 + 0.7 * activityLevel);
    el.setAttribute("stroke-width", (0.6 + intensity * 3).toFixed(2));
    el.setAttribute("stroke-opacity", Math.min(1, 0.08 + intensity * 0.9).toFixed(3));
  });
}

function buildEegGraph(activity) {
  const svg = document.getElementById("eeg-graph-svg");
  svg.innerHTML = "";

  eegLayer1Edges = buildGcnLayerCircle(svg, EEG_GRAPH.layer1, activity.gcn1_adjacency, "gcn1", "GCN layer 1", "gcn1_node_activity");
  eegLayer2Edges = buildGcnLayerCircle(svg, EEG_GRAPH.layer2, activity.gcn2_adjacency, "gcn2", "GCN layer 2", "gcn2_node_activity");

  // Layer 1's output feeds layer 2 (both keep the same 8-channel structure —
  // see model/graph.py) — one connecting edge per channel makes that a
  // visible pipeline instead of two unrelated-looking circles.
  for (let i = 0; i < NUM_EEG_NODES; i++) {
    const from = eegNodePosition(EEG_GRAPH.layer1, i);
    const to = eegNodePosition(EEG_GRAPH.layer2, i);
    svg.appendChild(svgEl("line", {
      class: "net-edge net-edge-flow", x1: from.x, y1: from.y, x2: to.x, y2: to.y,
      "stroke-width": 1, "stroke-opacity": 0.15,
    }));
  }

  flowEdgeIn(svg, { x: EEG_GRAPH.layer2.center.x + EEG_GRAPH.layer2.radius, y: EEG_GRAPH.layer2.center.y }, EEG_GRAPH.embed);
  const embedNode = svgEl("circle", {
    id: "eeg-embed-node", class: "net-node net-node-embed", cx: EEG_GRAPH.embed.x, cy: EEG_GRAPH.embed.y, r: 18,
  });
  svg.appendChild(withTitle(embedNode, "EEG embedding — GCN layer 2's pooled output across all 8 channels"));
  bindInspectable(embedNode, () => ({
    kicker: "Embedding — EEG branch",
    title: "EEG embedding",
    subtitle: "GCN layer 2's output, mean-pooled across all 8 channels — this vector is what gets concatenated with the behavior embedding in Fusion.",
    rows: [{ label: "Vector norm (current)", value: lastNetworkActivity.eeg_embedding_norm.toFixed(4) }],
  }));
  svg.appendChild(svgEl("text", { class: "net-label", x: EEG_GRAPH.embed.x, y: EEG_GRAPH.embed.y + 32 })).textContent = "EEG embed";

  // A second, real classifier — trained with behavior zeroed out — reading
  // directly off the EEG embedding. Only populated when network_activity
  // carries eeg_only_* fields (i.e. the ablation checkpoint exists on the
  // server); otherwise it just sits at its built default ("—") harmlessly.
  flowEdgeIn(svg, EEG_GRAPH.embed, EEG_GRAPH.eegOnly);
  const eegOnlyNode = svgEl("circle", {
    id: "eeg-only-node", class: "net-node net-node-eegonly", cx: EEG_GRAPH.eegOnly.x, cy: EEG_GRAPH.eegOnly.y, r: 12,
  });
  svg.appendChild(withTitle(eegOnlyNode, "EEG-only classifier — predicts cognitive state from EEG alone (no behavior data)"));
  bindInspectable(eegOnlyNode, () => ({
    kicker: "Classifier — EEG-only",
    title: "EEG-only classifier",
    subtitle: "A real second classifier (model/ablation.py's eeg_only variant), trained with behavior zeroed out — reads only the EEG embedding.",
    rows: lastNetworkActivity.eeg_only_state
      ? [
          { label: "Predicted state", value: lastNetworkActivity.eeg_only_state },
          { label: "Confidence", value: `${(lastNetworkActivity.eeg_only_confidence * 100).toFixed(1)}%` },
        ]
      : [{ label: "Status", value: "Ablation checkpoint not loaded on this server" }],
  }));
  svg.appendChild(svgEl("text", {
    class: "net-label", x: EEG_GRAPH.eegOnly.x, y: EEG_GRAPH.eegOnly.y - 20,
  })).textContent = "EEG-only net";
  svg.appendChild(svgEl("text", {
    id: "eeg-only-state-label", class: "net-label", x: EEG_GRAPH.eegOnly.x, y: EEG_GRAPH.eegOnly.y + 28,
  })).textContent = "—";
}

function updateEegGraph(activity) {
  const layer1Levels = activity.gcn1_node_activity.map((v) => squash(v, 2));
  const layer2Levels = activity.gcn2_node_activity.map((v) => squash(v, 2));
  updateGcnLayerCircle("gcn1", eegLayer1Edges, layer1Levels);
  updateGcnLayerCircle("gcn2", eegLayer2Edges, layer2Levels);

  const eegLevel = squash(activity.eeg_embedding_norm, 2);
  document.getElementById("eeg-embed-node").setAttribute("fill-opacity", 0.35 + eegLevel * 0.65);

  if (activity.eeg_only_state) {
    const eegOnlyNode = document.getElementById("eeg-only-node");
    const eegOnlyConf = activity.eeg_only_confidence || 0;
    eegOnlyNode.setAttribute("r", 10 + eegOnlyConf * 10);
    eegOnlyNode.setAttribute("fill-opacity", 0.35 + eegOnlyConf * 0.65);
    document.getElementById("eeg-only-state-label").textContent =
      `${activity.eeg_only_state} (${Math.round(eegOnlyConf * 100)}%)`;
  }
  return eegLevel;
}

// ---------------------------------------------------------------------------
// Panel 3: fusion + LSTM + the 5 cognitive-state outputs. Takes the two
// embeddings produced by panels 1 and 2 as its inputs (duplicated nodes, same
// real values) so the three panels still read as one pipeline left-to-right.
// ---------------------------------------------------------------------------

const FUSION_NET = {
  behaviorIn: { x: 40, y: 100 },
  eegIn: { x: 40, y: 200 },
  fusion: { x: 150, y: 150 },
  // The LSTM is a real 32-unit hidden layer, not one opaque blob — rendering
  // it as a full layer (like the behavior branch's hidden layer) lets its
  // connection to the outputs below be drawn with the classifier's REAL
  // learned weights instead of generic, weightless "data moves" lines.
  lstm: { x: 250, count: 32, y0: 20, spacing: 8 },
  outputs: { x: 350, count: 5, y0: 40, spacing: 55 },
};

let fusionFlowEdges = {}; // named refs to the flow-edge <line> elements, re-highlighted per move
let fusionToLstmEdges = []; // fan-out from the fusion node — real endpoints, but no learned per-edge weight to show (that's the LSTM's own recurrence, not modeled here)
let lstmToOutputEdges = []; // {el, i, j, weight} — the real classifier.weight matrix (5 states x 32 hidden units)

function buildFusionGraph(activity) {
  const svg = document.getElementById("fusion-svg");
  svg.innerHTML = "";
  fusionFlowEdges = {
    behaviorToFusion: flowEdgeIn(svg, FUSION_NET.behaviorIn, FUSION_NET.fusion),
    eegToFusion: flowEdgeIn(svg, FUSION_NET.eegIn, FUSION_NET.fusion),
  };

  const behaviorInNode = svgEl("circle", {
    id: "fusion-behavior-in", class: "net-node net-node-embed", cx: FUSION_NET.behaviorIn.x, cy: FUSION_NET.behaviorIn.y, r: 16,
  });
  svg.appendChild(withTitle(behaviorInNode, "Behavior embedding — carried over from the behavior neural network panel"));
  bindInspectable(behaviorInNode, () => ({
    kicker: "Embedding — Fusion",
    title: "Behavior embedding (in)",
    subtitle: "Same vector as the behavior neural network panel's output layer.",
    rows: [{ label: "Vector norm (current)", value: lastNetworkActivity.behavior_embedding_norm.toFixed(4) }],
  }));
  svg.appendChild(svgEl("text", { class: "net-label", x: FUSION_NET.behaviorIn.x, y: FUSION_NET.behaviorIn.y - 24 })).textContent = "Behavior embed";

  const eegInNode = svgEl("circle", {
    id: "fusion-eeg-in", class: "net-node net-node-embed", cx: FUSION_NET.eegIn.x, cy: FUSION_NET.eegIn.y, r: 16,
  });
  svg.appendChild(withTitle(eegInNode, "EEG embedding — carried over from the EEG graph panel"));
  bindInspectable(eegInNode, () => ({
    kicker: "Embedding — Fusion",
    title: "EEG embedding (in)",
    subtitle: "Same vector as the EEG graph panel's EEG embed node.",
    rows: [{ label: "Vector norm (current)", value: lastNetworkActivity.eeg_embedding_norm.toFixed(4) }],
  }));
  svg.appendChild(svgEl("text", { class: "net-label", x: FUSION_NET.eegIn.x, y: FUSION_NET.eegIn.y + 30 })).textContent = "EEG embed";

  const fusionNode = svgEl("circle", {
    id: "fusion-node", class: "net-node net-node-fusion", cx: FUSION_NET.fusion.x, cy: FUSION_NET.fusion.y, r: 20,
  });
  svg.appendChild(withTitle(fusionNode, "Fusion layer — concatenates the EEG and behavior embeddings into one vector"));
  bindInspectable(fusionNode, () => ({
    kicker: "Fusion",
    title: "Fusion layer",
    subtitle: "Concatenation, not a learned layer — no weights of its own. Combines the two embeddings above into the vector the LSTM sees.",
    rows: [
      { label: "Behavior embedding norm", value: lastNetworkActivity.behavior_embedding_norm.toFixed(4) },
      { label: "EEG embedding norm", value: lastNetworkActivity.eeg_embedding_norm.toFixed(4) },
    ],
  }));
  svg.appendChild(svgEl("text", { class: "net-label", x: FUSION_NET.fusion.x, y: FUSION_NET.fusion.y + 34 })).textContent = "Fusion";

  fusionToLstmEdges = [];
  for (let i = 0; i < FUSION_NET.lstm.count; i++) {
    const y = layerNodeY(FUSION_NET.lstm, i);
    const el = svg.appendChild(svgEl("line", {
      class: "net-edge net-edge-flow", x1: FUSION_NET.fusion.x, y1: FUSION_NET.fusion.y,
      x2: FUSION_NET.lstm.x, y2: y, "stroke-width": 0.6, "stroke-opacity": 0.08,
    }));
    bindInspectable(el, () => ({
      kicker: "Connection — Fusion → LSTM",
      title: `Fusion → LSTM unit ${i + 1}`,
      subtitle: "The LSTM's own recurrent gating (its weight_ih/weight_hh matrices) isn't modeled in this diagram — this edge has no single displayable weight, only the real fused signal flowing into the recurrence.",
      rows: [{ label: "LSTM unit's current activation", value: fmtSigned(lastNetworkActivity.lstm_hidden_activity[i]) }],
    }));
    fusionToLstmEdges.push(el);
    const node = svgEl("circle", {
      id: `lstm-node-${i}`, class: "net-node net-node-lstm", cx: FUSION_NET.lstm.x, cy: y, r: 3,
    });
    svg.appendChild(withTitle(node, `LSTM hidden unit ${i + 1} of 32 — carries memory across the last 5 attempts`));
    bindInspectable(node, () => ({
      kicker: "Hidden unit — LSTM",
      title: `LSTM unit ${i + 1} of 32`,
      subtitle: "Outgoing weights to each cognitive state (the real classifier.weight column for this unit):",
      rows: [
        { label: "Current activation", value: fmtSigned(lastNetworkActivity.lstm_hidden_activity[i]) },
        ...KNOWN_STATES.map((state, j) => ({ label: `→ ${state}`, value: fmtSigned(lastNetworkActivity.classifier_weight[j][i]) })),
      ],
    }));
  }
  svg.appendChild(svgEl("text", {
    class: "net-label", x: FUSION_NET.lstm.x, y: layerNodeY(FUSION_NET.lstm, FUSION_NET.lstm.count - 1) + 20,
  })).textContent = "LSTM (32)";

  // The classifier is a real, simple Linear(32, 5) layer — its weight matrix
  // is exactly what buildWeightEdges/updateWeightEdges already draw for the
  // behavior branch, reused here so this final hop is genuinely
  // weight-accurate instead of a set of generic flow lines.
  lstmToOutputEdges = buildWeightEdges(svg, FUSION_NET.lstm, FUSION_NET.outputs, activity.classifier_weight, (i, j, weight) => ({
    kicker: "Connection — classifier (Linear 32 → 5)",
    title: `LSTM unit ${i + 1} → ${KNOWN_STATES[j]}`,
    subtitle: "A fixed, trained weight — lit up right now by how active its source LSTM unit currently is.",
    rows: [
      { label: "Weight (fixed)", value: fmtSigned(weight) },
      { label: "LSTM unit's current activation", value: fmtSigned(lastNetworkActivity.lstm_hidden_activity[i]) },
    ],
  }));

  KNOWN_STATES.forEach((state, i) => {
    const y = layerNodeY(FUSION_NET.outputs, i);
    const outputNode = svgEl("circle", {
      id: `output-node-${state}`, class: "net-node net-node-output", cx: FUSION_NET.outputs.x, cy: y, r: 12,
    });
    svg.appendChild(withTitle(outputNode, `Predicted probability of cognitive state: ${state}`));
    bindInspectable(outputNode, () => ({
      kicker: "Output",
      title: state,
      subtitle: "Incoming classifier weights from the 32 LSTM units, ranked by |weight|:",
      rows: [
        { label: "Predicted probability", value: `${(lastNetworkActivity.probs[state] * 100).toFixed(1)}%` },
        ...weightRows(
          lastNetworkActivity.classifier_weight[i],
          Array.from({ length: 32 }, (_, u) => `LSTM ${u + 1}`),
        ),
      ],
    }));
    svg.appendChild(svgEl("text", {
      // +30, not +20: the circle's radius can grow up to 20 (see updateFusionGraph's
      // 8 + p*12 for p=1), which would otherwise sit under the label at a high-confidence
      // prediction. Anchor is set via inline `style`, not the `text-anchor` attribute:
      // the .net-label CSS class's own text-anchor:middle otherwise wins the cascade over
      // a same-specificity presentation attribute, silently re-centering the label on x.
      class: "net-label", x: FUSION_NET.outputs.x + 30, y: y + 4, style: "text-anchor: start",
    })).textContent = state;
  });
}

// A flow edge has no per-connection weight to show (it's just "data moves
// from A to B"), so its highlight is driven entirely by how active its two
// real endpoints are right now — still genuine per-move signal, not decoration.
function setFlowIntensity(edge, level) {
  edge.setAttribute("stroke-opacity", (0.2 + level * 0.7).toFixed(3));
  edge.setAttribute("stroke-width", (1 + level * 2.5).toFixed(2));
}

function updateFusionGraph(activity, eegLevel) {
  const behaviorLevel = squash(activity.behavior_embedding_norm, 2);
  document.getElementById("fusion-behavior-in").setAttribute("fill-opacity", 0.35 + behaviorLevel * 0.65);
  document.getElementById("fusion-eeg-in").setAttribute("fill-opacity", 0.35 + eegLevel * 0.65);
  setFlowIntensity(fusionFlowEdges.behaviorToFusion, behaviorLevel);
  setFlowIntensity(fusionFlowEdges.eegToFusion, eegLevel);
  document.getElementById("fusion-node").setAttribute("fill-opacity", 0.35 + (eegLevel + behaviorLevel) / 2 * 0.65);
  fusionToLstmEdges.forEach((edge) => setFlowIntensity(edge, (eegLevel + behaviorLevel) / 2));

  // Real per-unit LSTM activation, not a single aggregate norm — each of the
  // 32 hidden units lights up independently based on its own actual value.
  const lstmLevels = activity.lstm_hidden_activity.map((v) => squash(v, 3));
  lstmLevels.forEach((level, i) => {
    document.getElementById(`lstm-node-${i}`).setAttribute("fill-opacity", 0.25 + level * 0.75);
  });
  updateWeightEdges(lstmToOutputEdges, lstmLevels);

  KNOWN_STATES.forEach((state) => {
    const node = document.getElementById(`output-node-${state}`);
    const p = activity.probs[state] || 0;
    node.setAttribute("r", 8 + p * 12); // max r=20, well clear of the label at +30 (see buildFusionGraph)
    node.setAttribute("fill-opacity", 0.3 + p * 0.7);
    node.classList.toggle("predicted", state === activity.state);
  });
}

// network_activity's shape is a live contract with the server's currently
// running model/inference.py — a stale browser tab talking to a redeployed
// server (or vice versa) could receive a payload missing fields this code
// expects. The whole "update" message handler continues past this call
// (setEvalBar, hints, gamification…), so an uncaught error here would take
// all of that down with it; this call site guards specifically against that.
function updateNetworkDiagram(activity) {
  if (!activity) return;
  try {
    renderNetworkDiagram(activity);
  } catch (err) {
    console.error("Network activity panel could not render this update — skipping.", err);
  }
}

function renderNetworkDiagram(activity) {
  lastNetworkActivity = activity;
  if (!panelsBuilt) {
    buildBehaviorNN(activity);
    buildEegGraph(activity);
    buildFusionGraph(activity);
    panelsBuilt = true;
  }

  updateBehaviorNN(activity);
  const eegLevel = updateEegGraph(activity);
  updateFusionGraph(activity, eegLevel);
  triggerFlowParticles();

  document.querySelectorAll(".net-panel svg").forEach((svg) => {
    svg.classList.remove("pulse");
    void svg.offsetWidth; // force reflow so re-adding the class restarts the CSS animation
    svg.classList.add("pulse");
  });
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

// ---------------------------------------------------------------------------
// "Visual dopamine" for a fully solved puzzle: a one-shot confetti burst, plus
// a pop animation on the feedback callout. Purely decorative (no model data
// involved) — unlike the network diagram above, this is deliberately theatrical.
// ---------------------------------------------------------------------------

const CONFETTI_COLORS = ["#4a7c2f", "#b58863", "#3f7fbf", "#b91c1c", "#c2410c", "#6d28d9", "#eab308"];

function celebrate() {
  const count = 40;
  for (let i = 0; i < count; i++) {
    const piece = document.createElement("div");
    piece.className = "confetti-piece";
    piece.style.left = `${Math.random() * 100}vw`;
    piece.style.background = CONFETTI_COLORS[i % CONFETTI_COLORS.length];
    const duration = 900 + Math.random() * 700;
    const delay = Math.random() * 200;
    piece.style.animationDuration = `${duration}ms`;
    piece.style.animationDelay = `${delay}ms`;
    document.body.appendChild(piece);
    setTimeout(() => piece.remove(), duration + delay + 50);
  }
}

function popCallout(el) {
  el.classList.remove("pop");
  void el.offsetWidth; // force reflow so re-adding the class restarts the animation
  el.classList.add("pop");
}

// ---------------------------------------------------------------------------
// Captured-pieces trays: derived fresh from currentBoard on every render (no
// separate tracking state to drift out of sync) by diffing against a standard
// 16-piece-per-side set. This also correctly accounts for pieces already
// missing when a mid-game puzzle position loads, not just captures made live.
// ---------------------------------------------------------------------------

const STANDARD_PIECE_COUNTS = { p: 8, n: 2, b: 2, r: 2, q: 1 };
const PIECE_VALUE = { p: 1, n: 3, b: 3, r: 5, q: 9 };

function renderCapturedGlyphs(pieces, advantage) {
  const nodes = [];
  for (const p of pieces) {
    const span = document.createElement("span");
    span.textContent = PIECE_GLYPH[p.toLowerCase()];
    span.style.opacity = "0.65";
    nodes.push(span);
  }
  if (advantage > 0) {
    const adv = document.createElement("span");
    adv.className = "material-diff";
    adv.textContent = `+${advantage}`;
    nodes.push(adv);
  }
  return nodes;
}

function renderCapturedTrays() {
  const topEl = document.getElementById("captured-top");
  const bottomEl = document.getElementById("captured-bottom");
  if (!currentBoard || !topEl || !bottomEl) return;

  const counts = {};
  for (const row of currentBoard) {
    for (const cell of row) {
      if (cell) counts[cell] = (counts[cell] || 0) + 1;
    }
  }
  const missingWhite = []; // white pieces gone from the board => captured BY black
  const missingBlack = []; // black pieces gone from the board => captured BY white
  let whiteCapturedValue = 0;
  let blackCapturedValue = 0;
  for (const [type, max] of Object.entries(STANDARD_PIECE_COUNTS)) {
    const missingW = max - (counts[type.toUpperCase()] || 0);
    for (let i = 0; i < missingW; i++) missingWhite.push(type.toUpperCase());
    const missingB = max - (counts[type] || 0);
    for (let i = 0; i < missingB; i++) missingBlack.push(type);
    blackCapturedValue += missingW * PIECE_VALUE[type]; // material Black has won
    whiteCapturedValue += missingB * PIECE_VALUE[type]; // material White has won
  }

  const capturedByWhite = renderCapturedGlyphs(missingBlack, whiteCapturedValue - blackCapturedValue);
  const capturedByBlack = renderCapturedGlyphs(missingWhite, blackCapturedValue - whiteCapturedValue);
  // Keep "captured by White" nearest White's own side of the board, whichever
  // edge that currently is.
  if (boardFlipped) {
    topEl.replaceChildren(...capturedByWhite);
    bottomEl.replaceChildren(...capturedByBlack);
  } else {
    topEl.replaceChildren(...capturedByBlack);
    bottomEl.replaceChildren(...capturedByWhite);
  }
}

// ---------------------------------------------------------------------------
// Sound effects: short synthesized tones (Web Audio, no external audio files)
// for moves/captures/results. All moves that trigger a sound originate from a
// real user click, so this never runs into autoplay-without-a-gesture limits.
// ---------------------------------------------------------------------------

let soundEnabled = true;
let audioCtx = null;

function getAudioCtx() {
  if (!audioCtx) {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    audioCtx = new Ctx();
  }
  return audioCtx;
}

function beep(freq, duration, type, gainPeak) {
  if (!soundEnabled) return;
  try {
    const ctx = getAudioCtx();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = type || "sine";
    osc.frequency.value = freq;
    gain.gain.setValueAtTime(gainPeak || 0.12, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + duration);
    osc.connect(gain).connect(ctx.destination);
    osc.start();
    osc.stop(ctx.currentTime + duration);
  } catch (err) {
    // Autoplay restrictions or an unsupported browser — sound is a nice-to-have,
    // never something the rest of the app should depend on.
  }
}

const SOUND_EFFECTS = {
  move: () => beep(420, 0.07, "square", 0.06),
  capture: () => beep(260, 0.1, "square", 0.09),
  correct: () => { beep(523.25, 0.12, "sine", 0.12); setTimeout(() => beep(659.25, 0.16, "sine", 0.12), 90); },
  wrong: () => beep(150, 0.28, "sawtooth", 0.1),
  invalid: () => beep(300, 0.08, "square", 0.05),
  achievement: () => { beep(660, 0.09, "sine", 0.1); setTimeout(() => beep(880, 0.14, "sine", 0.1), 80); },
};

function playSound(name) {
  const effect = SOUND_EFFECTS[name];
  if (effect) effect();
}

function toggleSound() {
  soundEnabled = !soundEnabled;
  const btn = document.getElementById("sound-toggle");
  btn.textContent = soundEnabled ? "Sound: On" : "Sound: Off";
  btn.classList.toggle("muted", !soundEnabled);
}

// ---------------------------------------------------------------------------
// Move list + session stats: built from the real solver_san/opponent_san sent
// in each "update" message and the real correct/incorrect status stream — a
// fresh puzzle resets the list (it shows this attempt's line), the counters
// accumulate for the whole session.
// ---------------------------------------------------------------------------

let moveHistory = [];
let currentPuzzleId = null;
let currentPuzzleRating = 1000;
let puzzleCounter = 0;
let solvedCount = 0;
let currentStreak = 0;
let bestStreak = 0;
let hintUsedThisPuzzle = false;
let lastSubmittedTimeToMove = null;

function renderMoveList() {
  const listEl = document.getElementById("move-list");
  listEl.innerHTML = "";
  let pairNumber = 0;
  for (const entry of moveHistory) {
    const li = document.createElement("li");
    if (entry.mover === "solver") {
      pairNumber += 1;
      li.className = "move-solver";
      li.textContent = `${pairNumber}. ${entry.san}`;
    } else {
      li.className = "move-opponent";
      li.textContent = `${pairNumber}... ${entry.san}`;
    }
    listEl.appendChild(li);
  }
  listEl.scrollTop = listEl.scrollHeight;
}

function updateSessionStats() {
  document.getElementById("solved-count").textContent = solvedCount;
  document.getElementById("best-streak").textContent = bestStreak;
  document.getElementById("puzzle-counter").textContent = puzzleCounter;
  const streakBadge = document.getElementById("streak-badge");
  if (currentStreak >= 2) {
    streakBadge.hidden = false;
    document.getElementById("streak-count").textContent = currentStreak;
  } else {
    streakBadge.hidden = true;
  }
}

// ---------------------------------------------------------------------------
// Move notation explanation: breaks a real SAN string (e.g. "Nbxd7+") down
// into its individual symbols with a plain-English meaning for each one —
// pure syntax decoding, so it's done directly off the string the server sent,
// no extra round trip needed.
// ---------------------------------------------------------------------------

const SAN_PIECE_NAMES = { K: "king", Q: "queen", R: "rook", B: "bishop", N: "knight" };

function sanLegend(san) {
  if (san === "O-O" || san === "0-0") return [{ symbol: san, meaning: "Kingside castling" }];
  if (san === "O-O-O" || san === "0-0-0") return [{ symbol: san, meaning: "Queenside castling" }];

  const legend = [];
  const suffix = san.endsWith("#") ? "#" : san.endsWith("+") ? "+" : "";
  let body = suffix ? san.slice(0, -1) : san;

  let promo = null;
  const eqIdx = body.indexOf("=");
  if (eqIdx !== -1) {
    promo = body.slice(eqIdx + 1);
    body = body.slice(0, eqIdx);
  }

  const isCapture = body.includes("x");
  const core = body.replace("x", "");
  const dest = core.slice(-2);
  const prefix = core.slice(0, -2);

  if (prefix && SAN_PIECE_NAMES[prefix[0]]) {
    legend.push({ symbol: prefix[0], meaning: SAN_PIECE_NAMES[prefix[0]] });
    const disambig = prefix.slice(1);
    if (disambig) {
      legend.push({
        symbol: disambig,
        meaning: `disambiguates — specifies the ${/^[a-h]$/.test(disambig) ? "file" : "rank"}`,
      });
    }
  } else {
    legend.push({ symbol: "(none)", meaning: "no letter = pawn move" });
    if (prefix) legend.push({ symbol: prefix, meaning: "the pawn's starting file (needed because this is a capture)" });
  }

  if (isCapture) legend.push({ symbol: "x", meaning: "captures a piece" });
  legend.push({ symbol: dest, meaning: "destination square" });
  if (promo) legend.push({ symbol: `=${promo}`, meaning: `promotes to a ${SAN_PIECE_NAMES[promo] || promo}` });
  if (suffix === "+") legend.push({ symbol: "+", meaning: "check" });
  if (suffix === "#") legend.push({ symbol: "#", meaning: "checkmate" });
  return legend;
}

function showMoveExplanation(san, moverLabel) {
  const box = document.getElementById("move-explain");
  const sanEl = document.getElementById("move-explain-san");
  const bodyEl = document.getElementById("move-explain-body");
  sanEl.textContent = `${san} (${moverLabel})`;
  bodyEl.innerHTML = "";
  for (const { symbol, meaning } of sanLegend(san)) {
    const chip = document.createElement("span");
    chip.className = "san-chip";
    const symbolEl = document.createElement("span");
    symbolEl.className = "san-symbol";
    symbolEl.textContent = symbol;
    const meaningEl = document.createElement("span");
    meaningEl.className = "san-meaning";
    meaningEl.textContent = meaning;
    chip.append(symbolEl, meaningEl);
    bodyEl.appendChild(chip);
  }
  box.hidden = false;
}

// ---------------------------------------------------------------------------
// Gamification: XP, levels, and achievements — all derived from real session
// events (puzzle rating, streak, time to move, hint usage), never fabricated.
// XP formula: base = max(10, rating/25); + 5 per streak step (capped at 10
// steps); halved if a hint was used on that puzzle. 100 XP per level.
// ---------------------------------------------------------------------------

let totalXp = 0;

// 50 achievements across 8 categories — every single one is checked against
// real tracked session data (streaks, solve counts, timing, rating, hint use,
// XP/level, full-game results, and the actual predicted cognitive states seen)
// — none are randomly or arbitrarily granted.
const ACHIEVEMENTS = [
  // Streak milestones
  { id: "streak-2", name: "Warm Up", description: "Reach a 2-puzzle solve streak." },
  { id: "hat-trick", name: "Hat Trick", description: "Reach a 3-puzzle solve streak." },
  { id: "unstoppable", name: "Unstoppable", description: "Reach a 5-puzzle solve streak." },
  { id: "streak-7", name: "Lucky Seven", description: "Reach a 7-puzzle solve streak." },
  { id: "streak-10", name: "Perfect Ten", description: "Reach a 10-puzzle solve streak." },
  { id: "streak-15", name: "On Fire", description: "Reach a 15-puzzle solve streak." },
  { id: "streak-20", name: "Relentless", description: "Reach a 20-puzzle solve streak." },
  { id: "streak-25", name: "Legendary Streak", description: "Reach a 25-puzzle solve streak." },
  // Solve-count milestones
  { id: "first-blood", name: "First Blood", description: "Solve your first puzzle." },
  { id: "solved-5", name: "Getting Started", description: "Solve 5 puzzles total." },
  { id: "solved-10", name: "Puzzle Rookie", description: "Solve 10 puzzles total." },
  { id: "solved-25", name: "Puzzle Enthusiast", description: "Solve 25 puzzles total." },
  { id: "solved-50", name: "Puzzle Adept", description: "Solve 50 puzzles total." },
  { id: "solved-100", name: "Century Solver", description: "Solve 100 puzzles total." },
  { id: "solved-200", name: "Puzzle Master", description: "Solve 200 puzzles total." },
  { id: "solved-500", name: "Puzzle Grandmaster", description: "Solve 500 puzzles total." },
  // Speed
  { id: "speed-10", name: "Quick Thinker", description: "Solve a puzzle in under 10 seconds." },
  { id: "speed-solver", name: "Speed Solver", description: "Solve a puzzle in under 5 seconds." },
  { id: "speed-3", name: "Lightning Fast", description: "Solve a puzzle in under 3 seconds." },
  { id: "speed-1", name: "Blink Of An Eye", description: "Solve a puzzle in under 1 second." },
  // Rating ceilings
  { id: "rating-1200", name: "Club Player", description: "Solve a puzzle rated 1200 or higher." },
  { id: "rating-1500", name: "Rising Star", description: "Solve a puzzle rated 1500 or higher." },
  { id: "rating-1800", name: "Skilled Tactician", description: "Solve a puzzle rated 1800 or higher." },
  { id: "grandmaster-in-training", name: "Grandmaster in Training", description: "Solve a puzzle rated 2000 or higher." },
  { id: "rating-2200", name: "International Master", description: "Solve a puzzle rated 2200 or higher." },
  { id: "rating-2500", name: "World Class", description: "Solve a puzzle rated 2500 or higher." },
  // Hint discipline
  { id: "no-hints", name: "No Hints Needed", description: "Solve a puzzle without requesting any hint." },
  { id: "no-hint-streak-5", name: "Self Reliant", description: "Solve 5 puzzles in a row without any hint." },
  { id: "no-hint-total-20", name: "Independent Thinker", description: "Solve 20 puzzles total without any hint." },
  // XP / level
  { id: "century", name: "Century", description: "Earn 100 total XP." },
  { id: "level-3", name: "Rising", description: "Reach level 3." },
  { id: "level-5", name: "Adept", description: "Reach level 5." },
  { id: "level-10", name: "Veteran", description: "Reach level 10." },
  { id: "level-20", name: "Elite", description: "Reach level 20." },
  // Full-game mode
  { id: "first-game", name: "First Game", description: "Finish a full game to the end." },
  { id: "first-win", name: "Victory", description: "Win a full game against the AI." },
  { id: "win-as-white", name: "Light Side", description: "Win a full game playing White." },
  { id: "win-as-black", name: "Dark Side", description: "Win a full game playing Black." },
  { id: "first-draw", name: "Split The Point", description: "Draw a full game." },
  { id: "games-5", name: "Regular Player", description: "Finish 5 full games." },
  { id: "games-10", name: "Dedicated Player", description: "Finish 10 full games." },
  { id: "games-25", name: "Veteran Player", description: "Finish 25 full games." },
  { id: "wins-5", name: "Serial Winner", description: "Win 5 full games." },
  { id: "wins-10", name: "Dominant Force", description: "Win 10 full games." },
  // Cognitive state & comebacks
  { id: "zone-focused", name: "In The Zone", description: "Solve a puzzle while your predicted state is Focused." },
  { id: "flow-engaged", name: "Flow State", description: "Solve a puzzle while your predicted state is Engaged." },
  { id: "cool-under-pressure", name: "Cool Under Pressure", description: "Solve a puzzle while Overloaded, Confused, or Fatigued." },
  { id: "bounce-back", name: "Bounce Back", description: "Solve a puzzle right after a wrong attempt on it." },
  { id: "comeback-kid", name: "Comeback Kid", description: "Solve a puzzle after 3 or more wrong attempts on it." },
  { id: "full-spectrum", name: "Full Spectrum", description: "See all 5 cognitive states predicted in one session." },
  { id: "night-owl", name: "Night Owl", description: "Solve a puzzle between midnight and 4am local time." },
];
const unlockedAchievements = new Set();

// Extra tracked state the expanded achievement set needs, beyond
// solvedCount/currentStreak/bestStreak/totalXp already maintained elsewhere.
let noHintStreak = 0;
let noHintTotalCount = 0;
let maxRatingSolved = 0;
let wrongAttemptsThisPuzzle = 0;
let lastPredictedState = null;
const statesSeenThisSession = new Set();
let gamesPlayed = 0;
let gamesWon = 0;

// ---------------------------------------------------------------------------
// Gamification progress (XP, streak, solve counts, unlocked achievements) is
// the learner's lifetime record, not a single page load's — without this it
// silently resets to zero on every refresh, wiping out unlocked achievements.
// statesSeenThisSession/wrongAttemptsThisPuzzle/etc. are deliberately NOT
// persisted here: those are genuinely per-session/per-puzzle signals that
// should reset with a fresh server session, same as the board/move list.
// ---------------------------------------------------------------------------
const GAMIFICATION_STORAGE_KEY = "neurotutor_gamification_v1";

function saveGamificationState() {
  try {
    localStorage.setItem(GAMIFICATION_STORAGE_KEY, JSON.stringify({
      totalXp, solvedCount, currentStreak, bestStreak,
      noHintStreak, noHintTotalCount, maxRatingSolved, gamesPlayed, gamesWon,
      puzzleCounter,
      unlockedAchievements: [...unlockedAchievements],
    }));
  } catch (err) {
    // localStorage can throw (private browsing, storage disabled) — progress
    // just won't survive a refresh in that case; nothing else breaks.
  }
  syncProgressToAccount();
  renderBoardThemePicker(); // a newly-unlocked theme should show up immediately
}

function currentProgressPayload() {
  return {
    total_xp: totalXp, solved_count: solvedCount, current_streak: currentStreak, best_streak: bestStreak,
    no_hint_streak: noHintStreak, no_hint_total_count: noHintTotalCount, max_rating_solved: maxRatingSolved,
    games_played: gamesPlayed, games_won: gamesWon, puzzles_served: puzzleCounter,
    unlocked_achievements: [...unlockedAchievements],
  };
}

// Best-effort, fire-and-forget: localStorage (above) is always the source of
// truth for this browser, so a failed sync here just means the account copy
// is a little stale until the next change — nothing breaks locally.
function syncProgressToAccount() {
  if (!loggedInUsername) return;
  fetch("/account/progress", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(currentProgressPayload()),
  }).catch(() => {});
}

function loadGamificationState() {
  // Reset to defaults first (not just on the happy path): this runs again on
  // logout, where the in-memory state may currently hold a different
  // account's numbers — a missing/corrupt localStorage entry at that point
  // must still land on a clean slate, not silently keep the account's values.
  totalXp = 0;
  solvedCount = 0;
  currentStreak = 0;
  bestStreak = 0;
  noHintStreak = 0;
  noHintTotalCount = 0;
  maxRatingSolved = 0;
  gamesPlayed = 0;
  gamesWon = 0;
  puzzleCounter = 0;
  unlockedAchievements.clear();
  try {
    const raw = localStorage.getItem(GAMIFICATION_STORAGE_KEY);
    if (!raw) return;
    const saved = JSON.parse(raw);
    totalXp = saved.totalXp || 0;
    solvedCount = saved.solvedCount || 0;
    currentStreak = saved.currentStreak || 0;
    bestStreak = saved.bestStreak || 0;
    noHintStreak = saved.noHintStreak || 0;
    noHintTotalCount = saved.noHintTotalCount || 0;
    maxRatingSolved = saved.maxRatingSolved || 0;
    gamesPlayed = saved.gamesPlayed || 0;
    gamesWon = saved.gamesWon || 0;
    puzzleCounter = saved.puzzleCounter || 0;
    for (const id of saved.unlockedAchievements || []) unlockedAchievements.add(id);
  } catch (err) {
    // Corrupt or unavailable storage — the defaults set above already stand.
  }
}

// ---------------------------------------------------------------------------
// Account system: a lightweight username/password profile (see web/server.py's
// /auth and /account endpoints) that lets the same gamification progress
// follow a learner across browsers/devices instead of being stuck in one
// browser's localStorage, and adds a daily play-streak meter (separate from
// the puzzle-solve streak above — see accounts/streak.py) that only makes
// sense once progress is tied to an account rather than a single browser.
// Signed-out play is untouched: it keeps working exactly as before, purely
// off localStorage.
// ---------------------------------------------------------------------------

function updateAccountUI() {
  document.getElementById("account-status").textContent = loggedInUsername || "";
  document.getElementById("account-button").textContent = loggedInUsername ? "Log out" : "Log in";
}

// Lives right next to the username in the header, not the sidebar — compact
// by design, so the full best-streak/freeze detail moves into the tooltip
// instead of its own row of text.
function renderDailyStreak(dailyStreak) {
  const el = document.getElementById("daily-streak-inline");
  if (!dailyStreak || dailyStreak.current <= 0) {
    el.textContent = "";
    el.title = "";
    return;
  }
  const freezes = dailyStreak.freezes_available || 0;
  el.textContent = `🔥${dailyStreak.current}`;
  el.title =
    `${dailyStreak.current}-day streak (best ${dailyStreak.longest})` +
    (freezes > 0 ? ` · ${freezes} freeze${freezes > 1 ? "s" : ""} available` : "");
}

// Adopts the server's copy of this learner's progress as the live state —
// called right after login/signup, and after the initial /auth/me check
// finds an existing session cookie. Overrides whatever loadGamificationState()
// already populated from localStorage, since the account is the more durable
// source once one exists.
function applyAccountProfile(profile) {
  loggedInUsername = profile.username;
  totalXp = profile.total_xp;
  solvedCount = profile.solved_count;
  currentStreak = profile.current_streak;
  bestStreak = profile.best_streak;
  noHintStreak = profile.no_hint_streak;
  noHintTotalCount = profile.no_hint_total_count;
  maxRatingSolved = profile.max_rating_solved;
  gamesPlayed = profile.games_played;
  gamesWon = profile.games_won;
  puzzleCounter = profile.puzzles_served || 0;
  unlockedAchievements.clear();
  for (const id of profile.unlocked_achievements || []) unlockedAchievements.add(id);

  renderXp();
  updateSessionStats();
  renderAchievements();
  renderDailyStreak(profile.daily_streak);
  updateAccountUI();
  applyBoardTheme(getSelectedBoardTheme());
  renderBoardThemePicker();
}

async function checkinDailyStreak() {
  try {
    const response = await fetch("/account/checkin", { method: "POST" });
    if (!response.ok) return;
    const result = await response.json();
    renderDailyStreak({
      current: result.current_streak, longest: result.longest_streak,
      freezes_available: result.freezes_available,
    });
  } catch (err) {
    // Offline/blip — the streak just won't tick for this page load; not fatal.
  }
}

async function checkAuthStatus() {
  try {
    const response = await fetch("/auth/me");
    const data = await response.json();
    if (data.logged_in) {
      applyAccountProfile(data);
      await checkinDailyStreak();
    }
  } catch (err) {
    // Treat a failed check as signed-out; localStorage-only play still works.
  }
}

// ---------------------------------------------------------------------------
// Board themes: cosmetic unlocks tied to progress that's already tracked
// (level, puzzles solved) — a purely visual CSS-variable swap, so unlocking
// one doesn't need a server round-trip; only the SELECTED theme is saved
// (localStorage only, not synced to the account — a per-browser preference,
// not "progress"). Unlock state is recomputed live off totalXp/solvedCount,
// so a newly-unlocked theme shows up the moment those change.
// ---------------------------------------------------------------------------

const BOARD_THEME_STORAGE_KEY = "neurotutor_board_theme";
const BOARD_THEMES = [
  {
    id: "classic", name: "Classic", light: "#f0d9b5", dark: "#b58863", border: "#4a3626",
    unlocked: () => true, hint: "Always available",
  },
  {
    id: "midnight", name: "Midnight", light: "#d7dde8", dark: "#4a5a7a", border: "#232838",
    unlocked: () => currentLevel() >= 5, hint: "Reach level 5",
  },
  {
    id: "forest", name: "Forest", light: "#dce8d0", dark: "#5a7a4a", border: "#2a3a20",
    unlocked: () => currentLevel() >= 10, hint: "Reach level 10",
  },
  {
    id: "coral", name: "Coral", light: "#ffe0d0", dark: "#c97a5a", border: "#5a2f1a",
    unlocked: () => solvedCount >= 100, hint: "Solve 100 puzzles",
  },
];

function currentLevel() {
  return 1 + Math.floor(totalXp / 100);
}

function getSelectedBoardTheme() {
  let saved = "classic";
  try {
    saved = localStorage.getItem(BOARD_THEME_STORAGE_KEY) || "classic";
  } catch (err) {
    // localStorage unavailable — just use the default theme.
  }
  const theme = BOARD_THEMES.find((t) => t.id === saved);
  return theme && theme.unlocked() ? saved : "classic";
}

function applyBoardTheme(themeId) {
  const theme = BOARD_THEMES.find((t) => t.id === themeId) || BOARD_THEMES[0];
  const root = document.documentElement.style;
  root.setProperty("--board-light", theme.light);
  root.setProperty("--board-dark", theme.dark);
  root.setProperty("--board-border", theme.border);
  try {
    localStorage.setItem(BOARD_THEME_STORAGE_KEY, theme.id);
  } catch (err) {
    // Selection just won't survive a refresh; the board itself still re-themes now.
  }
}

function renderBoardThemePicker() {
  const listEl = document.getElementById("board-theme-list");
  if (!listEl) return;
  listEl.innerHTML = "";
  const selected = getSelectedBoardTheme();
  for (const theme of BOARD_THEMES) {
    const unlocked = theme.unlocked();
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "theme-swatch" + (theme.id === selected ? " selected" : "");
    btn.style.setProperty("--swatch-light", theme.light);
    btn.style.setProperty("--swatch-dark", theme.dark);
    btn.disabled = !unlocked;
    btn.title = unlocked ? theme.name : `${theme.name} — ${theme.hint}`;
    btn.setAttribute("aria-label", unlocked ? `Use ${theme.name} board theme` : `${theme.name} locked: ${theme.hint}`);
    btn.addEventListener("click", () => {
      applyBoardTheme(theme.id);
      renderBoardThemePicker();
    });
    listEl.appendChild(btn);
  }
}

// ---------------------------------------------------------------------------
// Leaderboard: public, ranked by total XP — a read-only dialog, no polling;
// it re-fetches every time it's opened so the numbers are always current.
// ---------------------------------------------------------------------------

async function loadLeaderboard() {
  const statusEl = document.getElementById("leaderboard-status");
  const listEl = document.getElementById("leaderboard-list");
  const rankEl = document.getElementById("leaderboard-your-rank");
  statusEl.textContent = "Loading…";
  listEl.innerHTML = "";
  rankEl.textContent = "";
  try {
    const response = await fetch("/leaderboard");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    if (data.entries.length === 0) {
      statusEl.textContent = "No accounts yet — sign up to be the first on the board.";
      return;
    }
    statusEl.textContent = `Top ${data.entries.length} by XP`;
    if (data.your_rank) rankEl.textContent = `— you're #${data.your_rank}`;
    data.entries.forEach((entry, index) => {
      const li = document.createElement("li");
      li.className = "leaderboard-item" + (entry.username === loggedInUsername ? " is-you" : "");
      const rank = document.createElement("span");
      rank.className = "leaderboard-rank";
      rank.textContent = `#${index + 1}`;
      const name = document.createElement("span");
      name.className = "leaderboard-name";
      name.textContent = entry.username;
      const stats = document.createElement("span");
      stats.className = "leaderboard-stats";
      stats.innerHTML =
        `<strong>${entry.total_xp}</strong> XP &middot; ${entry.solved_count} solved &middot; ` +
        `best streak <strong>${entry.best_streak}</strong>`;
      li.append(rank, name, stats);
      listEl.appendChild(li);
    });
  } catch (err) {
    statusEl.textContent = `Could not load leaderboard: ${err.message}`;
  }
}

document.getElementById("leaderboard-button").addEventListener("click", () => {
  document.getElementById("leaderboard-dialog").hidden = false;
  loadLeaderboard();
});
document.getElementById("leaderboard-dialog-close").addEventListener("click", () => {
  document.getElementById("leaderboard-dialog").hidden = true;
});
// Click on the dimmed backdrop (not the card itself) closes the modal.
document.getElementById("leaderboard-dialog").addEventListener("click", (event) => {
  if (event.target.id === "leaderboard-dialog") event.currentTarget.hidden = true;
});

function setAccountError(message) {
  document.getElementById("account-error").textContent = message || "";
}

async function submitAccountForm(endpoint) {
  const username = document.getElementById("account-username").value.trim();
  const password = document.getElementById("account-password").value;
  setAccountError("");
  const payload = { username, password };
  if (endpoint === "/auth/signup") payload.progress = currentProgressPayload();

  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) {
      setAccountError(data.detail || "Something went wrong.");
      return;
    }
    document.getElementById("account-dialog").hidden = true;
    applyAccountProfile(data);
  } catch (err) {
    setAccountError("Network error — try again.");
  }
}

document.getElementById("account-button").addEventListener("click", async () => {
  if (loggedInUsername) {
    await fetch("/auth/logout", { method: "POST" }).catch(() => {});
    loggedInUsername = null;
    updateAccountUI();
    renderDailyStreak(null);
    // Fall back to whatever this browser had saved locally before login.
    loadGamificationState();
    renderXp();
    updateSessionStats();
    renderAchievements();
    applyBoardTheme(getSelectedBoardTheme());
    renderBoardThemePicker();
  } else {
    document.getElementById("account-username").value = "";
    document.getElementById("account-password").value = "";
    setAccountError("");
    document.getElementById("account-dialog").hidden = false;
  }
});
document.getElementById("account-dialog-close").addEventListener("click", () => {
  document.getElementById("account-dialog").hidden = true;
});
// Click on the dimmed backdrop (not the card itself) closes the modal.
document.getElementById("account-dialog").addEventListener("click", (event) => {
  if (event.target.id === "account-dialog") event.currentTarget.hidden = true;
});
document.getElementById("account-login-button").addEventListener("click", () => submitAccountForm("/auth/login"));
document.getElementById("account-signup-button").addEventListener("click", () => submitAccountForm("/auth/signup"));

function computeXpGain(rating, streakAtSolve, usedHint) {
  const base = Math.max(10, Math.round(rating / 25));
  const streakBonus = Math.min(streakAtSolve, 10) * 5;
  return Math.round((base + streakBonus) * (usedHint ? 0.5 : 1));
}

function renderXp(justGained) {
  const level = 1 + Math.floor(totalXp / 100);
  const xpIntoLevel = totalXp % 100;
  document.getElementById("level-badge").textContent = level;
  document.getElementById("xp-fill").style.width = `${xpIntoLevel}%`;
  document.getElementById("xp-value").textContent =
    justGained ? `${xpIntoLevel} / 100 XP (+${justGained} just now)` : `${xpIntoLevel} / 100 XP`;
}

function addXp(amount) {
  totalXp += amount;
  renderXp(amount);
}

function renderAchievements() {
  const listEl = document.getElementById("achievement-list");
  listEl.innerHTML = "";
  for (const a of ACHIEVEMENTS) {
    const unlocked = unlockedAchievements.has(a.id);
    const li = document.createElement("li");
    li.id = `achievement-${a.id}`;
    li.className = unlocked ? "unlocked" : "";
    const strong = document.createElement("strong");
    strong.textContent = a.name;
    const desc = document.createElement("span");
    desc.textContent = unlocked ? a.description : `Locked — ${a.description}`;
    li.append(strong, desc);
    listEl.appendChild(li);
  }
  document.getElementById("achievement-count").textContent = `${unlockedAchievements.size} / ${ACHIEVEMENTS.length}`;
}

function unlockAchievement(id) {
  if (unlockedAchievements.has(id)) return;
  unlockedAchievements.add(id);
  renderAchievements();
  const li = document.getElementById(`achievement-${id}`);
  if (li) {
    li.classList.add("just-unlocked");
    setTimeout(() => li.classList.remove("just-unlocked"), 600);
  }
  playSound("achievement");
}

function unlockIfAtLeast(value, tiers) {
  for (const [threshold, id] of tiers) if (value >= threshold) unlockAchievement(id);
}

function checkSolveAchievements(rating, streakAtSolve, usedHint) {
  if (solvedCount === 1) unlockAchievement("first-blood");
  unlockIfAtLeast(streakAtSolve, [
    [2, "streak-2"], [3, "hat-trick"], [5, "unstoppable"], [7, "streak-7"],
    [10, "streak-10"], [15, "streak-15"], [20, "streak-20"], [25, "streak-25"],
  ]);
  unlockIfAtLeast(solvedCount, [
    [5, "solved-5"], [10, "solved-10"], [25, "solved-25"], [50, "solved-50"],
    [100, "solved-100"], [200, "solved-200"], [500, "solved-500"],
  ]);

  if (lastSubmittedTimeToMove !== null) {
    unlockIfAtLeast(-lastSubmittedTimeToMove, [
      [-10, "speed-10"], [-5, "speed-solver"], [-3, "speed-3"], [-1, "speed-1"],
    ]);
  }

  maxRatingSolved = Math.max(maxRatingSolved, rating);
  unlockIfAtLeast(maxRatingSolved, [
    [1200, "rating-1200"], [1500, "rating-1500"], [1800, "rating-1800"],
    [2000, "grandmaster-in-training"], [2200, "rating-2200"], [2500, "rating-2500"],
  ]);

  if (usedHint) {
    noHintStreak = 0;
  } else {
    unlockAchievement("no-hints");
    noHintStreak += 1;
    noHintTotalCount += 1;
  }
  unlockIfAtLeast(noHintStreak, [[5, "no-hint-streak-5"]]);
  unlockIfAtLeast(noHintTotalCount, [[20, "no-hint-total-20"]]);

  unlockIfAtLeast(totalXp, [[100, "century"]]);
  unlockIfAtLeast(1 + Math.floor(totalXp / 100), [
    [3, "level-3"], [5, "level-5"], [10, "level-10"], [20, "level-20"],
  ]);

  if (lastPredictedState === "Focused") unlockAchievement("zone-focused");
  if (lastPredictedState === "Engaged") unlockAchievement("flow-engaged");
  if (["Overloaded", "Confused", "Fatigued"].includes(lastPredictedState)) unlockAchievement("cool-under-pressure");
  if (wrongAttemptsThisPuzzle >= 1) unlockAchievement("bounce-back");
  if (wrongAttemptsThisPuzzle >= 3) unlockAchievement("comeback-kid");
  if (statesSeenThisSession.size >= 5) unlockAchievement("full-spectrum");

  const hour = new Date().getHours();
  if (hour >= 0 && hour < 4) unlockAchievement("night-owl");

  saveGamificationState();
}

function checkGameOverAchievements(gameResult) {
  gamesPlayed += 1;
  const won = (gameResult === "1-0" && solverColor === "white") || (gameResult === "0-1" && solverColor === "black");
  const drew = gameResult === "1/2-1/2";

  unlockAchievement("first-game");
  if (won) {
    gamesWon += 1;
    unlockAchievement("first-win");
    unlockAchievement(solverColor === "white" ? "win-as-white" : "win-as-black");
  }
  if (drew) unlockAchievement("first-draw");
  unlockIfAtLeast(gamesPlayed, [[5, "games-5"], [10, "games-10"], [25, "games-25"]]);
  unlockIfAtLeast(gamesWon, [[5, "wins-5"], [10, "wins-10"]]);

  saveGamificationState();
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

// whiteWinProb (0..1) comes straight from the server's chess_task.evaluator —
// material-count-based unless a Stockfish binary is installed, in which case
// it's a real engine evaluation converted with the same logistic curve lichess
// uses for its eval bar (see win_probability() in chess_task/evaluator.py).
function setEvalBar(whiteWinProb) {
  const pct = Math.round(whiteWinProb * 100);
  document.getElementById("eval-bar-white").style.height = `${pct}%`;
  document.getElementById("eval-label").textContent =
    pct >= 50 ? `White ${pct}%` : `Black ${100 - pct}%`;
}

function setConnectionStatus(state, label) {
  const dot = document.getElementById("connection-dot");
  dot.classList.remove("live", "error");
  if (state) dot.classList.add(state);
  document.getElementById("connection-label").textContent = label;
}

function describeGameResult(result) {
  if (result === "1-0") return "Checkmate — White wins!";
  if (result === "0-1") return "Checkmate — Black wins!";
  if (result === "1/2-1/2") return "Game drawn.";
  return "Game over.";
}

// ---------------------------------------------------------------------------
// Mode (Puzzle / Full Game) + adaptive policy (rule-based / RL) selection, and
// a client-generated learner id for multi-session personalization: the server
// resumes a returning learner's difficulty (and, under the RL policy, their
// learned Q-table) via storage.db's learner_profiles table.
// ---------------------------------------------------------------------------

let currentMode = "puzzle";

function getLearnerId() {
  try {
    let id = localStorage.getItem("neurotutor_learner_id");
    if (!id) {
      id = (crypto.randomUUID ? crypto.randomUUID() : `learner-${Date.now()}-${Math.random().toString(16).slice(2)}`);
      localStorage.setItem("neurotutor_learner_id", id);
    }
    return id;
  } catch (err) {
    return null; // localStorage unavailable (private mode, etc.) — session just won't personalize.
  }
}

function applyModeLabels(mode) {
  currentMode = mode;
  // Full-game mode reuses the puzzle "rating" field to carry the AI opponent's
  // difficulty (see chess_task.full_game.FullGameEngine.get_puzzle) — showing
  // that as a bare number would just read as an intimidating Elo rating, so
  // it's hidden entirely here rather than relabeled.
  document.getElementById("rating-badge").hidden = mode === "game";
  document.getElementById("counter-badge").hidden = mode === "game";
}

// Set by startTargetedPractice() (the coaching summary's "Practice your weak
// spot" button), consumed once by the next connect() call, then cleared —
// so a later ordinary reconnect doesn't keep silently re-applying it.
let targetedPracticeRange = null;

function startTargetedPractice(ratingMin, ratingMax) {
  targetedPracticeRange = { min: ratingMin, max: ratingMax };
  reconnect();
}

function connect() {
  const mode = document.getElementById("mode-select").value;
  const policy = document.getElementById("policy-select").value;
  const learnerId = getLearnerId();
  const params = new URLSearchParams({ mode, policy });
  if (learnerId) params.set("learner_id", learnerId);
  if (targetedPracticeRange) {
    params.set("target_min", targetedPracticeRange.min);
    params.set("target_max", targetedPracticeRange.max);
    targetedPracticeRange = null;
  }
  ws = new WebSocket(`ws://${window.location.host}/ws/session?${params.toString()}`);
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
      if (msg.mode) applyModeLabels(msg.mode);
      document.getElementById("puzzle-rating").textContent = msg.rating;
      currentPuzzleRating = msg.rating;
      setEvalBar(msg.white_win_prob);
      // Game mode's puzzle_id increments every single ply ("game-ply-0",
      // "game-ply-1", ...) since there's no discrete "puzzle" there — only the
      // FIRST message of a fresh connection should reset the move list/counter,
      // not every move, or the game's move list would wipe itself out each turn.
      const isFreshStart = currentMode === "game" ? currentPuzzleId === null : msg.puzzle_id !== currentPuzzleId;
      if (isFreshStart) {
        hintUsedThisPuzzle = false;
        wrongAttemptsThisPuzzle = 0;
        puzzleCounter += 1;
        document.getElementById("puzzle-counter").textContent = puzzleCounter;
        moveHistory = [];
        renderMoveList();
        lastMove = null;
        document.getElementById("move-explain").hidden = true;
      }
      currentPuzzleId = msg.puzzle_id;
      hintSquare = null;
      document.getElementById("puzzle-hint-box").hidden = true;
      pendingOptimisticMove = null;
      // Feedback deliberately lingers across this reset — it's cleared only when the
      // player actually attempts their next move (see onSquareClick) or requests a
      // fresh puzzle_hint, not the instant the board resets for a retry.
      renderBoard(msg.fen);
      document.getElementById("loading-overlay").hidden = true;
    } else if (msg.type === "hint") {
      if (msg.square === selectedSquare) {
        legalTargets = msg.targets;
        drawBoard();
      }
    } else if (msg.type === "puzzle_hint") {
      hintSquare = msg.square;
      const puzzleHintBox = document.getElementById("puzzle-hint-box");
      puzzleHintBox.hidden = false;
      document.getElementById("puzzle-hint-text").textContent = msg.text;
      bounceMascot();
      drawBoard();
    } else if (msg.type === "invalid_move") {
      legalTargets = msg.targets || legalTargets;
      setCallout(feedbackEl, `Invalid move — ${msg.reason}`, "invalid");
      boardLocked = false; // A rejected move must not leave the board stuck.
      playSound("invalid");
      drawBoard();
    } else if (msg.type === "update") {
      setState(msg.predicted_state);
      lastPredictedState = msg.predicted_state;
      statesSeenThisSession.add(msg.predicted_state);
      setConfidence(msg.confidence);
      document.getElementById("difficulty").textContent = msg.difficulty.toFixed(0);
      updateNetworkDiagram(msg.network_activity);
      setEvalBar(msg.white_win_prob);
      if (msg.action.show_hint) {
        hintBox.hidden = false;
        document.getElementById("hint-text").textContent =
          STATE_HINT_MESSAGES[msg.predicted_state] || DEFAULT_HINT_MESSAGE;
        bounceMascot();
      } else {
        hintBox.hidden = true;
      }

      if (msg.played_san) showMoveExplanation(msg.played_san, "your move");

      if (msg.status === "solved") {
        if (msg.solver_san) moveHistory.push({ san: msg.solver_san, mover: "solver" });
        renderMoveList();
        solvedCount += 1;
        currentStreak += 1;
        bestStreak = Math.max(bestStreak, currentStreak);
        updateSessionStats();
        addXp(computeXpGain(currentPuzzleRating, currentStreak, hintUsedThisPuzzle));
        checkSolveAchievements(currentPuzzleRating, currentStreak, hintUsedThisPuzzle);
        pendingOptimisticMove = null;
        setCallout(feedbackEl, "Correct! Puzzle solved — next one incoming.", "correct");
        popCallout(feedbackEl);
        celebrate();
        playSound("correct");
      } else if (msg.status === "continue" || msg.status === "game_move") {
        if (msg.solver_san) moveHistory.push({ san: msg.solver_san, mover: "solver" });
        if (msg.status === "game_move") renderMoveList();
        pendingOptimisticMove = null;
        // The opponent's reply is already known now (scripted in puzzle mode,
        // live-computed in game mode) — play it out immediately instead of
        // waiting for the next "puzzle" confirmation, so the board doesn't sit
        // frozen through the full pacing delay.
        if (msg.opponent_move) {
          const { from, to, capturedPiece } = applyUciMoveLocally(msg.opponent_move);
          if (msg.status === "continue" && msg.opponent_san) moveHistory.push({ san: msg.opponent_san, mover: "opponent" });
          lastMove = { from, to };
          renderCurrentBoard();
          animateSlideIn(to, from);
          playSound(capturedPiece ? "capture" : "move");
        }
        if (msg.status === "continue") {
          renderMoveList();
          setCallout(feedbackEl, `Correct! Opponent plays ${msg.opponent_san || msg.opponent_move} — keep going.`, "continue");
          popCallout(feedbackEl);
        } else if (msg.correct) {
          setCallout(feedbackEl, "Good move.", "continue");
        } else {
          // Game mode never retries: the move stands, but the player still
          // gets the same "here's why it wasn't best" feedback puzzle mode gives.
          setCalloutWithMoveQuality(
            feedbackEl, msg.reason || "That's legal, but there's a better move.", "incorrect",
            msg.move_score, msg.move_quality_label, msg.move_description, msg.proactive_intervention,
          );
        }
        // The opponent hasn't replied yet — a separate "opponent_move" message
        // reveals it after FullGameEngine's simulated "thinking" pause.
        document.getElementById("thinking-indicator").hidden = !msg.awaiting_opponent;
      } else if (msg.status === "retry") {
        // Wrong (but legal) move in puzzle mode: undo the optimistic move so the
        // piece visibly slides back to where it started, matching the puzzle
        // resetting to its start position — game mode's "game_move" (above)
        // never does this, since a real game move can't be undone.
        if (pendingOptimisticMove) {
          const { from, to, movingPiece, capturedPiece } = pendingOptimisticMove;
          const [fr, ff] = squareToIndices(from);
          const [tr, tf] = squareToIndices(to);
          currentBoard[fr][ff] = movingPiece;
          currentBoard[tr][tf] = capturedPiece;
          pendingOptimisticMove = null;
          lastMove = null;
          renderCurrentBoard();
          animateSlideIn(from, to);
        }
        wrongAttemptsThisPuzzle += 1;
        currentStreak = 0;
        updateSessionStats();
        saveGamificationState();
        setCalloutWithMoveQuality(
          feedbackEl, msg.reason || "There's a better move — try again.", "incorrect",
          msg.move_score, msg.move_quality_label, msg.move_description, msg.proactive_intervention,
        );
        playSound("wrong");
      } else if (msg.status === "game_over") {
        // The player's OWN move ended the game (e.g. delivered checkmate) —
        // there's no opponent reply to wait for.
        pendingOptimisticMove = null;
        document.getElementById("thinking-indicator").hidden = true;
        boardLocked = true;
        checkGameOverAchievements(msg.game_result);
        setCallout(feedbackEl, describeGameResult(msg.game_result), "correct");
        popCallout(feedbackEl);
        celebrate();
      }
    } else if (msg.type === "opponent_move") {
      // The AI opponent's reply, revealed after its simulated "thinking" pause
      // (see FullGameEngine.thinking_time) — a separate message from the
      // player's own move result so that result appears immediately instead of
      // waiting behind this delay.
      document.getElementById("thinking-indicator").hidden = true;
      const { from, to, capturedPiece } = applyUciMoveLocally(msg.move);
      if (msg.san) moveHistory.push({ san: msg.san, mover: "opponent" });
      renderMoveList();
      lastMove = { from, to };
      renderCurrentBoard();
      animateSlideIn(to, from);
      playSound(capturedPiece ? "capture" : "move");
      setEvalBar(msg.white_win_prob);
      if (msg.status === "game_over") {
        boardLocked = true;
        checkGameOverAchievements(msg.game_result);
        setCallout(feedbackEl, describeGameResult(msg.game_result), "correct");
        popCallout(feedbackEl);
        celebrate();
      }
      // Otherwise the feedback callout is deliberately left showing the
      // player's own move result (set moments ago) rather than overwritten
      // with "opponent plays X" — the move list and board already show that.
    } else if (msg.type === "game_over") {
      // Reconnecting to an already-finished game (no move was just made this
      // turn) — same result message, no board/network updates to apply.
      boardLocked = true;
      setCallout(feedbackEl, describeGameResult(msg.result), "correct");
    } else if (msg.type === "error") {
      lastServerError = msg.message;
      setCallout(feedbackEl, msg.message, "error");
      boardLocked = false;
      setConnectionStatus("error", msg.message);
    }
  };
  ws.onopen = () => {
    reconnectAttempts = 0;
    setConnectionStatus("live", "Live");
  };
  ws.onerror = () => {
    setCallout(feedbackEl, "Connection error.", "error");
    setConnectionStatus("error", "Connection error");
  };
  ws.onclose = () => {
    setCallout(
      feedbackEl,
      lastServerError ? `${lastServerError} (connection closed)` : "Connection closed.",
      "error",
    );
    scheduleReconnect();
  };
}

const MAX_RECONNECT_DELAY_MS = 10000;

// A dropped connection (server restart, wifi blip) used to be permanent —
// the only recovery was a manual page reload. Retries with capped
// exponential backoff instead, so the tutor recovers on its own; the
// reconnect gets a brand-new server-side session (session/puzzle state was
// never going to survive a real disconnect), but gamification progress is
// safe in localStorage (see saveGamificationState) regardless.
function scheduleReconnect() {
  if (reconnectTimer) return; // already counting down to one
  reconnectAttempts += 1;
  const delaySeconds = Math.round(Math.min(1000 * 2 ** (reconnectAttempts - 1), MAX_RECONNECT_DELAY_MS) / 1000);
  setConnectionStatus("error", `Disconnected — reconnecting in ${delaySeconds}s…`);
  reconnectTimer = setTimeout(() => {
    reconnectTimer = null;
    currentPuzzleId = null; // the new session's first puzzle must be treated as fresh, not a retry
    connect();
  }, delaySeconds * 1000);
}

function reconnect() {
  // Switching mode/policy starts a genuinely new server-side TutorSession, so
  // the board/puzzle state must reset — but gamification stats (XP, streak,
  // achievements) are treated as belonging to the learner across the whole
  // page session, not any one server connection, so they're left alone.
  if (ws) {
    ws.onclose = null; // this is a deliberate reconnect, not a real disconnect
    ws.close();
  }
  // Cancel any auto-reconnect already counting down from a real drop, so a
  // manual mode/policy switch can't race it into a duplicate connection.
  if (reconnectTimer) {
    clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }
  reconnectAttempts = 0;
  document.getElementById("loading-overlay").hidden = false;
  currentPuzzleId = null; // forces the next "puzzle" message to be treated as new
  connect();
}

document.getElementById("summary-button").addEventListener("click", showSummary);
document.getElementById("flip-button").addEventListener("click", toggleFlip);
document.getElementById("hint-button").addEventListener("click", requestHint);
document.getElementById("sound-toggle").addEventListener("click", toggleSound);
document.getElementById("mode-select").addEventListener("change", reconnect);
document.getElementById("policy-select").addEventListener("change", reconnect);

// Delegated on #board itself (not per-square) so it survives every
// renderCurrentBoard() re-render, which throws all the square elements away
// and rebuilds them from scratch. Enter/Space mirrors a click; arrow keys
// move focus one square at a time in whatever orientation is currently
// displayed (DOM order already matches the visual grid, flipped or not).
document.getElementById("board").addEventListener("keydown", (event) => {
  const squareEl = event.target.closest(".square");
  if (!squareEl) return;
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    onSquareClick(squareEl.dataset.square);
    return;
  }
  const ARROW_DELTAS = { ArrowRight: 1, ArrowLeft: -1, ArrowDown: 8, ArrowUp: -8 };
  const delta = ARROW_DELTAS[event.key];
  if (delta === undefined) return;
  event.preventDefault();
  const squares = Array.from(document.querySelectorAll("#board .square"));
  const currentIndex = squares.indexOf(squareEl);
  const nextIndex = currentIndex + delta;
  if (nextIndex < 0 || nextIndex >= squares.length) return;
  squares[currentIndex].tabIndex = -1;
  squares[nextIndex].tabIndex = 0;
  boardFocusIndex = nextIndex;
  squares[nextIndex].focus();
});

// Segmented button groups are a progressive-enhancement skin over the hidden
// mode-select/policy-select — clicking a segment just updates the real
// <select>'s value and fires "change" on it, so reconnect() and everything
// else that reads those selects needs no changes at all.
document.querySelectorAll(".segmented").forEach((group) => {
  const select = document.getElementById(group.dataset.select);
  group.querySelectorAll(".segmented-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      if (btn.classList.contains("active")) return;
      group.querySelectorAll(".segmented-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      select.value = btn.dataset.value;
      select.dispatchEvent(new Event("change"));
    });
  });
});

// Sticky header grows a subtle border/shadow once the page has actually
// scrolled underneath it, instead of always showing one.
const appHeaderEl = document.querySelector(".app-header");
if (appHeaderEl) {
  window.addEventListener("scroll", () => {
    appHeaderEl.classList.toggle("scrolled", window.scrollY > 4);
  }, { passive: true });
}

// A one-time nudge toward /learn for brand-new visitors — never shown again
// once they've completed the tutorial or dismissed the banner themselves.
function maybeShowOnboardingBanner() {
  try {
    if (localStorage.getItem("neurotutor_onboarding_completed")) return;
    if (localStorage.getItem("neurotutor_onboarding_banner_dismissed")) return;
  } catch (err) {
    return; // No localStorage available — err on the side of not nagging.
  }
  document.getElementById("onboarding-banner").hidden = false;
}

document.getElementById("onboarding-banner-dismiss").addEventListener("click", () => {
  document.getElementById("onboarding-banner").hidden = true;
  try {
    localStorage.setItem("neurotutor_onboarding_banner_dismissed", "1");
  } catch (err) {
    // Not persisted — the banner may reappear next visit; harmless.
  }
});

maybeShowOnboardingBanner();

loadGamificationState();
renderXp();
updateSessionStats();
renderAchievements();
applyBoardTheme(getSelectedBoardTheme());
renderBoardThemePicker();
// checkAuthStatus() must resolve (and apply the account's puzzleCounter, if
// any) BEFORE connect() opens the websocket — the first "puzzle" message
// increments puzzleCounter immediately, and if that race lost (fetch slower
// than the local websocket), applyAccountProfile() would stomp the fresh
// increment back down to the account's last-synced value right after.
(async () => {
  await checkAuthStatus();
  connect();
})();
