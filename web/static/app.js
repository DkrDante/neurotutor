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

function pieceAt(square) {
  if (!currentBoard) return null;
  const [rankIdx, fileIdx] = squareToIndices(square);
  return currentBoard[rankIdx][fileIdx];
}

function isOwnPiece(piece) {
  // Every puzzle in this build is White-to-move, so "own" pieces are the uppercase ones.
  return !!piece && piece === piece.toUpperCase();
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
    if ((piece === "P" && destRank === "8")) {
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

    if (msg.type === "puzzle") {
      currentAttemptToken = msg.attempt_token;
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
      if (msg.action.show_hint) {
        hintBox.hidden = false;
        document.getElementById("hint-text").textContent =
          STATE_HINT_MESSAGES[msg.predicted_state] || DEFAULT_HINT_MESSAGE;
      } else {
        hintBox.hidden = true;
      }
      setCallout(
        feedbackEl,
        msg.correct ? "Correct! Next puzzle incoming." : "Not quite — next puzzle incoming.",
        msg.correct ? "correct" : "incorrect",
      );
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
