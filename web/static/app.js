const PIECE_UNICODE = {
  K: "♔", Q: "♕", R: "♖", B: "♗", N: "♘", P: "♙",
  k: "♚", q: "♛", r: "♜", b: "♝", n: "♞", p: "♟",
};

const HINT_MESSAGES = {
  Overloaded: "Hint: you look overloaded — slow down and re-scan the whole board before moving.",
  Confused: "Hint: look for checks, captures, and threats one at a time.",
  Fatigued: "Hint: fatigue detected — a short breather might help before the next puzzle.",
};
const DEFAULT_HINT_MESSAGE = "Hint: take your time and re-check the position.";

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
let attemptStartMs = null;
let ws = null;
let sessionId = null;
let lastServerError = null;
let currentAttemptToken = null;
let boardLocked = false;
let currentFen = null;
let currentBoard = null;
let boardFlipped = false;

function clearSelection() {
  selectedSquare = null;
  document.querySelectorAll(".square.selected").forEach((el) => el.classList.remove("selected"));
}

// Renders the current board state without touching per-puzzle bookkeeping (selection,
// attempt timer, lock) — used both for a fresh puzzle and for a pure flip-orientation redraw.
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
      square.className = "square " + ((r + f) % 2 === 0 ? "light" : "dark");
      square.dataset.square = name;
      const piece = currentBoard[r][f];
      if (piece) square.textContent = PIECE_UNICODE[piece];
      if (name === selectedSquare) square.classList.add("selected");
      square.addEventListener("click", () => onSquareClick(name, square));
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
  drawBoard();
  clearSelection();
  attemptStartMs = performance.now();
  // A new puzzle round has started: the board is playable again, and any move sent
  // against the previous round's token is now stale and will be rejected server-side.
  boardLocked = false;
}

function toggleFlip() {
  boardFlipped = !boardFlipped;
  if (currentFen) drawBoard();
}

function onSquareClick(square, el) {
  if (boardLocked) return; // Waiting on the server (pacing delay / next puzzle).
  if (!selectedSquare) {
    selectedSquare = square;
    el.classList.add("selected");
    return;
  }
  if (square === selectedSquare) {
    // Clicking the same square again means "deselect", not the null move a1a1.
    clearSelection();
    return;
  }
  let moveUci = selectedSquare + square;
  const [rankIdx, fileIdx] = squareToIndices(selectedSquare);
  const piece = currentBoard[rankIdx][fileIdx];
  const destRank = square[1];
  if ((piece === "P" && destRank === "8") || (piece === "p" && destRank === "1")) {
    moveUci += "q"; // Auto-queen; the frontend has no promotion-choice UI.
  }
  const timeToMove = (performance.now() - attemptStartMs) / 1000.0;
  ws.send(JSON.stringify({
    move_uci: moveUci,
    time_to_move: timeToMove,
    attempt_token: currentAttemptToken,
  }));
  clearSelection();
  boardLocked = true; // Locked until the next "puzzle" message arrives.
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
      `Attempts: ${s.num_attempts} | Accuracy: ${(s.accuracy_rate * 100).toFixed(0)}% | ` +
      `Avg latency: ${s.avg_latency.toFixed(3)}s | States: ${s.state_trend.join(", ") || "-"}`;
  } catch (err) {
    summaryEl.textContent = `Could not load summary: ${err.message}`;
  }
}

function setConfidence(confidence) {
  document.getElementById("confidence").textContent = confidence.toFixed(2);
  const fillEl = document.getElementById("confidence-fill");
  fillEl.style.width = `${Math.round(confidence * 100)}%`;
}

function connect() {
  ws = new WebSocket(`ws://${window.location.host}/ws/session`);
  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.session_id) sessionId = msg.session_id;
    if (msg.type === "puzzle") {
      currentAttemptToken = msg.attempt_token;
      renderBoard(msg.fen);
      document.getElementById("feedback").textContent = "";
    } else if (msg.type === "update") {
      document.getElementById("state").textContent = msg.predicted_state;
      setConfidence(msg.confidence);
      document.getElementById("difficulty").textContent = msg.difficulty.toFixed(0);
      const hintEl = document.getElementById("hint");
      hintEl.hidden = !msg.action.show_hint;
      if (msg.action.show_hint) {
        hintEl.textContent = HINT_MESSAGES[msg.predicted_state] || DEFAULT_HINT_MESSAGE;
      }
      document.getElementById("feedback").textContent = msg.correct ? "Correct!" : "Not quite — next puzzle incoming.";
    } else if (msg.type === "error") {
      lastServerError = msg.message;
      document.getElementById("feedback").textContent = msg.message;
      boardLocked = false; // A rejected/malformed move must not leave the board stuck.
    }
  };
  ws.onerror = () => {
    document.getElementById("feedback").textContent = "Connection error.";
  };
  ws.onclose = () => {
    // Keep a server-sent explanation visible rather than replacing it with a generic notice.
    document.getElementById("feedback").textContent = lastServerError
      ? `${lastServerError} (connection closed)`
      : "Connection closed.";
  };
}

document.getElementById("summary-button").addEventListener("click", showSummary);
document.getElementById("flip-button").addEventListener("click", toggleFlip);
connect();
