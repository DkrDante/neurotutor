const PIECE_UNICODE = {
  K: "♔", Q: "♕", R: "♖", B: "♗", N: "♘", P: "♙",
  k: "♚", q: "♛", r: "♜", b: "♝", n: "♞", p: "♟",
};

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

let selectedSquare = null;
let attemptStartMs = null;
let ws = null;
let lastServerError = null;

function clearSelection() {
  selectedSquare = null;
  document.querySelectorAll(".square.selected").forEach((el) => el.classList.remove("selected"));
}

function renderBoard(fen) {
  const board = parseFen(fen);
  const boardEl = document.getElementById("board");
  boardEl.innerHTML = "";
  for (let r = 0; r < 8; r++) {
    for (let f = 0; f < 8; f++) {
      const square = document.createElement("div");
      const name = squareName(r, f);
      square.className = "square " + ((r + f) % 2 === 0 ? "light" : "dark");
      square.dataset.square = name;
      const piece = board[r][f];
      if (piece) square.textContent = PIECE_UNICODE[piece];
      square.addEventListener("click", () => onSquareClick(name, square));
      boardEl.appendChild(square);
    }
  }
  clearSelection();
  attemptStartMs = performance.now();
}

function onSquareClick(square, el) {
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
  const moveUci = selectedSquare + square;
  const timeToMove = (performance.now() - attemptStartMs) / 1000.0;
  ws.send(JSON.stringify({ move_uci: moveUci, time_to_move: timeToMove }));
  clearSelection();
}

function connect() {
  ws = new WebSocket(`ws://${window.location.host}/ws/session`);
  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.type === "puzzle") {
      renderBoard(msg.fen);
      document.getElementById("feedback").textContent = "";
    } else if (msg.type === "update") {
      document.getElementById("state").textContent = msg.predicted_state;
      document.getElementById("confidence").textContent = msg.confidence.toFixed(2);
      document.getElementById("difficulty").textContent = msg.difficulty.toFixed(0);
      document.getElementById("hint").hidden = !msg.action.show_hint;
      document.getElementById("feedback").textContent = msg.correct ? "Correct!" : "Not quite — next puzzle incoming.";
    } else if (msg.type === "error") {
      lastServerError = msg.message;
      document.getElementById("feedback").textContent = msg.message;
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

connect();
