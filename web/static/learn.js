// ---------------------------------------------------------------------------
// Learn to Play — a self-contained, backend-independent onboarding tutorial.
// Deliberately separate from app.js: this is fixed, curated teaching content
// (hand-picked positions with hand-picked correct squares), not a puzzle the
// adaptive engine serves — it doesn't need a session, a model, or a server
// round-trip at all, so it can't be affected by (or accidentally affect) the
// real tutoring session's state.
// ---------------------------------------------------------------------------

const PIECE_GLYPH = { k: "♚", q: "♛", r: "♜", b: "♝", n: "♞", p: "♟" };
const FILES = "abcdefgh";
const ONBOARDING_DONE_KEY = "neurotutor_onboarding_completed";

function squareName(rankIdx, fileIdx) {
  return `${FILES[fileIdx]}${8 - rankIdx}`;
}

// board[0] = rank 8 ... board[7] = rank 1 — same convention as app.js's
// parseFen, so a lesson FEN reads exactly the way it looks on paper.
function parseFen(fen) {
  const board = [];
  for (const row of fen.split(" ")[0].split("/")) {
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
  return board;
}

function isWhitePiece(piece) {
  return piece === piece.toUpperCase();
}

function squareIndices(square) {
  return [8 - Number(square[1]), FILES.indexOf(square[0])];
}

// ---------------------------------------------------------------------------
// Lesson content. Every FEN + target-square list below was verified against
// python-chess's real legal-move generator (see the position-design step of
// this feature) — these aren't guessed squares, they're the actual legal
// destinations for that piece in that position.
// ---------------------------------------------------------------------------

const PIECE_LESSONS = [
  {
    piece: "p", name: "Pawn",
    fen: "8/8/8/8/8/3p1p2/4P3/8 w - - 0 1", pieceSquare: "e2",
    targets: ["d3", "e3", "e4", "f3"],
    description: "Pawns move straight ahead one square — or two, on their very first move. " +
      "They can never move backward, and they can never capture straight ahead. " +
      "The only way a pawn captures is diagonally, one square ahead.",
    successText: "Exactly right — forward to move, diagonal to capture.",
    failText: "Pawns only move straight ahead (one or two squares from here) or capture diagonally — try one of the highlighted squares.",
    learnMore: "A pawn that reaches the far end of the board gets \"promoted\" — traded for a queen, rook, bishop, or knight of the same color.",
  },
  {
    piece: "n", name: "Knight",
    fen: "8/8/8/8/3N4/8/8/8 w - - 0 1", pieceSquare: "d4",
    targets: ["b3", "b5", "c2", "c6", "e2", "e6", "f3", "f5"],
    description: "Knights move in an L-shape: two squares in one direction, then one square to the side. " +
      "They're the only piece that can jump clean over anything in the way.",
    successText: "That's the L-shape! Knights are the only piece that can hop over others.",
    failText: "Knights move in an L-shape — two squares one way, then one square to the side. Try a highlighted square.",
    learnMore: "Because they jump, knights are especially strong in crowded positions where other pieces are boxed in.",
  },
  {
    piece: "b", name: "Bishop",
    fen: "8/8/8/8/3B4/8/8/8 w - - 0 1", pieceSquare: "d4",
    targets: ["a1", "a7", "b2", "b6", "c3", "c5", "e3", "e5", "f2", "f6", "g1", "g7", "h8"],
    description: "Bishops slide diagonally, any number of squares. Each bishop is stuck on one color for the " +
      "whole game — this one only ever reaches light squares.",
    successText: "Nice slide — straight down the diagonal.",
    failText: "Bishops only move diagonally — try a highlighted square along one of the two diagonals.",
    learnMore: "Because each bishop only reaches half the board's squares, your two bishops working together cover far more ground than either alone.",
  },
  {
    piece: "r", name: "Rook",
    fen: "8/8/8/8/3R4/8/8/8 w - - 0 1", pieceSquare: "d4",
    targets: ["a4", "b4", "c4", "d1", "d2", "d3", "d5", "d6", "d7", "d8", "e4", "f4", "g4", "h4"],
    description: "Rooks slide in straight lines — forward, backward, or sideways, any number of squares — but " +
      "never diagonally.",
    successText: "Straight line, just like that.",
    failText: "Rooks only move in straight lines (up/down/sideways), never diagonally — try a highlighted square.",
    learnMore: "Rooks are especially strong on open files (columns with no pawns in the way) and deep on the opponent's second rank.",
  },
  {
    piece: "q", name: "Queen",
    fen: "8/8/8/8/3Q4/8/8/8 w - - 0 1", pieceSquare: "d4",
    targets: [
      "a1", "a4", "a7", "b2", "b4", "b6", "c3", "c4", "c5", "d1", "d2", "d3", "d5", "d6", "d7", "d8",
      "e3", "e4", "e5", "f2", "f4", "f6", "g1", "g4", "g7", "h4", "h8",
    ],
    description: "The queen combines the rook and the bishop — straight lines or diagonals, any distance. " +
      "She's the most powerful piece on the board.",
    successText: "That's the queen's full range on display.",
    failText: "The queen moves like a rook AND a bishop combined — straight or diagonal, any distance. Try a highlighted square.",
    learnMore: "Because she's so valuable, bringing your queen out too early is often risky — she can be chased around by cheaper pieces, costing you time.",
  },
  {
    piece: "k", name: "King",
    fen: "8/8/8/8/3K4/8/8/8 w - - 0 1", pieceSquare: "d4",
    targets: ["c3", "c4", "c5", "d3", "d5", "e3", "e4", "e5"],
    description: "The king moves just one square in any direction. He isn't powerful, but protecting him is the " +
      "entire point of the game.",
    successText: "One careful step at a time — that's the king.",
    failText: "The king only moves one square in any direction — try a highlighted square right next to him.",
    learnMore: "If your king is under attack, that's \"check\" — you must deal with it immediately. If you can't, that's \"checkmate\" and the game is over.",
  },
];

const PRACTICE_PUZZLES = [
  {
    fen: "8/8/8/2p5/8/3N4/8/8 w - - 0 1", pieceSquare: "d3", targets: ["c5"],
    prompt: "Capture the pawn with your knight.",
    successText: "Perfect — same L-shaped move, just landing on an enemy piece.",
    failText: "Remember the knight's L-shape — try again.",
  },
  {
    fen: "8/8/6r1/8/8/3B4/8/8 w - - 0 1", pieceSquare: "d3", targets: ["g6"],
    prompt: "Attack the rook with your bishop.",
    successText: "You've got it — straight down the diagonal to the rook.",
    failText: "The bishop only moves diagonally — look for the diagonal that reaches the rook.",
  },
  {
    fen: "7k/5ppp/8/8/8/8/8/3QK3 w - - 0 1", pieceSquare: "d1", targets: ["d8"],
    prompt: "The black king is boxed in by its own pawns — deliver checkmate with your queen.",
    successText: "Checkmate! The king had nowhere to run — that's the whole goal of the game.",
    failText: "Look for a square where your queen attacks the king along a line with nothing in the way.",
  },
];

// ---------------------------------------------------------------------------
// Notation, special moves, check/checkmate/draws, and named checkmate
// patterns — same "verified against python-chess's real legal-move
// generator" standard as PIECE_LESSONS above, extended with two optional
// fields attemptMove() understands: `extraMove` (a second from/to the board
// applies at the same time — castling's rook hop) and `promoteTo` (the
// moved piece's letter changes on arrival — pawn promotion).
// ---------------------------------------------------------------------------

const NOTATION_LESSON = {
  fen: "8/8/8/8/8/8/4P3/8 w - - 0 1", pieceSquare: "e2", targets: ["e4"],
  successText: "That's it — in notation, this move is simply written “e4”: just the destination square.",
  failText: "Try moving the pawn two squares ahead, to e4 — the move notation is named for.",
};

const CASTLING_LESSON = {
  fen: "4k3/8/8/8/8/8/8/4K2R w K - 0 1", pieceSquare: "e1", targets: ["g1"],
  extraMove: { from: "h1", to: "f1" },
  successText: "That's castling — the king hopped two squares toward the rook, and the rook jumped right over to sit beside it.",
  failText: "To castle kingside, move the king two squares toward its rook — try g1.",
};

const EN_PASSANT_LESSON = {
  fen: "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", pieceSquare: "e5", targets: ["d6"],
  enPassantCapture: "d5",
  successText: "That's en passant — you captured the pawn as though it had only moved one square.",
  failText: "Black's pawn just jumped two squares past yours — capture it as if it had only moved one, landing on d6.",
};

const PROMOTION_LESSON = {
  fen: "7k/4P3/8/8/8/8/8/4K3 w - - 0 1", pieceSquare: "e7", targets: ["e8"],
  promoteTo: "q",
  successText: "Promoted to a queen! Most players promote to a queen almost every time, since she's the strongest piece.",
  failText: "Push the pawn all the way to the last rank, e8, to promote it.",
};

const CHECK_ESCAPE_LESSON = {
  fen: "4r3/8/8/8/8/8/8/4K3 w - - 0 1", pieceSquare: "e1", targets: ["d1", "d2", "f1", "f2"],
  successText: "Safe! Moving off the e-file gets you out of check.",
  failText: "The rook attacks your king along the whole e-file — move the king off of it.",
};

// Illustrative only (rendered non-interactively) — a textbook stalemate: no
// legal move for the side to move, but NOT in check, so it's a draw rather
// than a loss.
const STALEMATE_FEN = "7k/5K2/6Q1/8/8/8/8/8 b - - 0 1";

const CHECKMATE_PATTERNS = [
  {
    name: "Back-rank mate",
    fen: "7k/5ppp/8/8/8/8/8/3QK3 w - - 0 1", pieceSquare: "d1", targets: ["d8"],
    explanation: "The black king is trapped behind its own pawns, with no square to step onto along the back rank.",
    successText: "Checkmate! The king's own pawns walled it in — it had no escape.",
    failText: "Find the move that checks the king along the back rank, where it has no escape.",
  },
  {
    name: "Ladder (two-rook) mate",
    fen: "7k/R7/8/8/8/8/8/1R5K w - - 0 1", pieceSquare: "b1", targets: ["b8"],
    explanation: "One rook already seals off an entire rank so the king can't step onto it; the other delivers mate along the back rank.",
    successText: "Checkmate! Your rook on a7 sealed off the 7th rank, so the king had nowhere to run.",
    failText: "Your rook on a7 already seals off the 7th rank — bring the other rook down the file to check the king on the back rank.",
  },
  {
    name: "Smothered mate",
    fen: "5rkr/5ppp/2N5/8/8/8/8/7K w - - 0 1", pieceSquare: "c6", targets: ["e7"],
    explanation: "The king is boxed in entirely by its own pieces — a single knight check is enough, since there's no square left to run to.",
    successText: "Checkmate! The king was smothered by its own pieces — it couldn't move anywhere, even to escape a knight.",
    failText: "The king is boxed in by its own pieces — find the knight move that delivers check.",
  },
];

// Cumulative reveal stages for the "how the board is set up" step — each
// stage lists every square that should be showing PIECES by that point.
const SETUP_STAGES = [
  { caption: "We start with an empty board.", squares: [] },
  {
    caption: "Each player's 8 pawns line up on their own second rank.",
    squares: ["a2", "b2", "c2", "d2", "e2", "f2", "g2", "h2", "a7", "b7", "c7", "d7", "e7", "f7", "g7", "h7"],
  },
  {
    caption: "Rooks take the corners, with a knight and then a bishop next to each one.",
    squares: [
      "a2", "b2", "c2", "d2", "e2", "f2", "g2", "h2", "a7", "b7", "c7", "d7", "e7", "f7", "g7", "h7",
      "a1", "b1", "c1", "f1", "g1", "h1", "a8", "b8", "c8", "f8", "g8", "h8",
    ],
  },
  {
    caption: "The queen goes on her own color (white queen on a light square), and the king fills the last spot.",
    squares: [
      "a2", "b2", "c2", "d2", "e2", "f2", "g2", "h2", "a7", "b7", "c7", "d7", "e7", "f7", "g7", "h7",
      "a1", "b1", "c1", "d1", "e1", "f1", "g1", "h1", "a8", "b8", "c8", "d8", "e8", "f8", "g8", "h8",
    ],
  },
];
const START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1";

// ---------------------------------------------------------------------------
// Step sequence. "piece" steps pull from PIECE_LESSONS by index; "practice"
// cycles through PRACTICE_PUZZLES; "welcome"/"setup"/"done" are non-graded.
// ---------------------------------------------------------------------------

const STEPS = [
  { type: "welcome", label: "Welcome" },
  { type: "setup", label: "Board setup" },
  ...PIECE_LESSONS.map((lesson) => ({ type: "piece", label: lesson.name, lesson })),
  { type: "notation", label: "Notation", lesson: NOTATION_LESSON },
  { type: "castling", label: "Castling", lesson: CASTLING_LESSON },
  { type: "enpassant", label: "En passant", lesson: EN_PASSANT_LESSON },
  { type: "promotion", label: "Promotion", lesson: PROMOTION_LESSON },
  { type: "checkmate", label: "Check & checkmate", lesson: CHECK_ESCAPE_LESSON },
  { type: "draws", label: "Draws" },
  { type: "patterns", label: "Checkmate patterns" },
  { type: "practice", label: "Practice" },
  { type: "done", label: "All done" },
];

let stepIndex = 0;
let practicePuzzleIndex = 0;
let patternIndex = 0;
const solvedSteps = new Set(); // step indices the learner has successfully completed
let liveBoard = null; // mutable board array for whichever lesson is currently shown
let selectedForClick = false;
let setupStageIndex = 0;
let setupTimer = null;

function currentStep() {
  return STEPS[stepIndex];
}

// The single source of truth for "what's the interactive lesson right now" —
// unifies the two shapes (a fixed PIECE_LESSONS entry vs. the active practice
// puzzle) behind one object so attemptMove() doesn't need to care which.
function activeLessonData() {
  const step = currentStep();
  if (step.type === "practice") return PRACTICE_PUZZLES[practicePuzzleIndex];
  if (step.type === "patterns") return CHECKMATE_PATTERNS[patternIndex];
  if (step.lesson) return step.lesson; // piece, notation, castling, enpassant, promotion, checkmate (escape)
  return null;
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

function renderProgress() {
  const listEl = document.getElementById("learn-progress");
  listEl.innerHTML = "";
  STEPS.forEach((step, index) => {
    const li = document.createElement("li");
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "learn-progress-dot";
    if (index === stepIndex) btn.classList.add("current");
    if (solvedSteps.has(index) || index < stepIndex) btn.classList.add("done");
    btn.title = step.label;
    btn.setAttribute("aria-label", `${step.label}${index === stepIndex ? " (current step)" : ""}`);
    btn.addEventListener("click", () => goToStep(index));
    li.appendChild(btn);
    listEl.appendChild(li);
  });
}

function renderLearnMore(text) {
  const wrap = document.getElementById("learn-more-wrap");
  const toggle = document.getElementById("learn-more-toggle");
  const textEl = document.getElementById("learn-more-text");
  if (!text) {
    wrap.hidden = true;
    return;
  }
  wrap.hidden = false;
  textEl.textContent = text;
  textEl.hidden = true;
  toggle.textContent = "Learn more ↓";
  toggle.onclick = () => {
    const isHidden = textEl.hidden;
    textEl.hidden = !isHidden;
    toggle.textContent = isHidden ? "Learn more ↑" : "Learn more ↓";
  };
}

function showLearnFeedback(text, variant) {
  const el = document.getElementById("learn-feedback");
  el.hidden = !text;
  el.className = "callout learn-feedback" + (variant ? ` callout-${variant}` : "");
  el.textContent = text || "";
}

function renderBoard(fen, opts) {
  liveBoard = parseFen(fen);
  renderLiveBoard(opts || {});
}

// Re-renders from `liveBoard` (already-parsed/mutated) rather than re-parsing
// a FEN — used both for the initial draw and after a successful move, so a
// solved lesson visibly shows the piece having actually moved.
function renderLiveBoard(opts) {
  const boardEl = document.getElementById("learn-board");
  boardEl.innerHTML = "";
  const targets = opts.targets || [];
  const pieceSquare = opts.pieceSquare;
  const interactive = !!opts.interactive;

  for (let rankIdx = 0; rankIdx < 8; rankIdx++) {
    for (let fileIdx = 0; fileIdx < 8; fileIdx++) {
      const name = squareName(rankIdx, fileIdx);
      const square = document.createElement("div");
      const classes = ["square", (rankIdx + fileIdx) % 2 === 0 ? "light" : "dark"];
      const piece = liveBoard[rankIdx][fileIdx];
      const isTarget = interactive && targets.includes(name) && selectedForClick;
      if (isTarget) {
        classes.push("legal-target");
        if (piece) classes.push("capture");
      }
      if (name === opts.justMoved) classes.push("last-move");
      square.className = classes.join(" ");
      square.dataset.square = name;

      if (piece) {
        const glyph = document.createElement("span");
        glyph.className = "piece " + (isWhitePiece(piece) ? "piece-white" : "piece-black");
        glyph.textContent = PIECE_GLYPH[piece.toLowerCase()];
        square.appendChild(glyph);
        if (interactive && name === pieceSquare) {
          glyph.classList.add("learn-hero-piece");
          glyph.addEventListener("pointerdown", onHeroPointerDown);
          square.tabIndex = 0;
        }
      }
      if (interactive && (name === pieceSquare || targets.includes(name))) {
        square.tabIndex = 0;
      }
      if (interactive) {
        square.addEventListener("click", () => onSquareClick(name));
        square.addEventListener("keydown", (event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            onSquareClick(name);
          }
        });
      }
      boardEl.appendChild(square);
    }
  }
}

function renderStep() {
  clearTimeout(setupTimer);
  selectedForClick = false;
  const step = currentStep();
  const contentEl = document.getElementById("learn-content");
  const boardRow = document.querySelector(".learn-board-row");
  showLearnFeedback("", null);
  renderLearnMore(null);
  document.getElementById("learn-replay").hidden = true;

  if (step.type === "welcome") {
    boardRow.hidden = true;
    contentEl.innerHTML = `
      <h2 class="panel-title">Welcome to chess!</h2>
      <p class="learn-body">
        Chess is a two-player strategy game played on an 8&times;8 board. Each of your 6 piece types moves
        differently, and the goal is <strong>checkmate</strong> — trapping your opponent's king so it can't
        escape capture.
      </p>
      <p class="learn-body">
        This short walkthrough introduces every piece, shows you how the board is set up, and lets you try
        a few real moves yourself before you jump into independent puzzles. Take your time — you can repeat
        any step as often as you like.
      </p>`;
  } else if (step.type === "setup") {
    boardRow.hidden = false;
    contentEl.innerHTML = `
      <h2 class="panel-title">How the board is set up</h2>
      <p class="learn-body" id="setup-caption"></p>`;
    document.getElementById("learn-replay").hidden = false;
    playSetupAnimation();
  } else if (step.type === "piece") {
    boardRow.hidden = false;
    const lesson = step.lesson;
    contentEl.innerHTML = `
      <h2 class="panel-title">${lesson.name}</h2>
      <p class="learn-body">${lesson.description}</p>
      <p class="learn-hint">Click or tap the ${lesson.name.toLowerCase()}, then a highlighted square to try
        the move — or drag it there directly.</p>`;
    renderLearnMore(lesson.learnMore);
    renderBoard(lesson.fen, { pieceSquare: lesson.pieceSquare, targets: lesson.targets, interactive: true });
  } else if (step.type === "notation") {
    boardRow.hidden = false;
    contentEl.innerHTML = `
      <h2 class="panel-title">Reading &amp; writing moves</h2>
      <p class="learn-body">
        Every square has a unique name: the file (column) letter <strong>a</strong>-<strong>h</strong>
        first, then the rank (row) number <strong>1</strong>-<strong>8</strong> — so "e4" is unambiguous.
      </p>
      <ul class="learn-notation-list">
        <li><strong>K Q R B N</strong> — King, Queen, Rook, Bishop, Knight. A pawn move has no letter at all — just the destination square, like "e4".</li>
        <li><strong>x</strong> — a capture, e.g. "Nxe5" (knight captures on e5).</li>
        <li><strong>+</strong> and <strong>#</strong> — check and checkmate.</li>
        <li><strong>O-O</strong> and <strong>O-O-O</strong> — castling kingside and queenside.</li>
        <li><strong>=Q</strong> — promotion, written after the pawn's move, e.g. "e8=Q".</li>
      </ul>
      <p class="learn-hint">Try the move that opens more games than any other:</p>`;
    renderBoard(step.lesson.fen, { pieceSquare: step.lesson.pieceSquare, targets: step.lesson.targets, interactive: true });
  } else if (step.type === "castling") {
    boardRow.hidden = false;
    contentEl.innerHTML = `
      <h2 class="panel-title">Castling</h2>
      <p class="learn-body">
        Castling is the only move where two pieces move at once: the king slides two squares toward a rook,
        and that rook hops to the square right next to the king. It's the one time your king can move more
        than one square — usually to tuck it away safely and bring a rook into play.
      </p>
      <p class="learn-hint">You can only castle if neither piece has moved yet, nothing sits between them,
        and the king isn't in check or passing through check. Try it — drag or click the king two squares
        toward its rook.</p>`;
    renderLearnMore("Castling toward the h-file rook (shown here) is kingside castling, written O-O. Castling toward the a-file rook is queenside castling, written O-O-O.");
    renderBoard(step.lesson.fen, { pieceSquare: step.lesson.pieceSquare, targets: step.lesson.targets, interactive: true });
  } else if (step.type === "enpassant") {
    boardRow.hidden = false;
    contentEl.innerHTML = `
      <h2 class="panel-title">En passant</h2>
      <p class="learn-body">
        "En passant" ("in passing") is a special pawn capture. If an enemy pawn moves two squares and lands
        right beside yours, you can capture it as though it had only moved one square — but only right away,
        on your very next move. Miss the chance and it's gone for good.
      </p>
      <p class="learn-hint">Black's pawn just jumped from d7 to d5, landing beside yours. Capture it en passant.</p>`;
    renderLearnMore("This rule exists so a pawn can't dodge a capture just by sprinting two squares past it — without it, the first-move jump would make pawns immune to a threat they'd normally face.");
    renderBoard(step.lesson.fen, { pieceSquare: step.lesson.pieceSquare, targets: step.lesson.targets, interactive: true });
  } else if (step.type === "promotion") {
    boardRow.hidden = false;
    contentEl.innerHTML = `
      <h2 class="panel-title">Promotion</h2>
      <p class="learn-body">
        A pawn that reaches the far end of the board — the 8th rank for White, the 1st for Black — is
        promoted: traded for a queen, rook, bishop, or knight of the same color, your choice. It can never
        stay a pawn, and it can never become a king.
      </p>
      <p class="learn-hint">Push the pawn all the way to the last rank.</p>`;
    renderLearnMore("Promoting to anything other than a queen is called \"underpromotion\" — rare, but occasionally correct (a knight, for instance, can deliver a check a queen couldn't from the same square).");
    renderBoard(step.lesson.fen, { pieceSquare: step.lesson.pieceSquare, targets: step.lesson.targets, interactive: true });
  } else if (step.type === "checkmate") {
    boardRow.hidden = false;
    contentEl.innerHTML = `
      <h2 class="panel-title">Check &amp; checkmate</h2>
      <p class="learn-body">
        <strong>Check</strong> means your king is under attack — you must deal with it immediately, by
        moving the king, blocking the attack, or capturing the attacker. <strong>Checkmate</strong> means
        none of those are possible — the game ends immediately, and whoever delivered it wins.
      </p>
      <p class="learn-hint">Your king is in check right now from the rook — find a way out.</p>`;
    renderBoard(step.lesson.fen, { pieceSquare: step.lesson.pieceSquare, targets: step.lesson.targets, interactive: true });
  } else if (step.type === "draws") {
    boardRow.hidden = false;
    contentEl.innerHTML = `
      <h2 class="panel-title">Draws</h2>
      <p class="learn-body">
        Not every game ends in checkmate. A game is drawn — nobody wins — if:
      </p>
      <ul class="learn-notation-list">
        <li><strong>Stalemate</strong> — the side to move has no legal move at all, but isn't in check (shown below).</li>
        <li><strong>Insufficient material</strong> — neither side has enough pieces left to possibly checkmate (e.g. king vs. king).</li>
        <li><strong>Threefold repetition</strong> — the exact same position occurs three times.</li>
        <li><strong>The 50-move rule</strong> — 50 moves pass with no pawn move and no capture.</li>
        <li>Both players can also simply agree to a draw at any time.</li>
      </ul>
      <p class="learn-hint">It's Black's move here, and Black has no legal move — but Black's king isn't in
        check. That combination is exactly what makes this a stalemate, not a loss.</p>`;
    renderBoard(STALEMATE_FEN, { interactive: false });
  } else if (step.type === "patterns") {
    boardRow.hidden = false;
    if (patternIndex >= CHECKMATE_PATTERNS.length) patternIndex = 0;
    const pattern = CHECKMATE_PATTERNS[patternIndex];
    contentEl.innerHTML = `
      <h2 class="panel-title">${pattern.name} <span class="panel-subtitle">${patternIndex + 1} / ${CHECKMATE_PATTERNS.length}</span></h2>
      <p class="learn-body">${pattern.explanation}</p>
      <p class="learn-hint">Find the checkmate.</p>`;
    renderBoard(pattern.fen, { pieceSquare: pattern.pieceSquare, targets: pattern.targets, interactive: true });
  } else if (step.type === "practice") {
    boardRow.hidden = false;
    if (practicePuzzleIndex >= PRACTICE_PUZZLES.length) practicePuzzleIndex = 0;
    const puzzle = PRACTICE_PUZZLES[practicePuzzleIndex];
    contentEl.innerHTML = `
      <h2 class="panel-title">Guided practice <span class="panel-subtitle">${practicePuzzleIndex + 1} / ${PRACTICE_PUZZLES.length}</span></h2>
      <p class="learn-body">${puzzle.prompt}</p>`;
    renderBoard(puzzle.fen, { pieceSquare: puzzle.pieceSquare, targets: puzzle.targets, interactive: true });
  } else if (step.type === "done") {
    boardRow.hidden = true;
    contentEl.innerHTML = `
      <h2 class="panel-title">You're ready!</h2>
      <p class="learn-body">
        You've seen how every piece moves and tried a few real ones yourself. From here, the best way to
        get better is to just start solving — the tutor adapts the difficulty and hints to you as you go.
      </p>`;
    try {
      localStorage.setItem(ONBOARDING_DONE_KEY, "1");
    } catch (err) {
      // No localStorage — the "New to chess?" banner just won't know to hide itself; harmless.
    }
  }

  renderProgress();
  document.getElementById("learn-back").disabled = stepIndex === 0;
  const nextBtn = document.getElementById("learn-next");
  if (step.type === "done") {
    nextBtn.textContent = "Start solving puzzles →";
    nextBtn.onclick = () => { window.location.href = "/"; };
  } else if (stepIndex === 0) {
    nextBtn.textContent = "Start tutorial →";
    nextBtn.onclick = () => goToStep(stepIndex + 1);
  } else {
    nextBtn.textContent = "Continue →";
    nextBtn.onclick = () => goToStep(stepIndex + 1);
  }
}

function playSetupAnimation() {
  setupStageIndex = 0;
  const advance = () => {
    const stage = SETUP_STAGES[setupStageIndex];
    const full = parseFen(START_FEN);
    const filtered = full.map((row, rankIdx) =>
      row.map((piece, fileIdx) => (stage.squares.includes(squareName(rankIdx, fileIdx)) ? piece : null)));
    liveBoard = filtered;
    renderLiveBoard({});
    document.getElementById("setup-caption").textContent = stage.caption;
    if (setupStageIndex < SETUP_STAGES.length - 1) {
      setupStageIndex += 1;
      setupTimer = setTimeout(advance, 1100);
    }
  };
  advance();
}

// ---------------------------------------------------------------------------
// Interaction: click-to-select-then-click, or drag directly. Both funnel
// into attemptMove() so success/failure feedback is identical either way.
// ---------------------------------------------------------------------------

function onSquareClick(name) {
  const lesson = activeLessonData();
  if (!lesson) return;
  if (name === lesson.pieceSquare) {
    selectedForClick = !selectedForClick;
    renderLiveBoard({ pieceSquare: lesson.pieceSquare, targets: lesson.targets, interactive: true });
    return;
  }
  if (selectedForClick) {
    attemptMove(name);
  }
}

function onHeroPointerDown(event) {
  const lesson = activeLessonData();
  if (!lesson) return;
  // preventDefault() here is load-bearing: per the Pointer Events spec, it
  // suppresses the browser's own compatibility "click" event that would
  // otherwise fire on this element right after pointerup — which matters
  // because a real drag already calls attemptMove() itself (below). Without
  // this, that follow-up click would call onSquareClick() a second time for
  // the same gesture. A flag-based "ignore the next click" workaround was
  // tried and removed: since preventDefault() means that click never fires
  // at all, such a flag never gets consumed and permanently swallows every
  // later tap-to-select attempt for the rest of the tutorial.
  event.preventDefault();
  const pieceEl = event.currentTarget;
  try {
    pieceEl.setPointerCapture(event.pointerId);
  } catch (err) {
    // Pointer capture can fail in some embedded/test contexts — dragging just
    // won't track outside the element in that case; click-to-move still works.
  }
  const drag = { pointerId: event.pointerId, pieceEl, startX: event.clientX, startY: event.clientY, moved: false, ghostEl: null };

  const onMove = (moveEvent) => {
    if (moveEvent.pointerId !== drag.pointerId) return;
    const dx = moveEvent.clientX - drag.startX;
    const dy = moveEvent.clientY - drag.startY;
    if (!drag.moved && Math.hypot(dx, dy) > 4) {
      drag.moved = true;
      drag.ghostEl = pieceEl.cloneNode(true);
      drag.ghostEl.classList.add("learn-drag-ghost");
      document.body.appendChild(drag.ghostEl);
      pieceEl.classList.add("learn-dragging-source");
    }
    if (drag.ghostEl) {
      drag.ghostEl.style.left = `${moveEvent.clientX}px`;
      drag.ghostEl.style.top = `${moveEvent.clientY}px`;
    }
  };

  const onUp = (upEvent) => {
    if (upEvent.pointerId !== drag.pointerId) return;
    pieceEl.removeEventListener("pointermove", onMove);
    pieceEl.removeEventListener("pointerup", onUp);
    pieceEl.removeEventListener("pointercancel", onUp);
    pieceEl.classList.remove("learn-dragging-source");
    if (drag.ghostEl) drag.ghostEl.remove();

    if (drag.moved) {
      const dropTarget = document.elementFromPoint(upEvent.clientX, upEvent.clientY);
      const squareEl = dropTarget && dropTarget.closest(".square");
      if (squareEl) attemptMove(squareEl.dataset.square);
    } else {
      onSquareClick(lesson.pieceSquare);
    }
  };

  pieceEl.addEventListener("pointermove", onMove);
  pieceEl.addEventListener("pointerup", onUp);
  pieceEl.addEventListener("pointercancel", onUp);
}

function attemptMove(targetSquare) {
  const lesson = activeLessonData();
  if (!lesson || targetSquare === lesson.pieceSquare) return;

  if (lesson.targets.includes(targetSquare)) {
    const [fr, ff] = squareIndices(lesson.pieceSquare);
    const [tr, tf] = squareIndices(targetSquare);
    let movingPiece = liveBoard[fr][ff];
    liveBoard[fr][ff] = null;

    // En passant: the captured pawn sits one square away from the actual
    // destination, not on it — remove it explicitly.
    if (lesson.enPassantCapture) {
      const [cr, cf] = squareIndices(lesson.enPassantCapture);
      liveBoard[cr][cf] = null;
    }
    // Promotion: the piece letter changes on arrival, same color.
    if (lesson.promoteTo) {
      movingPiece = isWhitePiece(movingPiece) ? lesson.promoteTo.toUpperCase() : lesson.promoteTo.toLowerCase();
    }
    liveBoard[tr][tf] = movingPiece;
    // Castling: the rook hops to the other side of the king at the same time.
    if (lesson.extraMove) {
      const [efr, eff] = squareIndices(lesson.extraMove.from);
      const [etr, etf] = squareIndices(lesson.extraMove.to);
      liveBoard[etr][etf] = liveBoard[efr][eff];
      liveBoard[efr][eff] = null;
    }

    selectedForClick = false;
    showLearnFeedback(lesson.successText, "correct");
    renderLiveBoard({ pieceSquare: null, targets: [], interactive: false, justMoved: targetSquare });
    solvedSteps.add(stepIndex);
    renderProgress();

    const step = currentStep();
    if (step.type === "practice") {
      setTimeout(() => advanceCycle(practicePuzzleIndex + 1, PRACTICE_PUZZLES.length, (next) => { practicePuzzleIndex = next; }), 1400);
    } else if (step.type === "patterns") {
      setTimeout(() => advanceCycle(patternIndex + 1, CHECKMATE_PATTERNS.length, (next) => { patternIndex = next; }), 1600);
    }
  } else {
    showLearnFeedback(lesson.failText, "incorrect");
  }
}

// Shared by "practice" and "patterns" — both cycle through a small list of
// positions within one step before moving on to the next step.
function advanceCycle(nextIndex, total, setIndex) {
  if (nextIndex < total) {
    setIndex(nextIndex);
    renderStep();
  } else {
    goToStep(stepIndex + 1);
  }
}

// ---------------------------------------------------------------------------
// Step navigation
// ---------------------------------------------------------------------------

function goToStep(index) {
  if (index < 0 || index >= STEPS.length) return;
  if (STEPS[index].type === "practice") practicePuzzleIndex = 0;
  if (STEPS[index].type === "patterns") patternIndex = 0;
  stepIndex = index;
  renderStep();
}

document.getElementById("learn-back").addEventListener("click", () => goToStep(stepIndex - 1));
document.getElementById("learn-replay").addEventListener("click", () => renderStep());

renderStep();
