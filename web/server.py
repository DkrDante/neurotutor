from __future__ import annotations
import asyncio
import csv
import io
import shutil
import uuid
from pathlib import Path
import chess
from fastapi import Depends, FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from accounts.storage import AccountStore, InvalidCredentialsError, UsernameTakenError
from eeg.simulated import SimulatedEEGSource
from chess_task.base import Puzzle
from chess_task.motifs import classify_puzzle
from chess_task.puzzles import DEFAULT_PUZZLE_CSV, PuzzleTaskEngine
from chess_task.full_game import FullGameEngine
from chess_task.evaluator import MoveEvaluator, StockfishEvaluator, win_probability
from chess_task.move_explainer import (
    PIECE_NAMES, describe_move, explain_illegal_move, explain_suboptimal_move, legal_targets, score_move,
)
import model.evidence as evidence
from model.inference import StatePredictor
from adaptive.rule_based import RuleBasedPolicy
from adaptive.rl_policy import QLearningPolicy
from coaching.narrator import build_coaching_report
from coaching.statistics import RATING_BANDS, build_statistics
from coaching.trace import build_attempt_trace, build_calculation_trace
from storage.db import SessionStore
from web.session import TutorSession

DEFAULT_CHECKPOINT = Path("model/checkpoints/best.pt")
# Real model trained with behavior features zeroed out (model/ablation.py) — loaded
# only if present, so a fresh checkout without the ablation study still serves fine
# (just without the "EEG-only" readout in the network-activity panel).
EEG_ONLY_CHECKPOINT = Path("model/checkpoints/ablation-eeg_only.pt")
STATIC_DIR = Path(__file__).parent / "static"
DEFAULT_STARTING_DIFFICULTY = 1000.0

# Maps a weakest_rating_band label (e.g. "1800-2199") back to the numeric
# bounds a targeted-practice /ws/session connection needs — built once from
# the same RATING_BANDS coaching.statistics already buckets attempts into,
# so the two can't silently drift apart.
RATING_BAND_BOUNDS = {label: (low, high) for low, high, label in RATING_BANDS}

# Below this adaptive difficulty, a learner has been missing enough puzzles
# that the fuller, more didactic explanation is warranted; above the upper
# bound they've been doing well enough that a terse one respects their time.
HAND_HOLDING_LOW_DIFFICULTY = 700.0
HAND_HOLDING_HIGH_DIFFICULTY = 1100.0

def hand_holding_tier(difficulty: float) -> str:
    """"high"/"medium"/"low" explanation depth for explain_suboptimal_move,
    driven by the session's current adaptive difficulty rather than a new
    signal — difficulty already rises on correct attempts and falls on wrong
    ones (adaptive.rule_based / adaptive.rl_policy), so it's already exactly
    "how well is this learner doing right now"."""
    if difficulty < HAND_HOLDING_LOW_DIFFICULTY:
        return "high"
    if difficulty < HAND_HOLDING_HIGH_DIFFICULTY:
        return "medium"
    return "low"

class NullEvaluator(MoveEvaluator):
    """Used when no Stockfish binary is available; treats any wrong move as a fixed loss."""
    def eval_loss(self, board: chess.Board, played_move: chess.Move, best_move: chess.Move) -> float:
        return 200.0

def build_evaluator() -> MoveEvaluator:
    if shutil.which("stockfish") is not None:
        return StockfishEvaluator()
    return NullEvaluator()

SESSION_COOKIE_NAME = "neurotutor_session"
SESSION_COOKIE_MAX_AGE_SECONDS = 60 * 60 * 24 * 365  # 1 year

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
_store = SessionStore("neurotutor.db")
_account_store = AccountStore("neurotutor.db")

@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/analytics")
def analytics_page():
    return FileResponse(STATIC_DIR / "analytics.html")

@app.get("/calculations")
def calculations_page():
    return FileResponse(STATIC_DIR / "calculations.html")

@app.get("/model-evidence")
def model_evidence_page():
    return FileResponse(STATIC_DIR / "model-evidence.html")

@app.get("/api/model-evidence")
def model_evidence_api(split: str = "test", sample_index: int = 0):
    """Real weights/biases, an independent precision/recall/F1/ROC-AUC
    evaluation, and a full forward-pass trace — computed fresh from the
    actual checkpoint web/server.py itself loads (DEFAULT_CHECKPOINT), the
    same numbers model/evaluate_model.py's offline report prints. See
    model/evidence.py for what's actually being computed here."""
    if not DEFAULT_CHECKPOINT.exists():
        raise HTTPException(status_code=404, detail=f"No trained checkpoint at {DEFAULT_CHECKPOINT}.")
    if split not in ("train", "val", "test"):
        raise HTTPException(status_code=400, detail="split must be train, val, or test.")
    model = evidence.load_model(DEFAULT_CHECKPOINT)
    try:
        trace = evidence.trace_forward_pass(model, split=split, sample_index=sample_index)
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err))
    return {
        "checkpoint_path": str(DEFAULT_CHECKPOINT),
        "weights": evidence.inspect_weights(model),
        "metrics": evidence.evaluate_metrics(model, split=split),
        "trace": trace,
        "training_history": evidence.load_training_history(DEFAULT_CHECKPOINT),
        "baselines": evidence.compute_baselines(),
        "feature_importance": evidence.compute_feature_importance(model, split=split),
    }

@app.get("/puzzles")
def puzzles_page():
    return FileResponse(STATIC_DIR / "puzzles.html")

@app.get("/api/puzzles")
def api_puzzles():
    """The full curated puzzle set PuzzleTaskEngine actually draws from —
    real rows read straight off chess_task/puzzle_data/sample_puzzles.csv
    right now, not a description of it. `source` is inferred from puzzle_id
    shape: the dozen hand-built mate patterns all use short, fixed prefixes
    (rb/qc/mm, 4 characters); every real Lichess-puzzle-database row is a
    5-character base62 id — the two never collide (see
    docs/PROJECT_REPORT.md for the full provenance note)."""
    puzzles = PuzzleTaskEngine._load_puzzles(DEFAULT_PUZZLE_CSV)
    rows = []
    for p in puzzles:
        solution_moves = p.solution_moves or [p.solution_move]
        try:
            motif = classify_puzzle(p.fen, solution_moves)
        except (ValueError, IndexError):
            motif = "other"
        rows.append({
            "puzzle_id": p.puzzle_id, "fen": p.fen, "solution_move": p.solution_move,
            "solution_moves": solution_moves, "rating": p.rating,
            "source": "hand-built" if len(p.puzzle_id) == 4 else "lichess",
            "motif": motif,
        })
    return {"total": len(rows), "puzzles": rows}

@app.get("/learn")
def learn_page():
    """The onboarding tutorial — self-contained (learn.html/learn.js), no
    session or model dependency, so it's served as a plain static page."""
    return FileResponse(STATIC_DIR / "learn.html")

class SignupRequest(BaseModel):
    username: str
    password: str
    # The signed-out client's localStorage gamification state, if any — folded
    # into the brand-new account so creating one doesn't discard progress
    # someone already made anonymously.
    progress: dict | None = None

class LoginRequest(BaseModel):
    username: str
    password: str

class ProgressUpdate(BaseModel):
    total_xp: int = 0
    solved_count: int = 0
    current_streak: int = 0
    best_streak: int = 0
    no_hint_streak: int = 0
    no_hint_total_count: int = 0
    max_rating_solved: int = 0
    games_played: int = 0
    games_won: int = 0
    puzzles_served: int = 0
    unlocked_achievements: list[str] = []

def _current_user_id(request: Request) -> str | None:
    token = request.cookies.get(SESSION_COOKIE_NAME)
    return _account_store.get_user_id_for_session(token) if token else None

def _require_login(request: Request) -> str:
    user_id = _current_user_id(request)
    if user_id is None:
        raise HTTPException(status_code=401, detail="Not logged in.")
    return user_id

def _set_session_cookie(response: Response, user_id: str) -> None:
    token = _account_store.create_account_session(user_id)
    response.set_cookie(
        SESSION_COOKIE_NAME, token, httponly=True, samesite="lax",
        max_age=SESSION_COOKIE_MAX_AGE_SECONDS,
    )

@app.post("/auth/signup")
def signup(body: SignupRequest, response: Response):
    """Lightweight account creation — hashed password (see accounts.auth),
    no email verification/reset, no login rate limiting. Scope decision:
    this is a solo capstone's account system, not a hardened auth service."""
    username = body.username.strip()
    if not (3 <= len(username) <= 32):
        raise HTTPException(status_code=400, detail="Username must be 3-32 characters.")
    if len(body.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")
    try:
        user_id = _account_store.create_user(username, body.password, initial_progress=body.progress)
    except UsernameTakenError:
        raise HTTPException(status_code=409, detail="That username is already taken.")
    _set_session_cookie(response, user_id)
    return {"logged_in": True, **_account_store.get_profile(user_id)}

@app.post("/auth/login")
def login(body: LoginRequest, response: Response):
    try:
        user_id = _account_store.verify_login(body.username.strip(), body.password)
    except InvalidCredentialsError:
        raise HTTPException(status_code=401, detail="Incorrect username or password.")
    _set_session_cookie(response, user_id)
    return {"logged_in": True, **_account_store.get_profile(user_id)}

@app.post("/auth/logout")
def logout(request: Request, response: Response):
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if token:
        _account_store.delete_account_session(token)
    response.delete_cookie(SESSION_COOKIE_NAME)
    return {"logged_out": True}

@app.get("/auth/me")
def whoami(request: Request):
    """Never raises for "not logged in" — the frontend calls this on every
    page load just to check login state, so that's a normal response, not
    an error."""
    user_id = _current_user_id(request)
    if user_id is None:
        return {"logged_in": False}
    profile = _account_store.get_profile(user_id)
    return {"logged_in": True, **profile}

@app.post("/account/progress")
def save_account_progress(body: ProgressUpdate, user_id: str = Depends(_require_login)):
    _account_store.save_progress(user_id, body.model_dump())
    return {"saved": True}

@app.post("/account/checkin")
def account_checkin(user_id: str = Depends(_require_login)):
    """Records "played at least once today" for the account's daily
    play-streak meter — call once per page load while logged in."""
    return _account_store.record_daily_checkin(user_id)

@app.get("/leaderboard")
def leaderboard(request: Request):
    """Public — ranked by total XP, ties broken by puzzles solved. No login
    required to view; if the caller IS logged in, their own rank is included
    even when they're outside the top of the returned list."""
    user_id = _current_user_id(request)
    return {
        "entries": _account_store.get_leaderboard(limit=50),
        "your_rank": _account_store.get_rank(user_id) if user_id else None,
    }

@app.get("/session/{session_id}/summary")
def session_summary(session_id: str):
    return _store.get_session_summary(session_id)

@app.get("/sessions")
def list_sessions():
    return _store.list_sessions()

def _policy_comparison_for_mode(mode: str) -> dict:
    """EXPLICIT rule-based-vs-RL comparison, scoped to one mode: coaching.trace.
    build_attempt_trace already reconstructs, per attempt, what the rule-based
    formula would have produced — this reruns that per session and aggregates
    how often the live policy actually diverged from it and by how much,
    rather than leaving that only visible one row at a time on the
    calculations page. Scoped by mode for the same reason get_aggregate_stats
    is: puzzle and full-game attempts mean different things."""
    divergent_attempts = 0
    total_traced_attempts = 0
    absolute_delta_differences = []
    for session in _store.list_sessions(mode=mode):
        trace = build_attempt_trace(_store.get_session_attempts(session["session_id"]))
        for attempt in trace:
            total_traced_attempts += 1
            recon = attempt["rule_based_reconstruction"]
            if not recon["matches_actual"]:
                divergent_attempts += 1
            absolute_delta_differences.append(abs(recon["predicted_delta"] - attempt["actual_delta"]))

    return {
        "total_attempts": total_traced_attempts,
        "divergent_attempts": divergent_attempts,
        "divergence_rate": (divergent_attempts / total_traced_attempts) if total_traced_attempts else None,
        "avg_absolute_delta_difference": (
            sum(absolute_delta_differences) / len(absolute_delta_differences)
        ) if absolute_delta_differences else None,
    }

@app.get("/analytics/aggregate")
def analytics_aggregate():
    """Cohort-level numbers across every recorded session — accuracy, sense-
    to-adapt latency, cognitive-state distribution, and an engagement trend,
    matching the design spec's own validation criteria (O4), which asked for
    these aggregated across a cohort, not eyeballed one session at a time.

    Split by mode ("puzzle" vs "game"), never combined: full-game attempts
    repurpose puzzle_rating to mean the live adaptive-difficulty number and
    puzzle_id to "game-ply-N" (see chess_task/full_game.py), so a single
    blended accuracy/state-distribution number would silently mix two
    populations that don't mean the same thing."""
    return {
        "puzzle": {**_store.get_aggregate_stats("puzzle"), "policy_comparison": _policy_comparison_for_mode("puzzle")},
        "game": {**_store.get_aggregate_stats("game"), "policy_comparison": _policy_comparison_for_mode("game")},
    }

@app.get("/sessions/export")
def export_sessions_csv():
    """Every logged attempt across every session, as CSV — the analytics page
    otherwise leaves everything trapped in SQLite with no way to pull it out
    for a report appendix or hand to a grader."""
    buffer = io.StringIO()
    fieldnames = [
        "session_id", "mode", "puzzle_id", "correct", "time_to_move", "eval_loss", "puzzle_rating",
        "predicted_state", "confidence", "difficulty_delta", "show_hint", "pacing_delay",
        "sense_to_adapt_latency", "created_at",
    ]
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for session in _store.list_sessions():
        for attempt in _store.get_session_attempts(session["session_id"]):
            writer.writerow({"session_id": session["session_id"], **attempt})
    return Response(
        content=buffer.getvalue(), media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=neurotutor_sessions.csv"},
    )

@app.get("/session/{session_id}/export")
def export_session_csv(session_id: str):
    """Same export as /sessions/export, scoped to one session."""
    buffer = io.StringIO()
    attempts = _store.get_session_attempts(session_id)
    fieldnames = [
        "mode", "puzzle_id", "correct", "time_to_move", "eval_loss", "puzzle_rating",
        "predicted_state", "confidence", "difficulty_delta", "show_hint", "pacing_delay",
        "sense_to_adapt_latency", "created_at",
    ]
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for attempt in attempts:
        writer.writerow(attempt)
    return Response(
        content=buffer.getvalue(), media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=neurotutor_session_{session_id}.csv"},
    )

def _calculation_notes() -> list[dict]:
    """Plain-English descriptions of the REAL formulas behind every number the
    analytics page shows — pulled from the actual constants/classes, not
    re-typed, so this can't silently drift out of sync with the code."""
    from adaptive.rule_based import (
        CONFIDENCE_THRESHOLD, CORRECT_DIFFICULTY_DELTA, STATE_RULES, WRONG_DIFFICULTY_DELTA,
    )
    rules_text = "; ".join(f"{state} {action.difficulty_delta:+.0f}" for state, action in STATE_RULES.items())
    return [
        {
            "title": "Accuracy",
            "formula": "correct_attempts / total_attempts",
            "explanation": "Fraction of attempts where the played move exactly matched the puzzle's solution move.",
        },
        {
            "title": "Confidence",
            "formula": "softmax(classifier(LSTM(fusion(EEG_embed, behavior_embed))))[predicted_state]",
            "explanation": "The fusion model's own softmax probability for whichever cognitive state it predicted for that attempt — not a separate calibration step.",
        },
        {
            "title": "Difficulty adjustment (rule-based policy)",
            "formula": (
                f"correct: {CORRECT_DIFFICULTY_DELTA:+.0f}, wrong: {WRONG_DIFFICULTY_DELTA:+.0f}; "
                f"if confidence >= {CONFIDENCE_THRESHOLD}, add a per-state delta ({rules_text})"
            ),
            "explanation": (
                "adaptive.rule_based.RuleBasedPolicy — correctness always adjusts difficulty (ground truth from "
                "the game); the predicted cognitive state adds its own push on top only once the model is "
                "confident enough to trust it. Session difficulty is floored at 400."
            ),
        },
        {
            "title": "Difficulty adjustment (RL policy)",
            "formula": "Q(s,a) += alpha * (reward + gamma * max_a' Q(s',a') - Q(s,a)); reward = +1 if correct else -1",
            "explanation": (
                "adaptive.rl_policy.QLearningPolicy — a tabular Q-learning agent over "
                "(cognitive state, confidence bucket) that LEARNS which of 5 difficulty/hint/pacing actions "
                "works best, instead of following a fixed rule table. Persisted per learner_id across sessions."
            ),
        },
        {
            "title": "Eval loss (this attempt's cost)",
            "formula": "0 if correct; Stockfish centipawn loss (best_move_eval - played_move_eval) if wrong; "
                       f"{PuzzleTaskEngine.ILLEGAL_MOVE_PENALTY:.0f} flat penalty if the move was illegal",
            "explanation": "Only a real engine score when a Stockfish binary is installed on the server; otherwise a fixed 200cp assumption for any wrong move (see NullEvaluator).",
        },
        {
            "title": "Win probability (board eval bar)",
            "formula": "1 / (1 + exp(-0.00368208 * eval_cp))",
            "explanation": "The same logistic curve lichess uses to turn a centipawn evaluation into a percentage. eval_cp itself is material-count-only unless Stockfish is installed (see chess_task.evaluator).",
        },
    ]

@app.get("/session/{session_id}/analytics")
def session_analytics(session_id: str):
    return {
        "summary": _store.get_session_summary(session_id),
        "attempts": _store.get_session_attempts(session_id),
        "calculations": _calculation_notes(),
    }

@app.get("/session/{session_id}/coach")
def session_coach(session_id: str):
    """Coaching feedback — not just accuracy/confidence, but which puzzle-rating
    band or predicted cognitive state is actually costing accuracy, whether the
    player rushes or overthinks before a mistake, blunder severity, and hint
    reliance. coaching.narrator.rule_based_tips() always answers this from the
    real per-attempt log; coaching.narrator.ai_narrative() additionally asks a
    local Ollama model (no API key, nothing leaves the machine) to turn those
    same computed numbers into a short coaching paragraph, and is None if
    Ollama isn't running or reachable. Also answers "how do puzzles adjust to
    this?": when a weak rating band stands out, `targeted_practice` gives the
    exact rating_min/rating_max a new /ws/session connection's ?target_min=/
    ?target_max= should carry to draw puzzles from THAT band specifically —
    the ordinary adaptive loop only ever moves a single difficulty number, so
    without this a weak spot never gets deliberately targeted on its own."""
    stats = build_statistics(_store.get_session_attempts(session_id))
    report = build_coaching_report(stats)
    weakest_band = stats.get("weakest_rating_band")
    if weakest_band:
        bounds = RATING_BAND_BOUNDS.get(weakest_band["key"])
        if bounds:
            report["targeted_practice"] = {
                "rating_min": bounds[0], "rating_max": bounds[1], "label": weakest_band["key"],
            }
    return report

@app.get("/session/{session_id}/calculations")
def session_calculations(session_id: str):
    """Every formula behind the tutor's numbers, with THIS session's real
    values substituted in — the generic formula reference (also shown on the
    analytics page), the aggregate results (accuracy, hint rate, difficulty
    trajectory, etc.) with real substituted numbers, and a per-attempt trace
    of the running accuracy and difficulty walk, including a reconstruction
    of what the rule-based policy would have produced for comparison."""
    attempts = _store.get_session_attempts(session_id)
    return {"formulas": _calculation_notes(), **build_calculation_trace(attempts)}

async def _receive_legal_move(
    websocket: WebSocket, send, board: chess.Board, attempt_token: str, expected_move: str | None,
    hint_state: dict | None = None,
):
    """Blocks until the client sends a legal move for `board`, transparently
    answering hint/puzzle_hint/check_move messages and rejecting malformed,
    stale, or illegal ones along the way. Shared by puzzle mode and full-game
    mode so their move-intake behavior can't silently drift apart.

    hint_state, when given, is mutated in place to record that the caller
    explicitly asked to see the solution (the "Solution" button — renamed
    from "Hint" client-side, and upgraded to reveal the real move, not just
    which piece to move) — the caller uses this to gate whether ordinary
    wrong-move feedback is also allowed to name the solution
    (chess_task.move_explainer.explain_suboptimal_move's reveal_solution).
    """
    while True:
        message = await websocket.receive_json()
        msg_type = message.get("type")

        if msg_type == "hint":
            square = message.get("square", "")
            await send({"type": "hint", "square": square, "targets": legal_targets(board, square)})
            continue

        if msg_type == "puzzle_hint":
            if hint_state is not None:
                hint_state["solution_revealed"] = True
            if expected_move:
                from_square_name, to_square_name = expected_move[:2], expected_move[2:4]
                piece = board.piece_at(chess.parse_square(from_square_name))
                piece_name = PIECE_NAMES[piece.piece_type] if piece else "piece"
                try:
                    solution_san = board.san(chess.Move.from_uci(expected_move))
                except (ValueError, IndexError, AssertionError):
                    solution_san = expected_move
                hint_text = f"{solution_san} — move your {piece_name} from {from_square_name} to {to_square_name}."
            else:
                from_square_name, to_square_name, hint_text = None, None, "No solution available for this position."
            await send({
                "type": "puzzle_hint", "square": from_square_name, "target_square": to_square_name, "text": hint_text,
            })
            continue

        if msg_type == "check_move":
            candidate = message.get("move_uci", "")
            await send({
                "type": "invalid_move",
                "reason": explain_illegal_move(board, candidate),
                "targets": legal_targets(board, candidate[:2]) if len(candidate) >= 2 else [],
            })
            continue

        try:
            move_uci = message["move_uci"]
            time_to_move = float(message["time_to_move"])
        except (KeyError, TypeError, ValueError) as exc:
            await send({"type": "error", "message": f"Malformed move: {exc}"})
            continue
        if message.get("attempt_token") != attempt_token:
            await send({
                "type": "error",
                "message": "Stale move ignored (a new puzzle has already started).",
            })
            continue

        # Defense in depth: the frontend only lets a player click legal target
        # squares, so this should already be legal — but never trust the client
        # alone. An illegal move here is NOT scored and does NOT advance or
        # reset anything; the player just gets a reason and can try again.
        try:
            parsed = chess.Move.from_uci(move_uci)
            is_legal = parsed in board.legal_moves
        except (ValueError, TypeError):
            is_legal = False
        if not is_legal:
            await send({
                "type": "invalid_move",
                "reason": explain_illegal_move(board, move_uci),
                "targets": legal_targets(board, move_uci[:2]) if len(move_uci) >= 2 else [],
            })
            continue

        return move_uci, time_to_move, parsed

# A struggling learner gets the solution auto-revealed rather than waiting
# indefinitely for them to ask — "multiple" wrong tries on the SAME puzzle.
AUTO_REVEAL_AFTER_CONSECUTIVE_WRONG = 3

async def _run_puzzle_mode(websocket: WebSocket, session: TutorSession, evaluator: MoveEvaluator, send, persist_profile) -> None:
    while True:
        puzzle = session.next_puzzle()
        # The solver's own moves plus the (pre-scripted, not engine-played) opponent
        # replies in between, e.g. [solver1, opponent1, solver2, ...]. Puzzles loaded
        # from CSV always populate this; the [puzzle.solution_move] fallback only
        # matters for a Puzzle built directly (e.g. in a test) without it.
        solution_moves = puzzle.solution_moves or [puzzle.solution_move]
        board = chess.Board(puzzle.fen)
        solver_is_white = board.turn == chess.WHITE
        ply_index = 0  # index into solution_moves the solver's NEXT move must match
        # Scoped to THIS puzzle_id — reset below whenever a genuinely new puzzle
        # starts, but preserved across a "retry" (same puzzle, board reset).
        hint_state = {"solution_revealed": False}
        consecutive_wrong = 0

        while True:  # one iteration per solver decision point: fresh puzzle, retry, or continuation
            # A fresh token per decision point: pacing_delay's asyncio.sleep (below)
            # means a client message can arrive after the next one has already been
            # sent. Without this, that stale move would be evaluated against the
            # wrong position. The client echoes this token back in its move message;
            # anything else is discarded here.
            attempt_token = str(uuid.uuid4())
            position_eval_cp = evaluator.evaluate_position(board)
            await send({
                "type": "puzzle",
                "puzzle_id": puzzle.puzzle_id,
                "fen": board.fen(),
                "session_id": session.session_id,
                "attempt_token": attempt_token,
                "solver_color": "white" if solver_is_white else "black",
                "rating": puzzle.rating,
                "eval_cp": position_eval_cp,
                "white_win_prob": win_probability(position_eval_cp),
                "mode": "puzzle",
            })
            # Known now (fixed for this decision point) so the "puzzle_hint" handler
            # below can use it without waiting for a move attempt.
            expected_move = solution_moves[ply_index] if ply_index < len(solution_moves) else None

            move_uci, time_to_move, parsed = await _receive_legal_move(
                websocket, send, board, attempt_token, expected_move, hint_state=hint_state,
            )

            is_right_move = move_uci == expected_move
            # Computed once, against the shared pre-move position, so it's available
            # for the notation explanation regardless of whether the move turns out
            # to be right or wrong (board.san needs the pre-move board to disambiguate,
            # e.g. "Nbd2" vs "Nd2", so this must happen before any push below).
            played_san = board.san(parsed)

            # submit_move still takes a single-ply Puzzle (unchanged interface) built
            # fresh for whichever ply/position is live right now — the rest of the
            # pipeline (model, adaptive engine, storage) doesn't need to know anything
            # about multi-move puzzles.
            scoring_puzzle = Puzzle(
                puzzle_id=puzzle.puzzle_id, fen=board.fen(),
                solution_move=expected_move or move_uci, rating=puzzle.rating,
            )
            # submit_move stays synchronous on the event loop on purpose: it contains
            # no await, so the stream_eeg() task cannot interleave mid-way through it.
            # Offloading it (asyncio.to_thread) is deferred until EpochBuffer, a plain
            # list, is made thread-safe — otherwise it would race with stream_eeg().
            update = session.submit_move(scoring_puzzle, move_uci, time_to_move)
            persist_profile()

            opponent_move = None
            move_reason = None
            move_quality = None
            move_description = None
            solver_san = played_san if is_right_move else None
            opponent_san = None
            if is_right_move:
                consecutive_wrong = 0
                board.push(chess.Move.from_uci(move_uci))
                ply_index += 1
                if ply_index >= len(solution_moves):
                    status = "solved"
                else:
                    opponent_move = solution_moves[ply_index]
                    opponent_san = board.san(chess.Move.from_uci(opponent_move))
                    board.push(chess.Move.from_uci(opponent_move))
                    ply_index += 1
                    status = "continue"
            else:
                status = "retry"
                consecutive_wrong += 1
                # Auto-reveal: either a proactive-intervention trend just fired
                # (adaptive.proactive — a struggle signal from cognitive state,
                # independent of this attempt's own correctness), or the learner
                # has now missed the SAME puzzle "multiple" times in a row —
                # either way, stop making them ask.
                if update.proactive_intervention or consecutive_wrong >= AUTO_REVEAL_AFTER_CONSECUTIVE_WRONG:
                    hint_state["solution_revealed"] = True
                # board here is still the pre-move position (only pushed above on
                # a right move), which is exactly what explain_suboptimal_move/
                # describe_move need.
                tier = hand_holding_tier(session.difficulty)
                move_reason = (
                    explain_suboptimal_move(
                        board, move_uci, expected_move, tier=tier, reveal_solution=hint_state["solution_revealed"],
                    ) if expected_move else None
                )
                move_quality = score_move(update.eval_loss)
                move_description = describe_move(board, move_uci)

            position_eval_cp = evaluator.evaluate_position(board)
            await send({
                "type": "update",
                "status": status,
                "correct": update.correct,
                "opponent_move": opponent_move,
                "played_san": played_san,
                "solver_san": solver_san,
                "opponent_san": opponent_san,
                "reason": move_reason,
                "solution_revealed": hint_state["solution_revealed"],
                "proactive_intervention": update.proactive_intervention,
                "move_score": move_quality["score"] if move_quality else None,
                "move_quality_label": move_quality["label"] if move_quality else None,
                "move_description": move_description,
                "eval_cp": position_eval_cp,
                "white_win_prob": win_probability(position_eval_cp),
                "predicted_state": update.predicted_state,
                "confidence": update.confidence,
                "probs": update.probs,
                "action": {
                    "difficulty_delta": update.action.difficulty_delta,
                    "show_hint": update.action.show_hint,
                    "pacing_delay": update.action.pacing_delay,
                },
                "difficulty": session.difficulty,
                "session_id": session.session_id,
                "network_activity": update.network_activity,
            })
            await asyncio.sleep(update.action.pacing_delay)

            if status == "solved":
                break  # move to a brand-new puzzle in the outer loop
            if status == "retry":
                board = chess.Board(puzzle.fen)  # same puzzle, reset to its start
                ply_index = 0
            # status == "continue": loop again on the same puzzle with the board
            # already advanced past the opponent's scripted reply.

async def _run_game_mode(websocket: WebSocket, session: TutorSession, evaluator: MoveEvaluator, send, persist_profile) -> None:
    """Full-game mode: play a whole game against an adaptive-strength AI
    opponent (see chess_task.full_game.FullGameEngine) instead of solving
    discrete puzzles. Unlike puzzle mode, a legal-but-suboptimal move is never
    retried — it's final, exactly as it would be in a real game — and the AI
    opponent's reply is computed live (chess_task.evaluator.MoveEvaluator),
    never pre-scripted.
    """
    # Duck-typed as FullGameEngine by contract (session_endpoint only calls this
    # loop when it built the session with one) — same trust puzzle mode already
    # places in session.task_engine being PuzzleTaskEngine-shaped, and it keeps
    # this symbol swappable in tests the same way PuzzleTaskEngine already is.
    game_engine = session.task_engine
    # Unlike puzzle mode there's no single puzzle_id to scope this to (every
    # ply gets a fresh one) — tracked across the whole game instead, so a run
    # of real misses still triggers the same auto-reveal.
    consecutive_wrong = 0

    while True:
        board = game_engine.board
        if board.is_game_over():
            await send({
                "type": "game_over", "result": board.result(), "fen": board.fen(),
                "session_id": session.session_id,
            })
            return

        puzzle = session.next_puzzle()  # packages the LIVE position + the evaluator's best move for it
        expected_move = puzzle.solution_move
        attempt_token = str(uuid.uuid4())
        position_eval_cp = evaluator.evaluate_position(board)
        await send({
            "type": "puzzle",
            "puzzle_id": puzzle.puzzle_id,
            "fen": board.fen(),
            "session_id": session.session_id,
            "attempt_token": attempt_token,
            "solver_color": "white" if board.turn == chess.WHITE else "black",
            "rating": puzzle.rating,
            "eval_cp": position_eval_cp,
            "white_win_prob": win_probability(position_eval_cp),
            "mode": "game",
        })

        hint_state = {"solution_revealed": False}
        move_uci, time_to_move, parsed = await _receive_legal_move(
            websocket, send, board, attempt_token, expected_move, hint_state=hint_state,
        )
        played_san = board.san(parsed)

        update = session.submit_move(puzzle, move_uci, time_to_move)
        persist_profile()

        move_reason = None
        move_quality = None
        move_description = None
        if not update.correct:
            consecutive_wrong += 1
            if update.proactive_intervention or consecutive_wrong >= AUTO_REVEAL_AFTER_CONSECUTIVE_WRONG:
                hint_state["solution_revealed"] = True
            # The move still stands — this is feedback, not a retry gate.
            tier = hand_holding_tier(session.difficulty)
            move_reason = explain_suboptimal_move(
                board, move_uci, expected_move, tier=tier, reveal_solution=hint_state["solution_revealed"],
            )
            move_quality = score_move(update.eval_loss)
            move_description = describe_move(board, move_uci)
        else:
            consecutive_wrong = 0
        game_engine.push_player_move(move_uci)

        game_over_after_player_move = board.is_game_over()
        position_eval_cp = evaluator.evaluate_position(board)
        # The opponent's reply isn't sent yet even though pick_opponent_move()
        # below would return instantly — this message lands first so the player
        # sees their OWN result right away, with the "AI is thinking" pause (if
        # any) happening visibly afterward rather than as an unexplained delay
        # before anything appears at all.
        await send({
            "type": "update",
            "status": "game_over" if game_over_after_player_move else "game_move",
            "correct": update.correct,
            "opponent_move": None,
            "played_san": played_san,
            "solver_san": played_san,
            "opponent_san": None,
            "reason": move_reason,
            "solution_revealed": hint_state["solution_revealed"],
            "proactive_intervention": update.proactive_intervention,
            "move_score": move_quality["score"] if move_quality else None,
            "move_quality_label": move_quality["label"] if move_quality else None,
            "move_description": move_description,
            "eval_cp": position_eval_cp,
            "white_win_prob": win_probability(position_eval_cp),
            "predicted_state": update.predicted_state,
            "confidence": update.confidence,
            "probs": update.probs,
            "action": {
                "difficulty_delta": update.action.difficulty_delta,
                "show_hint": update.action.show_hint,
                "pacing_delay": update.action.pacing_delay,
            },
            "difficulty": session.difficulty,
            "session_id": session.session_id,
            "network_activity": update.network_activity,
            "game_result": board.result() if game_over_after_player_move else None,
            "awaiting_opponent": not game_over_after_player_move,
        })

        if game_over_after_player_move:
            return

        # The AI "thinks" before its move is revealed — see
        # FullGameEngine.thinking_time's docstring for why this is purely a UX
        # pacing device layered on top of a move that's already been chosen.
        await asyncio.sleep(game_engine.thinking_time(session.difficulty))

        opp_move = game_engine.pick_opponent_move(session.difficulty)
        opponent_san = board.san(opp_move)
        game_engine.push_opponent_move(opp_move)
        game_over_after_opponent_move = board.is_game_over()
        position_eval_cp = evaluator.evaluate_position(board)
        await send({
            "type": "opponent_move",
            "move": opp_move.uci(),
            "san": opponent_san,
            "fen": board.fen(),
            "eval_cp": position_eval_cp,
            "white_win_prob": win_probability(position_eval_cp),
            "status": "game_over" if game_over_after_opponent_move else "game_move",
            "game_result": board.result() if game_over_after_opponent_move else None,
            "session_id": session.session_id,
        })
        await asyncio.sleep(update.action.pacing_delay)

        if game_over_after_opponent_move:
            return

@app.websocket("/ws/session")
async def session_endpoint(websocket: WebSocket):
    await websocket.accept()

    if not DEFAULT_CHECKPOINT.exists():
        await websocket.send_json({
            "type": "error",
            "message": f"Model checkpoint not found at {DEFAULT_CHECKPOINT}. Run training first (see README).",
        })
        return

    # ?mode=puzzle|game, ?policy=rule|rl, ?learner_id=<opaque client-generated id>,
    # ?target_min=&target_max=<rating band to draw puzzles from>. All optional and
    # backward compatible: no query params behaves exactly as before (puzzle mode,
    # rule-based policy, no personalization, puzzles picked by nearest-to-difficulty
    # across the whole set rather than a specific weak rating band).
    mode = websocket.query_params.get("mode", "puzzle")
    policy_name = websocket.query_params.get("policy", "rule")
    learner_id = websocket.query_params.get("learner_id")
    target_min_raw = websocket.query_params.get("target_min")
    target_max_raw = websocket.query_params.get("target_max")
    target_rating_min = float(target_min_raw) if target_min_raw else None
    target_rating_max = float(target_max_raw) if target_max_raw else None

    eeg_source = SimulatedEEGSource(target_state="Focused")
    evaluator = build_evaluator()

    # Multi-session personalization: a returning learner resumes at their last
    # difficulty, and — if using the RL policy — their learned Q-table too,
    # instead of starting cold every session.
    profile = _store.get_learner_profile(learner_id) if learner_id else None
    initial_difficulty = profile["difficulty"] if profile else DEFAULT_STARTING_DIFFICULTY

    if mode == "game":
        task_engine = FullGameEngine(evaluator)
    else:
        # Seeded from the learner's persisted served_puzzle_ids so a reconnect
        # (a page refresh included) doesn't immediately re-serve a puzzle
        # already shown — a fresh PuzzleTaskEngine is otherwise constructed
        # per connection and starts with no no-repeat history of its own.
        already_served = profile.get("served_puzzle_ids") if profile else None
        task_engine = PuzzleTaskEngine(evaluator=evaluator, already_served=already_served)

    predictor = StatePredictor(DEFAULT_CHECKPOINT)
    eeg_only_predictor = StatePredictor(EEG_ONLY_CHECKPOINT) if EEG_ONLY_CHECKPOINT.exists() else None

    if policy_name == "rl":
        policy = (
            QLearningPolicy.from_state(profile["policy_state"])
            if profile and profile.get("policy_state") else QLearningPolicy()
        )
    else:
        policy = RuleBasedPolicy()

    session = TutorSession(
        eeg_source, task_engine, predictor, policy, _store,
        initial_difficulty=initial_difficulty, eeg_only_predictor=eeg_only_predictor,
        target_rating_min=target_rating_min, target_rating_max=target_rating_max,
        mode=mode,
    )

    def persist_profile() -> None:
        if not learner_id:
            return
        policy_state = policy.to_state() if isinstance(policy, QLearningPolicy) else None
        # hasattr, not isinstance: tests monkeypatch server_module.PuzzleTaskEngine
        # with a plain lambda/subclass, which isinstance would reject here.
        served_puzzle_ids = (
            task_engine.served_puzzle_ids if hasattr(task_engine, "served_puzzle_ids") else None
        )
        _store.save_learner_profile(learner_id, session.difficulty, policy_state, served_puzzle_ids)

    # stream_eeg() (below) and the main loop both send on this one websocket from two
    # different asyncio tasks; this lock keeps individual send_json calls from
    # interleaving with each other.
    send_lock = asyncio.Lock()

    async def send(payload: dict) -> None:
        async with send_lock:
            await websocket.send_json(payload)

    async def stream_eeg() -> None:
        # Continuously fill the epoch buffer so extract_epoch() sees a full 4.0s of real
        # signal instead of a single zero-padded 0.25s chunk (which made the GCN branch
        # see ~99.97% zeros at serving time while training used full 4.0s chunks). Each
        # chunk is also forwarded to the frontend for the live EEG waveform display —
        # this is the real streamed signal, not a decorative animation.
        async for chunk in eeg_source.stream():
            session.record_eeg_chunk(chunk)
            await send({
                "type": "eeg_chunk",
                "samples": chunk.samples.tolist(),  # (chunk_samples, num_channels)
            })

    eeg_task = asyncio.create_task(stream_eeg())
    try:
        if mode == "game":
            await _run_game_mode(websocket, session, evaluator, send, persist_profile)
        else:
            await _run_puzzle_mode(websocket, session, evaluator, send, persist_profile)
    except WebSocketDisconnect:
        pass
    finally:
        eeg_task.cancel()
        evaluator.close()
