from __future__ import annotations
import asyncio
import shutil
import uuid
from pathlib import Path
import chess
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from eeg.simulated import SimulatedEEGSource
from chess_task.base import Puzzle
from chess_task.puzzles import PuzzleTaskEngine
from chess_task.evaluator import MoveEvaluator, StockfishEvaluator
from chess_task.move_explainer import explain_illegal_move, legal_targets
from model.inference import StatePredictor
from adaptive.rule_based import RuleBasedPolicy
from storage.db import SessionStore
from web.session import TutorSession

DEFAULT_CHECKPOINT = Path("model/checkpoints/best.pt")
STATIC_DIR = Path(__file__).parent / "static"

class NullEvaluator(MoveEvaluator):
    """Used when no Stockfish binary is available; treats any wrong move as a fixed loss."""
    def eval_loss(self, board: chess.Board, played_move: chess.Move, best_move: chess.Move) -> float:
        return 200.0

def build_evaluator() -> MoveEvaluator:
    if shutil.which("stockfish") is not None:
        return StockfishEvaluator()
    return NullEvaluator()

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
_store = SessionStore("neurotutor.db")

@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/session/{session_id}/summary")
def session_summary(session_id: str):
    return _store.get_session_summary(session_id)

@app.websocket("/ws/session")
async def session_endpoint(websocket: WebSocket):
    await websocket.accept()

    if not DEFAULT_CHECKPOINT.exists():
        await websocket.send_json({
            "type": "error",
            "message": f"Model checkpoint not found at {DEFAULT_CHECKPOINT}. Run training first (see README).",
        })
        return

    eeg_source = SimulatedEEGSource(target_state="Focused")
    evaluator = build_evaluator()
    task_engine = PuzzleTaskEngine(evaluator=evaluator)
    predictor = StatePredictor(DEFAULT_CHECKPOINT)
    session = TutorSession(eeg_source, task_engine, predictor, RuleBasedPolicy(), _store)

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

            while True:  # one iteration per solver decision point: fresh puzzle, retry, or continuation
                # A fresh token per decision point: pacing_delay's asyncio.sleep (below)
                # means a client message can arrive after the next one has already been
                # sent. Without this, that stale move would be evaluated against the
                # wrong position. The client echoes this token back in its move message;
                # anything else is discarded here.
                attempt_token = str(uuid.uuid4())
                await send({
                    "type": "puzzle",
                    "puzzle_id": puzzle.puzzle_id,
                    "fen": board.fen(),
                    "session_id": session.session_id,
                    "attempt_token": attempt_token,
                    "solver_color": "white" if solver_is_white else "black",
                })

                while True:
                    message = await websocket.receive_json()
                    msg_type = message.get("type")

                    if msg_type == "hint":
                        square = message.get("square", "")
                        await send({
                            "type": "hint", "square": square, "targets": legal_targets(board, square),
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
                    # reset the puzzle; the player just gets a reason and can try again.
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

                    break  # legal move, ready to score

                expected_move = solution_moves[ply_index] if ply_index < len(solution_moves) else None
                is_right_move = move_uci == expected_move

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

                opponent_move = None
                if is_right_move:
                    board.push(chess.Move.from_uci(move_uci))
                    ply_index += 1
                    if ply_index >= len(solution_moves):
                        status = "solved"
                    else:
                        opponent_move = solution_moves[ply_index]
                        board.push(chess.Move.from_uci(opponent_move))
                        ply_index += 1
                        status = "continue"
                else:
                    status = "retry"

                await send({
                    "type": "update",
                    "status": status,
                    "correct": update.correct,
                    "opponent_move": opponent_move,
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
    except WebSocketDisconnect:
        pass
    finally:
        eeg_task.cancel()
        evaluator.close()
