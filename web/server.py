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
from chess_task.puzzles import PuzzleTaskEngine
from chess_task.evaluator import MoveEvaluator, StockfishEvaluator
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

    async def stream_eeg() -> None:
        # Continuously fill the epoch buffer so extract_epoch() sees a full 4.0s of real
        # signal instead of a single zero-padded 0.25s chunk (which made the GCN branch
        # see ~99.97% zeros at serving time while training used full 4.0s chunks).
        async for chunk in eeg_source.stream():
            session.record_eeg_chunk(chunk)

    eeg_task = asyncio.create_task(stream_eeg())
    try:
        while True:
            puzzle = session.next_puzzle()
            # A fresh token per puzzle round: pacing_delay's asyncio.sleep (below) means a
            # client message can arrive after the NEXT puzzle has already been sent. Without
            # this, that stale move would be evaluated against the wrong puzzle. The client
            # echoes this token back in its move message; anything else is discarded here.
            attempt_token = str(uuid.uuid4())
            await websocket.send_json({
                "type": "puzzle",
                "puzzle_id": puzzle.puzzle_id,
                "fen": puzzle.fen,
                "session_id": session.session_id,
                "attempt_token": attempt_token,
            })

            while True:
                message = await websocket.receive_json()
                try:
                    move_uci = message["move_uci"]
                    time_to_move = float(message["time_to_move"])
                except (KeyError, TypeError, ValueError) as exc:
                    await websocket.send_json({"type": "error", "message": f"Malformed move: {exc}"})
                    continue
                if message.get("attempt_token") != attempt_token:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Stale move ignored (a new puzzle has already started).",
                    })
                    continue
                # submit_move stays synchronous on the event loop on purpose: it contains
                # no await, so the stream_eeg() task cannot interleave mid-way through it.
                # Offloading it (asyncio.to_thread) is deferred until EpochBuffer, a plain
                # list, is made thread-safe — otherwise it would race with stream_eeg().
                update = session.submit_move(puzzle, move_uci, time_to_move)
                break

            await websocket.send_json({
                "type": "update",
                "correct": update.correct,
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
            })
            await asyncio.sleep(update.action.pacing_delay)
    except WebSocketDisconnect:
        pass
    finally:
        eeg_task.cancel()
        evaluator.close()
