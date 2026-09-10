from __future__ import annotations
import shutil
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

@app.websocket("/ws/session")
async def session_endpoint(websocket: WebSocket):
    await websocket.accept()
    eeg_source = SimulatedEEGSource(target_state="Focused")
    task_engine = PuzzleTaskEngine(evaluator=build_evaluator())
    predictor = StatePredictor(DEFAULT_CHECKPOINT)
    session = TutorSession(eeg_source, task_engine, predictor, RuleBasedPolicy(), _store)

    try:
        while True:
            puzzle = session.next_puzzle()
            session.record_eeg_chunk(eeg_source.generate_chunk())
            await websocket.send_json({"type": "puzzle", "puzzle_id": puzzle.puzzle_id, "fen": puzzle.fen})

            message = await websocket.receive_json()
            update = session.submit_move(puzzle, message["move_uci"], message["time_to_move"])

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
            })
    except WebSocketDisconnect:
        pass
