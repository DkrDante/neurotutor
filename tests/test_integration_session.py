from pathlib import Path
import torch
from eeg.simulated import SimulatedEEGSource
from chess_task.base import TaskEngine, Puzzle, BehaviorEvent
from model.fusion import FusionLSTMClassifier
from model.inference import StatePredictor
from adaptive.rule_based import RuleBasedPolicy
from storage.db import SessionStore
from web.session import TutorSession
from common.config import NUM_CHANNELS, BAND_NAMES, NUM_BEHAVIOR_FEATS, SEQ_LEN
from common.states import STATES

class FakeTaskEngine(TaskEngine):
    def get_puzzle(self, difficulty: float) -> Puzzle:
        return Puzzle(puzzle_id="fake1", fen="irrelevant", solution_move="e2e4", rating=1000)

    def submit_move(self, puzzle: Puzzle, move_uci: str, time_to_move: float) -> BehaviorEvent:
        return BehaviorEvent(
            puzzle_id=puzzle.puzzle_id, correct=(move_uci == puzzle.solution_move),
            time_to_move=time_to_move, eval_loss=0.0, puzzle_rating=puzzle.rating,
        )

def _make_untrained_checkpoint(tmp_path: Path) -> Path:
    model = FusionLSTMClassifier(
        num_channels=NUM_CHANNELS, num_bands=len(BAND_NAMES),
        num_behavior_feats=NUM_BEHAVIOR_FEATS, num_classes=len(STATES),
    )
    path = tmp_path / "checkpoint.pt"
    torch.save(model.state_dict(), path)
    return path

def test_submit_move_produces_valid_session_update(tmp_path: Path):
    eeg_source = SimulatedEEGSource(target_state="Focused", seed=0)
    task_engine = FakeTaskEngine()
    predictor = StatePredictor(_make_untrained_checkpoint(tmp_path))
    policy = RuleBasedPolicy()
    store = SessionStore(":memory:")
    session = TutorSession(eeg_source, task_engine, predictor, policy, store)

    puzzle = session.next_puzzle()
    session.record_eeg_chunk(eeg_source.generate_chunk())
    update = session.submit_move(puzzle, "e2e4", time_to_move=5.0)

    assert update.correct is True
    assert update.predicted_state in STATES
    assert 0.0 <= update.confidence <= 1.0
    assert update.sense_to_adapt_latency >= 0.0

    summary = store.get_session_summary(session.session_id)
    assert summary["num_attempts"] == 1
    store.close()

def test_multiple_attempts_maintain_rolling_window(tmp_path: Path):
    eeg_source = SimulatedEEGSource(target_state="Overloaded", seed=1)
    task_engine = FakeTaskEngine()
    predictor = StatePredictor(_make_untrained_checkpoint(tmp_path))
    policy = RuleBasedPolicy()
    store = SessionStore(":memory:")
    session = TutorSession(eeg_source, task_engine, predictor, policy, store)

    for _ in range(8):
        puzzle = session.next_puzzle()
        session.record_eeg_chunk(eeg_source.generate_chunk())
        session.submit_move(puzzle, "e2e4", time_to_move=6.0)

    summary = store.get_session_summary(session.session_id)
    assert summary["num_attempts"] == 8
    assert len(session._eeg_history) == SEQ_LEN
    store.close()
