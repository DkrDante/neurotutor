from pathlib import Path
import pytest
import torch
from eeg.simulated import SimulatedEEGSource
from chess_task.base import TaskEngine, Puzzle, BehaviorEvent
from model.fusion import FusionLSTMClassifier
from model.inference import StatePredictor
from model.evidence import DEFAULT_CHECKPOINT
from adaptive.rule_based import RuleBasedPolicy
from storage.db import SessionStore
from web.session import TutorSession
from common.config import NUM_CHANNELS, BAND_NAMES, NUM_BEHAVIOR_FEATS, SEQ_LEN
from common.states import STATES
from adaptive.proactive import TREND_WINDOW

class FakeTaskEngine(TaskEngine):
    def get_puzzle(self, difficulty: float, rating_min: float | None = None, rating_max: float | None = None) -> Puzzle:
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

def test_eeg_only_predictor_adds_network_activity_fields(tmp_path: Path):
    eeg_source = SimulatedEEGSource(target_state="Focused", seed=0)
    task_engine = FakeTaskEngine()
    predictor = StatePredictor(_make_untrained_checkpoint(tmp_path))
    eeg_only_predictor = StatePredictor(_make_untrained_checkpoint(tmp_path))
    session = TutorSession(
        eeg_source, task_engine, predictor, RuleBasedPolicy(), SessionStore(":memory:"),
        eeg_only_predictor=eeg_only_predictor,
    )

    puzzle = session.next_puzzle()
    session.record_eeg_chunk(eeg_source.generate_chunk())
    update = session.submit_move(puzzle, "e2e4", time_to_move=5.0)

    assert update.network_activity["eeg_only_state"] in STATES
    assert 0.0 <= update.network_activity["eeg_only_confidence"] <= 1.0
    assert set(update.network_activity["eeg_only_probs"]) == set(STATES)

def test_without_eeg_only_predictor_no_eeg_only_fields(tmp_path: Path):
    eeg_source = SimulatedEEGSource(target_state="Focused", seed=0)
    task_engine = FakeTaskEngine()
    predictor = StatePredictor(_make_untrained_checkpoint(tmp_path))
    session = TutorSession(eeg_source, task_engine, predictor, RuleBasedPolicy(), SessionStore(":memory:"))

    puzzle = session.next_puzzle()
    session.record_eeg_chunk(eeg_source.generate_chunk())
    update = session.submit_move(puzzle, "e2e4", time_to_move=5.0)

    assert "eeg_only_state" not in update.network_activity

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

def test_proactive_intervention_triggers_after_sustained_struggle_trend(tmp_path: Path, monkeypatch):
    eeg_source = SimulatedEEGSource(target_state="Focused", seed=0)
    task_engine = FakeTaskEngine()
    predictor = StatePredictor(_make_untrained_checkpoint(tmp_path))
    session = TutorSession(eeg_source, task_engine, predictor, RuleBasedPolicy(), SessionStore(":memory:"))

    # Bypass the (untrained, effectively random) model entirely so the trend
    # is deterministic: every attempt reads as a confident Overloaded state.
    monkeypatch.setattr(
        predictor, "predict_with_internals",
        lambda eeg_seq, behavior_seq: {"state": "Overloaded", "confidence": 0.9, "probs": {}},
    )

    updates = []
    for _ in range(TREND_WINDOW):
        puzzle = session.next_puzzle()
        session.record_eeg_chunk(eeg_source.generate_chunk())
        updates.append(session.submit_move(puzzle, "e2e4", time_to_move=5.0))

    assert [u.proactive_intervention for u in updates[:-1]] == [False] * (TREND_WINDOW - 1)
    assert updates[-1].proactive_intervention is True  # window just completed

    # A further consecutive struggling attempt is the SAME episode — no re-trigger.
    puzzle = session.next_puzzle()
    session.record_eeg_chunk(eeg_source.generate_chunk())
    update_next = session.submit_move(puzzle, "e2e4", time_to_move=5.0)
    assert update_next.proactive_intervention is False

def test_proactive_intervention_can_retrigger_after_a_focused_gap(tmp_path: Path, monkeypatch):
    eeg_source = SimulatedEEGSource(target_state="Focused", seed=0)
    task_engine = FakeTaskEngine()
    predictor = StatePredictor(_make_untrained_checkpoint(tmp_path))
    session = TutorSession(eeg_source, task_engine, predictor, RuleBasedPolicy(), SessionStore(":memory:"))

    state_sequence = ["Overloaded"] * TREND_WINDOW + ["Focused"] * TREND_WINDOW + ["Overloaded"] * TREND_WINDOW
    calls = iter(state_sequence)
    monkeypatch.setattr(
        predictor, "predict_with_internals",
        lambda eeg_seq, behavior_seq: {"state": next(calls), "confidence": 0.9, "probs": {}},
    )

    triggers = []
    for _ in state_sequence:
        puzzle = session.next_puzzle()
        session.record_eeg_chunk(eeg_source.generate_chunk())
        triggers.append(session.submit_move(puzzle, "e2e4", time_to_move=5.0).proactive_intervention)

    # One trigger at the end of the first Overloaded run, and a second after
    # the Focused gap resets _proactive_active and a new Overloaded run completes.
    assert triggers.count(True) == 2

@pytest.mark.skipif(not DEFAULT_CHECKPOINT.exists(), reason="model/checkpoints/best.pt not trained in this checkout")
def test_real_trained_checkpoint_produces_valid_predictions_through_a_live_session():
    """Every other TutorSession test uses a freshly-initialized, untrained
    checkpoint (see _make_untrained_checkpoint) — a random-weights model
    would still produce a valid-shaped output, so those tests never actually
    exercise the real, trained model this app serves in production. This one
    loads the ACTUAL checkpoint web/server.py's DEFAULT_CHECKPOINT points at
    and drives several real attempts through it, end to end."""
    eeg_source = SimulatedEEGSource(target_state="Focused", seed=0)
    task_engine = FakeTaskEngine()
    predictor = StatePredictor(DEFAULT_CHECKPOINT)
    store = SessionStore(":memory:")
    session = TutorSession(eeg_source, task_engine, predictor, RuleBasedPolicy(), store)

    updates = []
    for _ in range(5):
        puzzle = session.next_puzzle()
        session.record_eeg_chunk(eeg_source.generate_chunk())
        updates.append(session.submit_move(puzzle, "e2e4", time_to_move=5.0))

    for update in updates:
        assert update.predicted_state in STATES
        assert 0.0 <= update.confidence <= 1.0
        assert update.sense_to_adapt_latency >= 0.0
        # The real model's softmax always sums to 1 — a stub returning a
        # constant/zeroed vector would fail this.
        assert abs(sum(update.probs.values()) - 1.0) < 1e-4

    summary = store.get_session_summary(session.session_id)
    assert summary["num_attempts"] == 5
    store.close()
