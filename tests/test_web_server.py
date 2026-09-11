import time
from pathlib import Path
import torch
from fastapi.testclient import TestClient
from model.fusion import FusionLSTMClassifier
from common.config import NUM_CHANNELS, BAND_NAMES, NUM_BEHAVIOR_FEATS
from common.states import STATES
import web.server as server_module
from storage.db import SessionStore

def _install_test_server(tmp_path, monkeypatch) -> SessionStore:
    checkpoint_path = tmp_path / "checkpoint.pt"
    model = FusionLSTMClassifier(
        num_channels=NUM_CHANNELS, num_bands=len(BAND_NAMES),
        num_behavior_feats=NUM_BEHAVIOR_FEATS, num_classes=len(STATES),
    )
    torch.save(model.state_dict(), checkpoint_path)
    store = SessionStore(":memory:")
    monkeypatch.setattr(server_module, "DEFAULT_CHECKPOINT", checkpoint_path)
    monkeypatch.setattr(server_module, "_store", store)
    return store

def test_websocket_session_round_trip(tmp_path, monkeypatch):
    _install_test_server(tmp_path, monkeypatch)

    client = TestClient(server_module.app)
    with client.websocket_connect("/ws/session") as ws:
        puzzle_msg = ws.receive_json()
        assert puzzle_msg["type"] == "puzzle"
        assert "fen" in puzzle_msg
        assert "session_id" in puzzle_msg
        assert "attempt_token" in puzzle_msg

        ws.send_json({
            "move_uci": "a1a8", "time_to_move": 5.0,
            "attempt_token": puzzle_msg["attempt_token"],
        })
        update_msg = ws.receive_json()
        assert update_msg["type"] == "update"
        assert update_msg["predicted_state"] in STATES
        assert 0.0 <= update_msg["confidence"] <= 1.0

def test_stale_attempt_token_is_ignored_not_scored(tmp_path, monkeypatch):
    """Regression test: pacing_delay's asyncio.sleep means a client message can arrive
    after the NEXT puzzle has already been sent. A move carrying the OLD token must be
    rejected (not silently evaluated against the new puzzle), and a move with the
    current token must still work afterward."""
    _install_test_server(tmp_path, monkeypatch)

    client = TestClient(server_module.app)
    with client.websocket_connect("/ws/session") as ws:
        first_puzzle = ws.receive_json()
        stale_token = first_puzzle["attempt_token"]

        # Answer correctly so the server advances to a new puzzle (with a new token).
        ws.send_json({"move_uci": "a1a8", "time_to_move": 3.0, "attempt_token": stale_token})
        assert ws.receive_json()["type"] == "update"
        second_puzzle = ws.receive_json()
        assert second_puzzle["type"] == "puzzle"
        assert second_puzzle["attempt_token"] != stale_token

        # A move carrying the stale token must be rejected, not scored against puzzle 2.
        ws.send_json({"move_uci": "a1a8", "time_to_move": 1.0, "attempt_token": stale_token})
        err = ws.receive_json()
        assert err["type"] == "error"
        assert "stale" in err["message"].lower()

        # The current token still works normally afterward.
        ws.send_json({
            "move_uci": "a1a8", "time_to_move": 2.0,
            "attempt_token": second_puzzle["attempt_token"],
        })
        update_msg = ws.receive_json()
        assert update_msg["type"] == "update"

def test_live_session_accumulates_real_eeg_not_zero_padding(tmp_path, monkeypatch):
    """Regression test for the original serving-time bug.

    The server used to record exactly ONE 0.25s chunk per puzzle, so EpochBuffer
    zero-padded it to 512 samples and the GCN branch saw ~99.97% zeros. With the
    background stream_eeg() task, waiting before moving must yield band-power
    features in the same order of magnitude as the training data.
    """
    _install_test_server(tmp_path, monkeypatch)

    sessions = []
    real_session_cls = server_module.TutorSession

    def _capturing_session(*args, **kwargs):
        session = real_session_cls(*args, **kwargs)
        sessions.append(session)
        return session

    monkeypatch.setattr(server_module, "TutorSession", _capturing_session)

    client = TestClient(server_module.app)
    with client.websocket_connect("/ws/session") as ws:
        puzzle_msg = ws.receive_json()
        assert puzzle_msg["type"] == "puzzle"
        time.sleep(2.0)  # let the background EEG stream fill the epoch buffer
        ws.send_json({
            "move_uci": "a1a8", "time_to_move": 2.0,
            "attempt_token": puzzle_msg["attempt_token"],
        })
        assert ws.receive_json()["type"] == "update"

    assert len(sessions) == 1
    node_features = sessions[0]._eeg_history[-1]
    # A single zero-padded 0.25s chunk sums to ~7e-4 here; ~2s of real streamed
    # signal is two to three orders of magnitude above that.
    assert node_features.sum() > 0.1

def test_websocket_survives_malformed_messages(tmp_path, monkeypatch):
    _install_test_server(tmp_path, monkeypatch)

    client = TestClient(server_module.app)
    with client.websocket_connect("/ws/session") as ws:
        puzzle_msg = ws.receive_json()
        assert puzzle_msg["type"] == "puzzle"
        token = puzzle_msg["attempt_token"]

        # Missing move_uci -> KeyError path (checked before the token, same either way).
        ws.send_json({"time_to_move": 3.0, "attempt_token": token})
        err = ws.receive_json()
        assert err["type"] == "error"
        assert "Malformed move" in err["message"]

        # Non-numeric time_to_move -> ValueError path.
        ws.send_json({"move_uci": "a1a8", "time_to_move": "soon", "attempt_token": token})
        err = ws.receive_json()
        assert err["type"] == "error"

        # An illegal same-square move must not crash the session either.
        ws.send_json({"move_uci": "a1a1", "time_to_move": 2.0, "attempt_token": token})
        update_msg = ws.receive_json()
        assert update_msg["type"] == "update"
        assert update_msg["correct"] is False

        # Connection is still alive and still serving puzzles.
        next_puzzle = ws.receive_json()
        assert next_puzzle["type"] == "puzzle"
        ws.send_json({
            "move_uci": "a1a8", "time_to_move": 4.0,
            "attempt_token": next_puzzle["attempt_token"],
        })
        update_msg = ws.receive_json()
        assert update_msg["type"] == "update"
        assert update_msg["predicted_state"] in STATES

def test_missing_checkpoint_reports_error_instead_of_hanging(tmp_path, monkeypatch):
    monkeypatch.setattr(server_module, "DEFAULT_CHECKPOINT", tmp_path / "absent.pt")
    monkeypatch.setattr(server_module, "_store", SessionStore(":memory:"))

    client = TestClient(server_module.app)
    with client.websocket_connect("/ws/session") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "error"
        assert "checkpoint not found" in msg["message"].lower()

def test_session_summary_endpoint_returns_expected_keys(tmp_path, monkeypatch):
    store = _install_test_server(tmp_path, monkeypatch)
    client = TestClient(server_module.app)

    session_id = store.create_session()
    store.log_attempt(
        session_id=session_id, puzzle_id="rb01", correct=True, time_to_move=4.0,
        eval_loss=0.0, puzzle_rating=700, predicted_state="Focused", confidence=0.8,
        difficulty_delta=100.0, show_hint=False, pacing_delay=0.0, sense_to_adapt_latency=0.02,
    )

    response = client.get(f"/session/{session_id}/summary")
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"num_attempts", "accuracy_rate", "avg_latency", "state_trend"}
    assert body["num_attempts"] == 1
    assert body["accuracy_rate"] == 1.0
    assert body["state_trend"] == ["Focused"]

def test_session_summary_endpoint_for_unknown_session(tmp_path, monkeypatch):
    _install_test_server(tmp_path, monkeypatch)
    client = TestClient(server_module.app)
    body = client.get("/session/does-not-exist/summary").json()
    assert body == {"num_attempts": 0, "accuracy_rate": 0.0, "avg_latency": 0.0, "state_trend": []}
