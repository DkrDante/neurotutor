import time
from pathlib import Path
import chess
import torch
from fastapi.testclient import TestClient
from model.fusion import FusionLSTMClassifier
from common.config import NUM_CHANNELS, BAND_NAMES, NUM_BEHAVIOR_FEATS
from common.states import STATES
import web.server as server_module
from storage.db import SessionStore
from chess_task.base import Puzzle
from chess_task.puzzles import PuzzleTaskEngine
from chess_task.evaluator import MoveEvaluator

def _recv(ws):
    """Read the next non-eeg_chunk message. The live server also streams eeg_chunk
    messages continuously in the background for the waveform display; tests (like a
    real frontend routing them straight to the chart) only care about the rest."""
    while True:
        msg = ws.receive_json()
        if msg.get("type") != "eeg_chunk":
            return msg

def _fix_served_puzzle(monkeypatch, puzzle: Puzzle) -> None:
    """Force session_endpoint's PuzzleTaskEngine to always serve exactly this puzzle,
    regardless of difficulty, for tests that need a specific known position."""
    class FixedPuzzleEngine(PuzzleTaskEngine):
        def get_puzzle(self, difficulty):
            return puzzle
    monkeypatch.setattr(
        server_module, "PuzzleTaskEngine", lambda evaluator: FixedPuzzleEngine(evaluator=evaluator),
    )

class _NoOpEvaluator(MoveEvaluator):
    def eval_loss(self, board, played_move, best_move) -> float:
        return 999.0

def _solution_for(puzzle_id: str) -> str:
    """The real, merged puzzle set (chess_task/puzzle_data/sample_puzzles.csv) has no
    fixed content tests can hardcode a move against — look up whatever solution the
    served puzzle actually has."""
    engine = PuzzleTaskEngine(evaluator=_NoOpEvaluator())
    for puzzle in engine.puzzles:
        if puzzle.puzzle_id == puzzle_id:
            return puzzle.solution_move
    raise KeyError(puzzle_id)

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
        puzzle_msg = _recv(ws)
        assert puzzle_msg["type"] == "puzzle"
        assert "fen" in puzzle_msg
        assert "session_id" in puzzle_msg
        assert "attempt_token" in puzzle_msg

        ws.send_json({
            "move_uci": _solution_for(puzzle_msg["puzzle_id"]), "time_to_move": 5.0,
            "attempt_token": puzzle_msg["attempt_token"],
        })
        update_msg = _recv(ws)
        assert update_msg["type"] == "update"
        assert update_msg["correct"] is True
        assert update_msg["predicted_state"] in STATES
        assert 0.0 <= update_msg["confidence"] <= 1.0

        # Real internals for the network-activity visualization, not placeholders.
        activity = update_msg["network_activity"]
        assert len(activity["gcn1_node_activity"]) == NUM_CHANNELS
        assert len(activity["gcn_adjacency"]) == NUM_CHANNELS
        assert len(activity["behavior_activity"]) == NUM_BEHAVIOR_FEATS

def test_stale_attempt_token_is_ignored_not_scored(tmp_path, monkeypatch):
    """Regression test: pacing_delay's asyncio.sleep means a client message can arrive
    after the NEXT puzzle has already been sent. A move carrying the OLD token must be
    rejected (not silently evaluated against the new puzzle), and a move with the
    current token must still work afterward."""
    _install_test_server(tmp_path, monkeypatch)

    client = TestClient(server_module.app)
    with client.websocket_connect("/ws/session") as ws:
        first_puzzle = _recv(ws)
        stale_token = first_puzzle["attempt_token"]

        # Answer correctly so the server advances to a new puzzle (with a new token).
        ws.send_json({
            "move_uci": _solution_for(first_puzzle["puzzle_id"]), "time_to_move": 3.0,
            "attempt_token": stale_token,
        })
        assert _recv(ws)["type"] == "update"
        second_puzzle = _recv(ws)
        assert second_puzzle["type"] == "puzzle"
        assert second_puzzle["attempt_token"] != stale_token

        # A move carrying the stale token must be rejected, not scored against puzzle 2.
        ws.send_json({
            "move_uci": _solution_for(first_puzzle["puzzle_id"]), "time_to_move": 1.0,
            "attempt_token": stale_token,
        })
        err = _recv(ws)
        assert err["type"] == "error"
        assert "stale" in err["message"].lower()

        # The current token still works normally afterward.
        ws.send_json({
            "move_uci": _solution_for(second_puzzle["puzzle_id"]), "time_to_move": 2.0,
            "attempt_token": second_puzzle["attempt_token"],
        })
        update_msg = _recv(ws)
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
        puzzle_msg = _recv(ws)
        assert puzzle_msg["type"] == "puzzle"
        time.sleep(2.0)  # let the background EEG stream fill the epoch buffer
        ws.send_json({
            "move_uci": _solution_for(puzzle_msg["puzzle_id"]), "time_to_move": 2.0,
            "attempt_token": puzzle_msg["attempt_token"],
        })
        assert _recv(ws)["type"] == "update"

    assert len(sessions) == 1
    node_features = sessions[0]._eeg_history[-1]
    # A single zero-padded 0.25s chunk sums to ~7e-4 here; ~2s of real streamed
    # signal is two to three orders of magnitude above that.
    assert node_features.sum() > 0.1

def test_eeg_chunk_messages_are_streamed_live(tmp_path, monkeypatch):
    """The background stream_eeg() task must actually forward chunks to the client,
    not just record them internally."""
    _install_test_server(tmp_path, monkeypatch)

    client = TestClient(server_module.app)
    with client.websocket_connect("/ws/session") as ws:
        _recv(ws)  # puzzle (discarded)
        # Read raw messages (not via _recv) until an eeg_chunk shows up.
        for _ in range(20):
            msg = ws.receive_json()
            if msg.get("type") == "eeg_chunk":
                assert isinstance(msg["samples"], list)
                assert len(msg["samples"]) > 0
                assert len(msg["samples"][0]) == NUM_CHANNELS
                return
        raise AssertionError("no eeg_chunk message arrived")

def test_websocket_survives_malformed_messages(tmp_path, monkeypatch):
    _install_test_server(tmp_path, monkeypatch)

    client = TestClient(server_module.app)
    with client.websocket_connect("/ws/session") as ws:
        puzzle_msg = _recv(ws)
        assert puzzle_msg["type"] == "puzzle"
        token = puzzle_msg["attempt_token"]

        # Missing move_uci -> KeyError path (checked before the token, same either way).
        ws.send_json({"time_to_move": 3.0, "attempt_token": token})
        err = _recv(ws)
        assert err["type"] == "error"
        assert "Malformed move" in err["message"]

        # Non-numeric time_to_move -> ValueError path.
        ws.send_json({"move_uci": "a1a8", "time_to_move": "soon", "attempt_token": token})
        err = _recv(ws)
        assert err["type"] == "error"

        # A same-square "move" (a1a1) is never even a legal move -> invalid_move, and
        # the puzzle does NOT advance: no new puzzle should follow this.
        ws.send_json({"move_uci": "a1a1", "time_to_move": 2.0, "attempt_token": token})
        invalid_msg = _recv(ws)
        assert invalid_msg["type"] == "invalid_move"
        assert invalid_msg["reason"]

        # The SAME puzzle/token is still live: answering it correctly now must work.
        ws.send_json({
            "move_uci": _solution_for(puzzle_msg["puzzle_id"]), "time_to_move": 4.0,
            "attempt_token": token,
        })
        update_msg = _recv(ws)
        assert update_msg["type"] == "update"
        assert update_msg["correct"] is True
        assert update_msg["predicted_state"] in STATES

        # Connection is still alive and still serving puzzles afterward.
        next_puzzle = _recv(ws)
        assert next_puzzle["type"] == "puzzle"

def test_illegal_move_does_not_advance_or_score(tmp_path, monkeypatch):
    store = _install_test_server(tmp_path, monkeypatch)

    client = TestClient(server_module.app)
    with client.websocket_connect("/ws/session") as ws:
        puzzle_msg = _recv(ws)
        session_id = puzzle_msg["session_id"]

        # Deterministically illegal regardless of which puzzle was served: find a
        # genuinely empty square from the puzzle's own FEN and "move" from it.
        board = chess.Board(puzzle_msg["fen"])
        empty_square = next(sq for sq in chess.SQUARES if board.piece_at(sq) is None)
        empty_name = chess.square_name(empty_square)
        destination = "a1" if empty_name != "a1" else "a2"

        ws.send_json({
            "move_uci": f"{empty_name}{destination}", "time_to_move": 1.0,
            "attempt_token": puzzle_msg["attempt_token"],
        })
        msg = _recv(ws)
        assert msg["type"] == "invalid_move"
        assert "no piece" in msg["reason"].lower()

        # Nothing was logged for a rejected, unscored attempt, and the puzzle is
        # still the same one (no "puzzle" message follows).
        summary = store.get_session_summary(session_id)
        assert summary["num_attempts"] == 0

def test_hint_returns_legal_targets_for_selected_square(tmp_path, monkeypatch):
    _install_test_server(tmp_path, monkeypatch)

    client = TestClient(server_module.app)
    with client.websocket_connect("/ws/session") as ws:
        puzzle_msg = _recv(ws)
        solution = _solution_for(puzzle_msg["puzzle_id"])
        from_square = solution[:2]

        ws.send_json({"type": "hint", "square": from_square})
        hint_msg = _recv(ws)
        assert hint_msg["type"] == "hint"
        assert hint_msg["square"] == from_square
        # The solution's destination must always be among the legal targets for its
        # own source square (the solution move is always legal, by construction).
        assert solution[2:4] in hint_msg["targets"]

def test_hint_for_empty_square_returns_no_targets(tmp_path, monkeypatch):
    _install_test_server(tmp_path, monkeypatch)

    client = TestClient(server_module.app)
    with client.websocket_connect("/ws/session") as ws:
        _recv(ws)  # puzzle
        ws.send_json({"type": "hint", "square": "z9"})
        hint_msg = _recv(ws)
        assert hint_msg["type"] == "hint"
        assert hint_msg["targets"] == []

def test_check_move_explains_an_illegal_move_without_scoring(tmp_path, monkeypatch):
    store = _install_test_server(tmp_path, monkeypatch)

    client = TestClient(server_module.app)
    with client.websocket_connect("/ws/session") as ws:
        puzzle_msg = _recv(ws)
        ws.send_json({"type": "check_move", "move_uci": "a1a1"})
        msg = _recv(ws)
        assert msg["type"] == "invalid_move"
        assert isinstance(msg["reason"], str) and len(msg["reason"]) > 0

        summary = store.get_session_summary(puzzle_msg["session_id"])
        assert summary["num_attempts"] == 0  # a check_move query is never scored

def test_multi_move_puzzle_auto_plays_opponent_and_continues(tmp_path, monkeypatch):
    _install_test_server(tmp_path, monkeypatch)
    _fix_served_puzzle(monkeypatch, Puzzle(
        puzzle_id="mm01", fen="3qk3/6pp/8/8/8/8/8/R5K1 w - - 0 1",
        solution_move="a1a8", solution_moves=["a1a8", "h7h6", "a8d8"], rating=1200,
    ))

    client = TestClient(server_module.app)
    with client.websocket_connect("/ws/session") as ws:
        puzzle_msg = _recv(ws)
        assert puzzle_msg["puzzle_id"] == "mm01"
        assert puzzle_msg["solver_color"] == "white"

        ws.send_json({
            "move_uci": "a1a8", "time_to_move": 2.0, "attempt_token": puzzle_msg["attempt_token"],
        })
        update_msg = _recv(ws)
        assert update_msg["type"] == "update"
        assert update_msg["status"] == "continue"
        assert update_msg["correct"] is True
        assert update_msg["opponent_move"] == "h7h6"

        # Same puzzle continues: puzzle_id unchanged, board reflects the opponent's
        # auto-played reply (the h7 pawn has moved to h6), fresh attempt_token.
        next_msg = _recv(ws)
        assert next_msg["type"] == "puzzle"
        assert next_msg["puzzle_id"] == "mm01"
        assert next_msg["attempt_token"] != puzzle_msg["attempt_token"]
        board_after_opponent = chess.Board(next_msg["fen"])
        assert board_after_opponent.piece_at(chess.H6) is not None
        assert board_after_opponent.piece_at(chess.H7) is None

        ws.send_json({
            "move_uci": "a8d8", "time_to_move": 2.0, "attempt_token": next_msg["attempt_token"],
        })
        final_update = _recv(ws)
        assert final_update["type"] == "update"
        assert final_update["status"] == "solved"
        assert final_update["correct"] is True
        assert final_update["opponent_move"] is None

        brand_new_puzzle = _recv(ws)
        assert brand_new_puzzle["type"] == "puzzle"

def test_wrong_legal_move_resets_puzzle_for_retry(tmp_path, monkeypatch):
    _install_test_server(tmp_path, monkeypatch)

    client = TestClient(server_module.app)
    with client.websocket_connect("/ws/session") as ws:
        puzzle_msg = _recv(ws)
        solution = _solution_for(puzzle_msg["puzzle_id"])
        board = chess.Board(puzzle_msg["fen"])
        wrong_move = next(m.uci() for m in board.legal_moves if m.uci() != solution)

        ws.send_json({
            "move_uci": wrong_move, "time_to_move": 2.0, "attempt_token": puzzle_msg["attempt_token"],
        })
        update_msg = _recv(ws)
        assert update_msg["type"] == "update"
        assert update_msg["status"] == "retry"
        assert update_msg["correct"] is False

        retry_puzzle = _recv(ws)
        assert retry_puzzle["type"] == "puzzle"
        assert retry_puzzle["puzzle_id"] == puzzle_msg["puzzle_id"]
        assert retry_puzzle["fen"] == puzzle_msg["fen"]  # reset to the original position
        assert retry_puzzle["attempt_token"] != puzzle_msg["attempt_token"]

        # The reset puzzle is still fully playable with the real solution.
        ws.send_json({
            "move_uci": solution, "time_to_move": 2.0, "attempt_token": retry_puzzle["attempt_token"],
        })
        final_update = _recv(ws)
        assert final_update["type"] == "update"
        assert final_update["correct"] is True

def test_solver_color_reported_for_black_to_move_puzzle(tmp_path, monkeypatch):
    _install_test_server(tmp_path, monkeypatch)
    _fix_served_puzzle(monkeypatch, Puzzle(
        puzzle_id="blacktest", fen="4k3/8/8/8/8/8/8/4K3 b - - 0 1",
        solution_move="e8d8", solution_moves=["e8d8"], rating=1000,
    ))

    client = TestClient(server_module.app)
    with client.websocket_connect("/ws/session") as ws:
        puzzle_msg = _recv(ws)
        assert puzzle_msg["solver_color"] == "black"

        # "Own" pieces for this puzzle are the lowercase (Black) ones — legal_targets
        # for the black king's square must be non-empty, confirming turn/color handling
        # is consistent end to end, not just in the reported label.
        ws.send_json({"type": "hint", "square": "e8"})
        hint_msg = _recv(ws)
        assert hint_msg["targets"] == ["d7", "d8", "e7", "f7", "f8"]

def test_missing_checkpoint_reports_error_instead_of_hanging(tmp_path, monkeypatch):
    monkeypatch.setattr(server_module, "DEFAULT_CHECKPOINT", tmp_path / "absent.pt")
    monkeypatch.setattr(server_module, "_store", SessionStore(":memory:"))

    client = TestClient(server_module.app)
    with client.websocket_connect("/ws/session") as ws:
        msg = _recv(ws)
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
