import json
import time
from pathlib import Path
import chess
import httpx
import pytest
import torch
from fastapi.testclient import TestClient
from model.fusion import FusionLSTMClassifier
from model.evidence import DEFAULT_CHECKPOINT as REAL_TRAINED_CHECKPOINT
from common.config import NUM_CHANNELS, BAND_NAMES, NUM_BEHAVIOR_FEATS
from common.states import STATES
import web.server as server_module
from accounts.storage import AccountStore
from storage.db import SessionStore
from chess_task.base import Puzzle
from chess_task.puzzles import PuzzleTaskEngine
from chess_task.full_game import FullGameEngine
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
        def get_puzzle(self, difficulty, rating_min=None, rating_max=None):
            return puzzle
    monkeypatch.setattr(
        server_module, "PuzzleTaskEngine",
        lambda evaluator, already_served=None: FixedPuzzleEngine(evaluator=evaluator, already_served=already_served),
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
    monkeypatch.setattr(server_module, "_account_store", AccountStore(":memory:"))
    return store

@pytest.mark.skipif(
    not REAL_TRAINED_CHECKPOINT.exists(), reason="model/checkpoints/best.pt not trained in this checkout",
)
def test_real_trained_checkpoint_serves_a_valid_websocket_session(tmp_path, monkeypatch):
    """_install_test_server (used by every other websocket test in this file)
    always points DEFAULT_CHECKPOINT at a freshly-initialized, untrained model
    — sufficient for exercising the wiring, but it never proves the ACTUAL
    checkpoint this app loads in production behaves correctly once wired into
    a live FastAPI/WebSocket session. This test points DEFAULT_CHECKPOINT at
    the real trained checkpoint instead, everything else identical."""
    store = _install_test_server(tmp_path, monkeypatch)
    monkeypatch.setattr(server_module, "DEFAULT_CHECKPOINT", REAL_TRAINED_CHECKPOINT)

    client = TestClient(server_module.app)
    with client.websocket_connect("/ws/session?learner_id=real-checkpoint-test") as ws:
        puzzle_msg = _recv(ws)
        ws.send_json({
            "move_uci": _solution_for(puzzle_msg["puzzle_id"]), "time_to_move": 5.0,
            "attempt_token": puzzle_msg["attempt_token"],
        })
        update_msg = _recv(ws)

    assert update_msg["type"] == "update"
    assert update_msg["predicted_state"] in STATES
    assert 0.0 <= update_msg["confidence"] <= 1.0

    profile = store.get_learner_profile("real-checkpoint-test")
    assert profile is not None

def test_websocket_session_round_trip(tmp_path, monkeypatch):
    _install_test_server(tmp_path, monkeypatch)

    client = TestClient(server_module.app)
    with client.websocket_connect("/ws/session") as ws:
        puzzle_msg = _recv(ws)
        assert puzzle_msg["type"] == "puzzle"
        assert "fen" in puzzle_msg
        assert "session_id" in puzzle_msg
        assert "attempt_token" in puzzle_msg
        assert isinstance(puzzle_msg["rating"], int)
        # Win-probability eval bar data — a real (material-only, absent Stockfish)
        # position evaluation, not a placeholder 50/50.
        assert isinstance(puzzle_msg["eval_cp"], float)
        assert 0.0 <= puzzle_msg["white_win_prob"] <= 1.0

        ws.send_json({
            "move_uci": _solution_for(puzzle_msg["puzzle_id"]), "time_to_move": 5.0,
            "attempt_token": puzzle_msg["attempt_token"],
        })
        update_msg = _recv(ws)
        assert update_msg["type"] == "update"
        assert update_msg["correct"] is True
        assert update_msg["predicted_state"] in STATES
        assert 0.0 <= update_msg["confidence"] <= 1.0
        # SAN notation for the move list — a non-empty algebraic string, not just the
        # raw UCI the client already has.
        assert update_msg["solver_san"]
        assert isinstance(update_msg["solver_san"], str)
        assert isinstance(update_msg["eval_cp"], float)
        assert 0.0 <= update_msg["white_win_prob"] <= 1.0

        # Real internals for the network-activity visualization, not placeholders.
        activity = update_msg["network_activity"]
        assert len(activity["gcn1_node_activity"]) == NUM_CHANNELS
        assert len(activity["gcn1_adjacency"]) == NUM_CHANNELS
        assert len(activity["gcn2_adjacency"]) == NUM_CHANNELS
        assert len(activity["classifier_weight"]) == len(STATES)
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
        assert update_msg["solver_san"] == "Ra8"
        assert update_msg["opponent_san"] == "h6"

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
        # A concrete, non-generic explanation of why the legal move wasn't the
        # solution (chess_task.move_explainer.explain_suboptimal_move), not just a
        # flat "wrong" flag.
        assert update_msg["reason"]
        # The played (wrong) move's own SAN is still reported, so the frontend can
        # show/explain its notation even though it wasn't the solution.
        assert update_msg["played_san"]
        assert update_msg["solver_san"] is None
        # A wrong move now also carries a 0-100 quality score + familiar label
        # (chess_task.move_explainer.score_move) and a factual one-line
        # description of what the move itself did.
        assert 0 <= update_msg["move_score"] <= 100
        assert update_msg["move_quality_label"] in {"Best", "Excellent", "Good", "Inaccuracy", "Mistake", "Blunder"}
        assert update_msg["move_description"]

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

def test_hand_holding_tier_thresholds():
    assert server_module.hand_holding_tier(400.0) == "high"
    assert server_module.hand_holding_tier(699.0) == "high"
    assert server_module.hand_holding_tier(700.0) == "medium"
    assert server_module.hand_holding_tier(1000.0) == "medium"
    assert server_module.hand_holding_tier(1099.0) == "medium"
    assert server_module.hand_holding_tier(1100.0) == "low"
    assert server_module.hand_holding_tier(2000.0) == "low"

def test_wrong_move_explanation_scales_with_adaptive_difficulty(tmp_path, monkeypatch):
    # A struggling learner (low adaptive difficulty) should get the fuller,
    # more didactic explanation for the SAME wrong move a cruising learner
    # (high difficulty) gets a terse one-liner for.
    store = _install_test_server(tmp_path, monkeypatch)
    _fix_served_puzzle(monkeypatch, Puzzle(
        puzzle_id="hangtest", fen="7k/8/8/1n6/8/8/8/3Q3K w - - 0 1",
        solution_move="d1d2", solution_moves=["d1d2"], rating=1000,
    ))
    client = TestClient(server_module.app)

    store.save_learner_profile("struggling-learner", difficulty=500.0)
    with client.websocket_connect("/ws/session?learner_id=struggling-learner") as ws:
        puzzle_msg = _recv(ws)
        ws.send_json({
            "move_uci": "d1d4", "time_to_move": 2.0, "attempt_token": puzzle_msg["attempt_token"],
        })
        high_tier_msg = _recv(ws)

    store.save_learner_profile("cruising-learner", difficulty=1500.0)
    with client.websocket_connect("/ws/session?learner_id=cruising-learner") as ws:
        puzzle_msg = _recv(ws)
        ws.send_json({
            "move_uci": "d1d4", "time_to_move": 2.0, "attempt_token": puzzle_msg["attempt_token"],
        })
        low_tier_msg = _recv(ws)

    assert len(high_tier_msg["reason"]) > len(low_tier_msg["reason"])
    assert "undefended" in high_tier_msg["reason"].lower()
    # Same wrong move on the same position -> the same eval_loss-derived score,
    # regardless of which explanation tier it's paired with.
    assert high_tier_msg["move_quality_label"] == low_tier_msg["move_quality_label"]
    assert high_tier_msg["move_score"] == low_tier_msg["move_score"]

def test_wrong_move_reason_withholds_the_solution_by_default(tmp_path, monkeypatch):
    _install_test_server(tmp_path, monkeypatch)
    _fix_served_puzzle(monkeypatch, Puzzle(
        puzzle_id="matetest", fen="6k1/5ppp/8/8/8/8/7K/R7 w - - 0 1",
        solution_move="a1a8", solution_moves=["a1a8"], rating=1000,
    ))
    client = TestClient(server_module.app)
    with client.websocket_connect("/ws/session") as ws:
        puzzle_msg = _recv(ws)
        ws.send_json({
            "move_uci": "h2h3", "time_to_move": 2.0, "attempt_token": puzzle_msg["attempt_token"],
        })
        update_msg = _recv(ws)
        assert update_msg["status"] == "retry"
        assert update_msg["solution_revealed"] is False
        assert "a8" not in update_msg["reason"].lower()
        assert "mate" in update_msg["reason"].lower()  # the WHY is still explained

def test_solution_button_reveals_the_full_move_and_unlocks_reason_text(tmp_path, monkeypatch):
    _install_test_server(tmp_path, monkeypatch)
    _fix_served_puzzle(monkeypatch, Puzzle(
        puzzle_id="matetest2", fen="6k1/5ppp/8/8/8/8/7K/R7 w - - 0 1",
        solution_move="a1a8", solution_moves=["a1a8"], rating=1000,
    ))
    client = TestClient(server_module.app)
    with client.websocket_connect("/ws/session") as ws:
        puzzle_msg = _recv(ws)

        # "Solution" (renamed from "Hint") now reveals the real move, not just
        # which piece to move.
        ws.send_json({"type": "puzzle_hint"})
        hint_msg = _recv(ws)
        assert hint_msg["type"] == "puzzle_hint"
        assert hint_msg["square"] == "a1"
        assert hint_msg["target_square"] == "a8"
        assert "a8" in hint_msg["text"].lower()

        ws.send_json({
            "move_uci": "h2h3", "time_to_move": 2.0, "attempt_token": puzzle_msg["attempt_token"],
        })
        update_msg = _recv(ws)
        assert update_msg["solution_revealed"] is True
        assert "a8" in update_msg["reason"].lower()

def test_solution_auto_reveals_after_repeated_wrong_attempts_on_same_puzzle(tmp_path, monkeypatch):
    _install_test_server(tmp_path, monkeypatch)
    _fix_served_puzzle(monkeypatch, Puzzle(
        puzzle_id="matetest3", fen="6k1/5ppp/8/8/8/8/7K/R7 w - - 0 1",
        solution_move="a1a8", solution_moves=["a1a8"], rating=1000,
    ))
    client = TestClient(server_module.app)
    with client.websocket_connect("/ws/session") as ws:
        puzzle_msg = _recv(ws)
        revealed_flags = []
        for _ in range(server_module.AUTO_REVEAL_AFTER_CONSECUTIVE_WRONG):
            ws.send_json({
                "move_uci": "h2h3", "time_to_move": 2.0, "attempt_token": puzzle_msg["attempt_token"],
            })
            update_msg = _recv(ws)
            revealed_flags.append(update_msg["solution_revealed"])
            puzzle_msg = _recv(ws)  # the retry's fresh puzzle message + attempt_token

        # Not revealed on the first attempts, auto-revealed once the threshold
        # of consecutive misses on this SAME puzzle is reached.
        assert revealed_flags[:-1] == [False] * (len(revealed_flags) - 1)
        assert revealed_flags[-1] is True

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

def test_puzzle_hint_reveals_solution_piece_without_scoring(tmp_path, monkeypatch):
    _install_test_server(tmp_path, monkeypatch)
    _fix_served_puzzle(monkeypatch, Puzzle(
        puzzle_id="hinttest", fen="4k3/8/8/8/8/8/8/4K3 b - - 0 1",
        solution_move="e8d8", solution_moves=["e8d8"], rating=1000,
    ))

    client = TestClient(server_module.app)
    with client.websocket_connect("/ws/session") as ws:
        puzzle_msg = _recv(ws)

        ws.send_json({"type": "puzzle_hint"})
        hint_msg = _recv(ws)
        assert hint_msg["type"] == "puzzle_hint"
        assert hint_msg["square"] == "e8"
        assert "king" in hint_msg["text"].lower()

        # The hint request must not have consumed/altered this decision point: the
        # real solution still scores correctly afterward with the original token.
        ws.send_json({
            "move_uci": "e8d8", "time_to_move": 2.0, "attempt_token": puzzle_msg["attempt_token"],
        })
        update_msg = _recv(ws)
        assert update_msg["type"] == "update"
        assert update_msg["correct"] is True

def test_missing_checkpoint_reports_error_instead_of_hanging(tmp_path, monkeypatch):
    monkeypatch.setattr(server_module, "DEFAULT_CHECKPOINT", tmp_path / "absent.pt")
    monkeypatch.setattr(server_module, "_store", SessionStore(":memory:"))

    client = TestClient(server_module.app)
    with client.websocket_connect("/ws/session") as ws:
        msg = _recv(ws)
        assert msg["type"] == "error"
        assert "checkpoint not found" in msg["message"].lower()

def test_model_evidence_page_serves_html(tmp_path, monkeypatch):
    _install_test_server(tmp_path, monkeypatch)
    client = TestClient(server_module.app)
    response = client.get("/model-evidence")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

def test_puzzles_page_serves_html(tmp_path, monkeypatch):
    _install_test_server(tmp_path, monkeypatch)
    client = TestClient(server_module.app)
    response = client.get("/puzzles")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

def test_api_puzzles_returns_the_real_curated_set(tmp_path, monkeypatch):
    _install_test_server(tmp_path, monkeypatch)
    client = TestClient(server_module.app)

    response = client.get("/api/puzzles")
    assert response.status_code == 200
    body = response.json()

    assert body["total"] > 300
    assert len(body["puzzles"]) == body["total"]

    sources = {p["source"] for p in body["puzzles"]}
    assert sources == {"hand-built", "lichess"}

    # Every row has real, usable fields — not placeholders.
    sample = body["puzzles"][0]
    assert set(sample) == {"puzzle_id", "fen", "solution_move", "solution_moves", "rating", "source", "motif"}
    assert isinstance(sample["rating"], int)
    assert sample["solution_move"] in sample["solution_moves"]

    from chess_task.motifs import MOTIFS
    assert all(p["motif"] in MOTIFS for p in body["puzzles"])

    # A known hand-built puzzle and a known real Lichess-format puzzle both
    # classify correctly by id shape (see api_puzzles' docstring).
    by_id = {p["puzzle_id"]: p for p in body["puzzles"]}
    assert by_id["rb01"]["source"] == "hand-built"
    assert by_id["00SeK"]["source"] == "lichess"
    assert by_id["rb01"]["motif"] == "back_rank"

def test_model_evidence_api_returns_weights_metrics_and_trace(tmp_path, monkeypatch):
    _install_test_server(tmp_path, monkeypatch)
    client = TestClient(server_module.app)

    response = client.get("/api/model-evidence?split=test&sample_index=0")
    assert response.status_code == 200
    body = response.json()

    assert body["checkpoint_path"]
    assert len(body["weights"]) > 0
    assert body["metrics"]["split"] == "test"
    assert body["metrics"]["num_samples"] > 0
    assert body["trace"]["sample_index"] == 0
    assert "softmax_probabilities" in body["trace"]

    # No history.json exists for the fresh, untrained test checkpoint
    # _install_test_server writes — must report None, not crash.
    assert body["training_history"] is None

    baseline_names = {b["name"] for b in body["baselines"]}
    assert baseline_names == {"majority_class", "logistic_regression_behavior_only"}
    for baseline in body["baselines"]:
        assert 0.0 <= baseline["test_accuracy"] <= 1.0

    importance = body["feature_importance"]
    assert 0.0 <= importance["baseline_accuracy"] <= 1.0
    assert {r["feature"] for r in importance["behavior_importance"]} == {
        "correct", "time_to_move", "eval_loss", "puzzle_rating",
    }
    assert len(importance["eeg_channel_importance"]) == NUM_CHANNELS

def test_model_evidence_api_returns_training_history_when_present(tmp_path, monkeypatch):
    store = _install_test_server(tmp_path, monkeypatch)
    # _install_test_server already pointed DEFAULT_CHECKPOINT at a real file;
    # write a real history.json next to it the same way train.py would.
    history_file = server_module.evidence.history_path_for(server_module.DEFAULT_CHECKPOINT)
    history_file.write_text(json.dumps([
        {"epoch": 1, "train_loss": 1.5, "val_accuracy": 0.3, "val_f1": 0.25},
        {"epoch": 2, "train_loss": 1.1, "val_accuracy": 0.5, "val_f1": 0.45},
    ]))
    client = TestClient(server_module.app)

    response = client.get("/api/model-evidence")
    assert response.status_code == 200
    history = response.json()["training_history"]
    assert len(history) == 2
    assert history[0]["epoch"] == 1
    assert history[1]["val_accuracy"] == 0.5

def test_model_evidence_api_404_without_a_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(server_module, "DEFAULT_CHECKPOINT", tmp_path / "absent.pt")
    client = TestClient(server_module.app)
    response = client.get("/api/model-evidence")
    assert response.status_code == 404

def test_model_evidence_api_400_for_out_of_range_sample_index(tmp_path, monkeypatch):
    _install_test_server(tmp_path, monkeypatch)
    client = TestClient(server_module.app)
    response = client.get("/api/model-evidence?sample_index=999999")
    assert response.status_code == 400

def test_analytics_aggregate_endpoint_reports_cohort_stats_and_policy_comparison(tmp_path, monkeypatch):
    store = _install_test_server(tmp_path, monkeypatch)
    client = TestClient(server_module.app)

    session_id = store.create_session()
    store.log_attempt(
        session_id=session_id, puzzle_id="rb01", correct=True, time_to_move=4.0,
        eval_loss=0.0, puzzle_rating=700, predicted_state="Focused", confidence=0.8,
        difficulty_delta=100.0, show_hint=False, pacing_delay=0.0, sense_to_adapt_latency=0.02,
        mode="puzzle",
    )

    response = client.get("/analytics/aggregate")
    assert response.status_code == 200
    body = response.json()
    puzzle = body["puzzle"]
    assert puzzle["mode"] == "puzzle"
    assert puzzle["total_sessions"] == 1
    assert puzzle["total_attempts"] == 1
    assert "state_distribution" in puzzle
    assert "engagement_trend_avg" in puzzle
    comparison = puzzle["policy_comparison"]
    assert comparison["total_attempts"] == 1
    assert comparison["divergent_attempts"] in (0, 1)
    assert 0.0 <= comparison["divergence_rate"] <= 1.0
    # No game-mode attempts logged — that side of the split must report zero,
    # not silently borrow the puzzle-mode numbers above.
    assert body["game"]["total_attempts"] == 0

def test_analytics_aggregate_endpoint_handles_no_sessions(tmp_path, monkeypatch):
    _install_test_server(tmp_path, monkeypatch)
    client = TestClient(server_module.app)
    response = client.get("/analytics/aggregate")
    assert response.status_code == 200
    body = response.json()
    assert body["puzzle"]["total_sessions"] == 0
    assert body["puzzle"]["policy_comparison"]["divergence_rate"] is None
    assert body["game"]["total_sessions"] == 0

def test_analytics_aggregate_endpoint_never_mixes_puzzle_and_game_attempts(tmp_path, monkeypatch):
    store = _install_test_server(tmp_path, monkeypatch)
    client = TestClient(server_module.app)

    puzzle_session = store.create_session()
    store.log_attempt(
        session_id=puzzle_session, puzzle_id="rb01", correct=True, time_to_move=4.0,
        eval_loss=0.0, puzzle_rating=700, predicted_state="Focused", confidence=0.8,
        difficulty_delta=100.0, show_hint=False, pacing_delay=0.0, sense_to_adapt_latency=0.02,
        mode="puzzle",
    )
    game_session = store.create_session()
    store.log_attempt(
        session_id=game_session, puzzle_id="game-ply-0", correct=False, time_to_move=12.0,
        eval_loss=0.0, puzzle_rating=400, predicted_state="Confused", confidence=0.7,
        difficulty_delta=-50.0, show_hint=True, pacing_delay=1.0, sense_to_adapt_latency=0.03,
        mode="game",
    )

    body = client.get("/analytics/aggregate").json()

    assert body["puzzle"]["total_attempts"] == 1
    assert body["puzzle"]["overall_accuracy"] == 1.0
    assert body["game"]["total_attempts"] == 1
    assert body["game"]["overall_accuracy"] == 0.0

def test_sessions_export_returns_csv_with_all_attempts(tmp_path, monkeypatch):
    store = _install_test_server(tmp_path, monkeypatch)
    client = TestClient(server_module.app)
    session_id = store.create_session()
    store.log_attempt(
        session_id=session_id, puzzle_id="rb01", correct=True, time_to_move=4.0,
        eval_loss=0.0, puzzle_rating=700, predicted_state="Focused", confidence=0.8,
        difficulty_delta=100.0, show_hint=False, pacing_delay=0.0, sense_to_adapt_latency=0.02,
    )

    response = client.get("/sessions/export")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    body = response.text
    assert "session_id" in body.splitlines()[0]
    assert session_id in body
    assert "rb01" in body

def test_session_export_scoped_to_one_session(tmp_path, monkeypatch):
    store = _install_test_server(tmp_path, monkeypatch)
    client = TestClient(server_module.app)
    session_id = store.create_session()
    store.log_attempt(
        session_id=session_id, puzzle_id="rb02", correct=False, time_to_move=6.0,
        eval_loss=120.0, puzzle_rating=900, predicted_state="Confused", confidence=0.6,
        difficulty_delta=-50.0, show_hint=True, pacing_delay=1.5, sense_to_adapt_latency=0.04,
    )

    response = client.get(f"/session/{session_id}/export")
    assert response.status_code == 200
    assert "rb02" in response.text
    assert "session_id" not in response.text.splitlines()[0]  # scoped export omits the column

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

def test_analytics_page_is_served(tmp_path, monkeypatch):
    _install_test_server(tmp_path, monkeypatch)
    client = TestClient(server_module.app)
    response = client.get("/analytics")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

def test_list_sessions_endpoint_returns_sessions_with_attempts(tmp_path, monkeypatch):
    store = _install_test_server(tmp_path, monkeypatch)
    client = TestClient(server_module.app)

    empty_session = store.create_session()  # never plays a puzzle — must be excluded
    played_session = store.create_session()
    store.log_attempt(
        session_id=played_session, puzzle_id="rb01", correct=True, time_to_move=4.0,
        eval_loss=0.0, puzzle_rating=700, predicted_state="Focused", confidence=0.8,
        difficulty_delta=100.0, show_hint=False, pacing_delay=0.0, sense_to_adapt_latency=0.02,
    )

    body = client.get("/sessions").json()
    session_ids = [s["session_id"] for s in body]
    assert played_session in session_ids
    assert empty_session not in session_ids

def test_session_analytics_endpoint_returns_summary_attempts_and_calculations(tmp_path, monkeypatch):
    store = _install_test_server(tmp_path, monkeypatch)
    client = TestClient(server_module.app)

    session_id = store.create_session()
    store.log_attempt(
        session_id=session_id, puzzle_id="rb01", correct=True, time_to_move=4.0,
        eval_loss=0.0, puzzle_rating=700, predicted_state="Focused", confidence=0.8,
        difficulty_delta=100.0, show_hint=False, pacing_delay=0.0, sense_to_adapt_latency=0.02,
    )

    body = client.get(f"/session/{session_id}/analytics").json()
    assert set(body) == {"summary", "attempts", "calculations"}
    assert body["summary"]["num_attempts"] == 1
    assert len(body["attempts"]) == 1
    assert body["attempts"][0]["puzzle_id"] == "rb01"
    # Real formulas, not placeholder text — every note must actually name a formula.
    assert len(body["calculations"]) >= 4
    for note in body["calculations"]:
        assert set(note) == {"title", "formula", "explanation"}
        assert note["formula"]

def test_session_coach_endpoint_returns_real_tips_without_ollama_running(tmp_path, monkeypatch):
    store = _install_test_server(tmp_path, monkeypatch)
    # This dev machine has a real Ollama server running — force the "not
    # reachable" path explicitly rather than relying on the ambient
    # environment, so this test is deterministic wherever it runs.
    def raise_connection_error(*args, **kwargs):
        raise httpx.ConnectError("connection refused")
    monkeypatch.setattr(httpx, "post", raise_connection_error)
    client = TestClient(server_module.app)

    session_id = store.create_session()
    # A rough-looking rating band on purpose, so a tip is guaranteed to fire.
    for _ in range(5):
        store.log_attempt(
            session_id=session_id, puzzle_id="p", correct=True, time_to_move=3.0,
            eval_loss=0.0, puzzle_rating=900, predicted_state="Focused", confidence=0.9,
            difficulty_delta=100.0, show_hint=False, pacing_delay=0.0, sense_to_adapt_latency=0.02,
        )
    for _ in range(5):
        store.log_attempt(
            session_id=session_id, puzzle_id="p", correct=False, time_to_move=3.0,
            eval_loss=300.0, puzzle_rating=2100, predicted_state="Focused", confidence=0.9,
            difficulty_delta=-100.0, show_hint=False, pacing_delay=0.0, sense_to_adapt_latency=0.02,
        )

    body = client.get(f"/session/{session_id}/coach").json()
    assert set(body) == {"statistics", "tips", "narrative", "targeted_practice"}
    assert body["statistics"]["num_attempts"] == 10
    assert body["narrative"] is None  # no Ollama server running in this test environment
    assert len(body["tips"]) >= 1
    assert any("1800-2199" in tip for tip in body["tips"])
    # A weak rating band was detected (the 2100-rated puzzles), so the coach
    # response also tells the frontend exactly what a targeted-practice
    # reconnect should ask for.
    assert body["targeted_practice"] == {"rating_min": 1800, "rating_max": 2199, "label": "1800-2199"}

def test_session_coach_endpoint_for_empty_session(tmp_path, monkeypatch):
    store = _install_test_server(tmp_path, monkeypatch)
    client = TestClient(server_module.app)
    session_id = store.create_session()

    body = client.get(f"/session/{session_id}/coach").json()
    assert body["statistics"]["num_attempts"] == 0
    assert len(body["tips"]) == 1

def test_session_calculations_endpoint_returns_formulas_aggregate_and_trace(tmp_path, monkeypatch):
    store = _install_test_server(tmp_path, monkeypatch)
    client = TestClient(server_module.app)
    session_id = store.create_session()
    # Focused + confidence 0.9 (trusted): correctness delta + STATE_RULES["Focused"] (+100).
    store.log_attempt(
        session_id=session_id, puzzle_id="p1", correct=True, time_to_move=3.0,
        eval_loss=0.0, puzzle_rating=1000, predicted_state="Focused", confidence=0.9,
        difficulty_delta=200.0, show_hint=False, pacing_delay=0.0, sense_to_adapt_latency=0.02,
    )
    store.log_attempt(
        session_id=session_id, puzzle_id="p2", correct=False, time_to_move=6.0,
        eval_loss=250.0, puzzle_rating=1000, predicted_state="Focused", confidence=0.9,
        difficulty_delta=0.0, show_hint=False, pacing_delay=0.0, sense_to_adapt_latency=0.02,
    )

    body = client.get(f"/session/{session_id}/calculations").json()
    assert set(body) == {"formulas", "num_attempts", "aggregate", "attempts"}
    assert body["num_attempts"] == 2
    assert len(body["formulas"]) >= 4

    accuracy_calc = next(c for c in body["aggregate"] if c["title"] == "Overall accuracy")
    assert accuracy_calc["substitution"] == "1 / 2"
    assert accuracy_calc["result"] == "50.0%"

    trace = body["attempts"]
    assert len(trace) == 2
    assert trace[0]["difficulty_before"] == 1000.0
    assert trace[0]["difficulty_after"] == 1200.0
    assert trace[1]["difficulty_after"] == 1200.0
    assert trace[0]["rule_based_reconstruction"]["matches_actual"] is True

def test_session_calculations_endpoint_for_empty_session(tmp_path, monkeypatch):
    store = _install_test_server(tmp_path, monkeypatch)
    client = TestClient(server_module.app)
    session_id = store.create_session()

    body = client.get(f"/session/{session_id}/calculations").json()
    assert body["num_attempts"] == 0
    assert body["aggregate"] == []
    assert body["attempts"] == []
    assert len(body["formulas"]) >= 4

def test_calculations_page_is_served(tmp_path, monkeypatch):
    _install_test_server(tmp_path, monkeypatch)
    client = TestClient(server_module.app)
    response = client.get("/calculations")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

def test_learn_page_is_served(tmp_path, monkeypatch):
    _install_test_server(tmp_path, monkeypatch)
    client = TestClient(server_module.app)
    response = client.get("/learn")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

def test_auth_me_reports_logged_out_with_no_cookie(tmp_path, monkeypatch):
    _install_test_server(tmp_path, monkeypatch)
    client = TestClient(server_module.app)
    body = client.get("/auth/me").json()
    assert body == {"logged_in": False}

def test_signup_logs_in_and_seeds_progress_from_anonymous_state(tmp_path, monkeypatch):
    _install_test_server(tmp_path, monkeypatch)
    client = TestClient(server_module.app)

    response = client.post("/auth/signup", json={
        "username": "alice", "password": "hunter22",
        "progress": {"total_xp": 340, "solved_count": 12, "unlocked_achievements": ["first-blood"]},
    })
    assert response.status_code == 200
    body = response.json()
    assert body["logged_in"] is True
    assert body["username"] == "alice"
    assert body["total_xp"] == 340
    assert body["unlocked_achievements"] == ["first-blood"]
    assert "neurotutor_session" in response.cookies

    # The session cookie set by signup keeps this client logged in.
    me = client.get("/auth/me").json()
    assert me["logged_in"] is True
    assert me["username"] == "alice"

def test_signup_rejects_short_username_and_short_password(tmp_path, monkeypatch):
    _install_test_server(tmp_path, monkeypatch)
    client = TestClient(server_module.app)
    assert client.post("/auth/signup", json={"username": "ab", "password": "hunter22"}).status_code == 400
    assert client.post("/auth/signup", json={"username": "alice", "password": "short"}).status_code == 400

def test_signup_rejects_duplicate_username(tmp_path, monkeypatch):
    _install_test_server(tmp_path, monkeypatch)
    client = TestClient(server_module.app)
    client.post("/auth/signup", json={"username": "alice", "password": "hunter22"})
    response = client.post("/auth/signup", json={"username": "alice", "password": "different1"})
    assert response.status_code == 409

def test_login_with_correct_and_incorrect_password(tmp_path, monkeypatch):
    _install_test_server(tmp_path, monkeypatch)
    client = TestClient(server_module.app)
    client.post("/auth/signup", json={"username": "alice", "password": "hunter22"})

    wrong = client.post("/auth/login", json={"username": "alice", "password": "wrongpass"})
    assert wrong.status_code == 401

    right = client.post("/auth/login", json={"username": "alice", "password": "hunter22"})
    assert right.status_code == 200
    assert right.json()["logged_in"] is True

def test_logout_clears_session_so_me_reports_logged_out(tmp_path, monkeypatch):
    _install_test_server(tmp_path, monkeypatch)
    client = TestClient(server_module.app)
    client.post("/auth/signup", json={"username": "alice", "password": "hunter22"})
    assert client.get("/auth/me").json()["logged_in"] is True

    client.post("/auth/logout")
    assert client.get("/auth/me").json()["logged_in"] is False

def test_account_progress_requires_login(tmp_path, monkeypatch):
    _install_test_server(tmp_path, monkeypatch)
    client = TestClient(server_module.app)
    response = client.post("/account/progress", json={"total_xp": 100})
    assert response.status_code == 401

def test_account_progress_updates_when_logged_in(tmp_path, monkeypatch):
    _install_test_server(tmp_path, monkeypatch)
    client = TestClient(server_module.app)
    client.post("/auth/signup", json={"username": "alice", "password": "hunter22"})

    response = client.post("/account/progress", json={
        "total_xp": 500, "solved_count": 20, "puzzles_served": 27,
        "unlocked_achievements": ["century", "streak-2"],
    })
    assert response.status_code == 200

    me = client.get("/auth/me").json()
    assert me["total_xp"] == 500
    assert me["solved_count"] == 20
    assert me["puzzles_served"] == 27
    assert me["unlocked_achievements"] == ["century", "streak-2"]

def test_account_checkin_requires_login(tmp_path, monkeypatch):
    _install_test_server(tmp_path, monkeypatch)
    client = TestClient(server_module.app)
    assert client.post("/account/checkin").status_code == 401

def test_account_checkin_builds_a_daily_streak(tmp_path, monkeypatch):
    _install_test_server(tmp_path, monkeypatch)
    client = TestClient(server_module.app)
    client.post("/auth/signup", json={"username": "alice", "password": "hunter22"})

    first = client.post("/account/checkin").json()
    assert first["current_streak"] == 1
    assert first["changed_today"] is True

    # Checking in again today (whatever "today" is when the test runs) is a no-op.
    second = client.post("/account/checkin").json()
    assert second["current_streak"] == 1
    assert second["changed_today"] is False

def test_leaderboard_is_public_and_ordered_by_xp(tmp_path, monkeypatch):
    _install_test_server(tmp_path, monkeypatch)
    client = TestClient(server_module.app)
    client.post("/auth/signup", json={"username": "alice", "password": "hunter22", "progress": {"total_xp": 100}})

    # A second, unauthenticated client (no session cookie) can still read it.
    anon_client = TestClient(server_module.app)
    body = anon_client.get("/leaderboard").json()
    assert body["your_rank"] is None
    assert body["entries"][0]["username"] == "alice"
    assert body["entries"][0]["total_xp"] == 100

def test_leaderboard_reports_the_caller_own_rank_when_logged_in(tmp_path, monkeypatch):
    _install_test_server(tmp_path, monkeypatch)
    client = TestClient(server_module.app)
    client.post("/auth/signup", json={"username": "alice", "password": "hunter22", "progress": {"total_xp": 50}})

    other_client = TestClient(server_module.app)
    other_client.post("/auth/signup", json={"username": "bob", "password": "hunter22", "progress": {"total_xp": 500}})

    body = client.get("/leaderboard").json()
    assert body["your_rank"] == 2  # bob (500 xp) ranks above alice (50 xp)

def test_game_mode_starts_from_the_standard_opening_position(tmp_path, monkeypatch):
    _install_test_server(tmp_path, monkeypatch)
    client = TestClient(server_module.app)
    with client.websocket_connect("/ws/session?mode=game") as ws:
        first_puzzle = _recv(ws)
        assert first_puzzle["type"] == "puzzle"
        assert first_puzzle["mode"] == "game"
        assert first_puzzle["fen"] == chess.Board().fen()
        assert first_puzzle["solver_color"] == "white"

def test_game_mode_never_resets_a_move_unlike_puzzle_mode_retry(tmp_path, monkeypatch):
    # Deterministic evaluator: legal_moves[0] is always "best", so playing
    # anything else is guaranteed graded wrong — and game mode must still keep
    # that move on the board rather than resetting to the start position.
    class _FirstLegalMoveIsBestEvaluator(MoveEvaluator):
        def eval_loss(self, board, played_move, best_move) -> float:
            return 999.0
        def best_move(self, board):
            return list(board.legal_moves)[0]

    _install_test_server(tmp_path, monkeypatch)
    monkeypatch.setattr(server_module, "build_evaluator", lambda: _FirstLegalMoveIsBestEvaluator())
    # Don't actually wait through the AI's simulated "thinking" pause in tests.
    monkeypatch.setattr(FullGameEngine, "thinking_time", lambda self, difficulty: 0.0)

    client = TestClient(server_module.app)
    with client.websocket_connect("/ws/session?mode=game") as ws:
        first_puzzle = _recv(ws)
        legal_moves = list(chess.Board(first_puzzle["fen"]).legal_moves)
        wrong_move = legal_moves[1].uci()  # deliberately not legal_moves[0], the "best" move

        ws.send_json({
            "move_uci": wrong_move, "time_to_move": 1.0, "attempt_token": first_puzzle["attempt_token"],
        })
        update = _recv(ws)
        assert update["type"] == "update"
        assert update["correct"] is False
        assert update["reason"]
        assert update["status"] == "game_move"
        assert update["awaiting_opponent"] is True

        # The opponent's reply is a separate, later message (simulating "thinking"
        # time) rather than bundled into the same "update" — see
        # chess_task.full_game.FullGameEngine.thinking_time.
        opponent_reply = _recv(ws)
        assert opponent_reply["type"] == "opponent_move"

        next_puzzle = _recv(ws)
        assert next_puzzle["type"] == "puzzle"
        assert next_puzzle["fen"] != first_puzzle["fen"]  # advanced, not reset to the start

def test_game_mode_ends_with_game_over_status_on_checkmate(tmp_path, monkeypatch):
    class _MateInOneEvaluator(MoveEvaluator):
        def eval_loss(self, board, played_move, best_move) -> float:
            return 0.0
        def best_move(self, board):
            return chess.Move.from_uci("h5f7")  # Qxf7# — scholar's-mate finish

    _install_test_server(tmp_path, monkeypatch)
    monkeypatch.setattr(server_module, "build_evaluator", lambda: _MateInOneEvaluator())

    # White queen h5 and bishop c4 already bearing on f7, black has just
    # blundered — one legal white move (Qxf7#) ends the game immediately.
    monkeypatch.setattr(
        server_module, "FullGameEngine",
        lambda evaluator: FullGameEngine(
            evaluator, starting_fen="rnbqkbnr/pppp1ppp/8/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 0 1",
        ),
    )

    client = TestClient(server_module.app)
    with client.websocket_connect("/ws/session?mode=game") as ws:
        first_puzzle = _recv(ws)
        ws.send_json({
            "move_uci": "h5f7", "time_to_move": 1.0, "attempt_token": first_puzzle["attempt_token"],
        })
        update = _recv(ws)
        assert update["type"] == "update"
        assert update["status"] == "game_over"
        assert update["correct"] is True
        assert update["game_result"] == "1-0"

def test_game_mode_reveals_opponent_move_in_a_separate_delayed_message(tmp_path, monkeypatch):
    _install_test_server(tmp_path, monkeypatch)
    # A real (non-zero) thinking time to prove the split is actually a
    # sequencing thing, not just "two messages sent back to back instantly".
    monkeypatch.setattr(FullGameEngine, "thinking_time", lambda self, difficulty: 0.05)

    client = TestClient(server_module.app)
    with client.websocket_connect("/ws/session?mode=game") as ws:
        first_puzzle = _recv(ws)
        board = chess.Board(first_puzzle["fen"])
        move = next(iter(board.legal_moves))

        ws.send_json({
            "move_uci": move.uci(), "time_to_move": 1.0, "attempt_token": first_puzzle["attempt_token"],
        })
        update = _recv(ws)
        assert update["type"] == "update"
        # The player's own result is known immediately — the opponent hasn't
        # "replied" yet from the client's point of view.
        assert update["opponent_move"] is None
        assert update["opponent_san"] is None
        assert update["awaiting_opponent"] is True

        opponent_reply = _recv(ws)
        assert opponent_reply["type"] == "opponent_move"
        assert opponent_reply["move"]
        assert opponent_reply["san"]
        assert opponent_reply["status"] in ("game_move", "game_over")
        assert "fen" in opponent_reply  # the resulting position, for the frontend to reconcile against

def test_rl_policy_query_param_persists_a_learner_profile_with_q_table(tmp_path, monkeypatch):
    store = _install_test_server(tmp_path, monkeypatch)
    client = TestClient(server_module.app)
    learner_id = "learner-rl-test"

    with client.websocket_connect(f"/ws/session?policy=rl&learner_id={learner_id}") as ws:
        puzzle_msg = _recv(ws)
        ws.send_json({
            "move_uci": _solution_for(puzzle_msg["puzzle_id"]), "time_to_move": 5.0,
            "attempt_token": puzzle_msg["attempt_token"],
        })
        update_msg = _recv(ws)
        assert update_msg["type"] == "update"

    profile = store.get_learner_profile(learner_id)
    assert profile is not None
    assert profile["policy_state"] is not None
    assert "q_table" in profile["policy_state"]
    assert profile["difficulty"] == update_msg["difficulty"]

def test_targeted_practice_query_params_constrain_puzzle_rating(tmp_path, monkeypatch):
    _install_test_server(tmp_path, monkeypatch)
    client = TestClient(server_module.app)

    with client.websocket_connect("/ws/session?target_min=1800&target_max=2199") as ws:
        for _ in range(5):
            puzzle_msg = _recv(ws)
            assert 1800 <= puzzle_msg["rating"] <= 2199
            solution = _solution_for(puzzle_msg["puzzle_id"])
            ws.send_json({
                "move_uci": solution, "time_to_move": 1.0, "attempt_token": puzzle_msg["attempt_token"],
            })
            _recv(ws)  # the "update" message for this solved attempt

def test_puzzle_reconnect_does_not_immediately_repeat_a_served_puzzle(tmp_path, monkeypatch):
    # Regression test: a fresh PuzzleTaskEngine is constructed per websocket
    # connection, so without seeding its no-repeat tracking from the learner's
    # persisted profile, a reconnect (a page refresh included) would silently
    # forget every puzzle already shown and could re-serve the exact same one.
    store = _install_test_server(tmp_path, monkeypatch)
    client = TestClient(server_module.app)
    learner_id = "learner-reconnect-test"

    with client.websocket_connect(f"/ws/session?learner_id={learner_id}") as ws:
        first_puzzle = _recv(ws)
        ws.send_json({
            "move_uci": _solution_for(first_puzzle["puzzle_id"]), "time_to_move": 5.0,
            "attempt_token": first_puzzle["attempt_token"],
        })
        _recv(ws)  # the "update" message, which triggers persist_profile()

    profile = store.get_learner_profile(learner_id)
    assert first_puzzle["puzzle_id"] in profile["served_puzzle_ids"]

    with client.websocket_connect(f"/ws/session?learner_id={learner_id}") as ws:
        second_puzzle = _recv(ws)

    assert second_puzzle["puzzle_id"] != first_puzzle["puzzle_id"]

def test_full_game_mode_personalizes_difficulty_across_connections(tmp_path, monkeypatch):
    store = _install_test_server(tmp_path, monkeypatch)
    monkeypatch.setattr(FullGameEngine, "thinking_time", lambda self, difficulty: 0.0)
    client = TestClient(server_module.app)
    learner_id = "learner-personalization-test"

    with client.websocket_connect(f"/ws/session?mode=game&learner_id={learner_id}") as ws:
        first_puzzle = _recv(ws)
        assert first_puzzle["rating"] == int(server_module.DEFAULT_STARTING_DIFFICULTY)
        board = chess.Board(first_puzzle["fen"])
        move = next(iter(board.legal_moves))
        ws.send_json({
            "move_uci": move.uci(), "time_to_move": 1.0, "attempt_token": first_puzzle["attempt_token"],
        })
        update = _recv(ws)
        new_difficulty = update["difficulty"]
        assert new_difficulty != server_module.DEFAULT_STARTING_DIFFICULTY

    assert store.get_learner_profile(learner_id)["difficulty"] == new_difficulty

    with client.websocket_connect(f"/ws/session?mode=game&learner_id={learner_id}") as ws:
        second_puzzle = _recv(ws)
        assert second_puzzle["rating"] == int(new_difficulty)

def test_full_game_attempts_are_logged_with_game_mode_not_puzzle(tmp_path, monkeypatch):
    # Regression test for the puzzle/full-game data-mixing bug: full-game
    # attempts must be tagged mode="game" so rating-band coaching stats and
    # /analytics/aggregate's cohort overview never treat the live-difficulty
    # number full-game mode logs as puzzle_rating as if it were a real
    # puzzle rating.
    store = _install_test_server(tmp_path, monkeypatch)
    monkeypatch.setattr(FullGameEngine, "thinking_time", lambda self, difficulty: 0.0)
    client = TestClient(server_module.app)

    with client.websocket_connect("/ws/session?mode=game") as ws:
        session_id = None
        puzzle_msg = _recv(ws)
        session_id = puzzle_msg["session_id"]
        board = chess.Board(puzzle_msg["fen"])
        move = next(iter(board.legal_moves))
        ws.send_json({
            "move_uci": move.uci(), "time_to_move": 1.0, "attempt_token": puzzle_msg["attempt_token"],
        })
        _recv(ws)

    attempts = store.get_session_attempts(session_id)
    assert len(attempts) == 1
    assert attempts[0]["mode"] == "game"
