import chess
from chess_task.move_explainer import (
    describe_move, explain_illegal_move, explain_suboptimal_move, legal_targets, score_move,
)

ROOK_BOARD = "2k5/1ppp4/8/8/8/8/8/R6K w - - 0 1"

def test_legal_targets_for_rook():
    board = chess.Board(ROOK_BOARD)
    targets = legal_targets(board, "a1")
    # The rook can slide anywhere along the empty a-file and 1st rank.
    assert "a8" in targets
    assert "h1" not in targets  # own king is there

def test_legal_targets_empty_square_is_empty_list():
    board = chess.Board(ROOK_BOARD)
    assert legal_targets(board, "e4") == []

def test_legal_targets_opponent_piece_is_empty_list():
    board = chess.Board(ROOK_BOARD)
    assert legal_targets(board, "c8") == []  # black king, not white's turn to move it

def test_legal_targets_invalid_square_name_is_empty_list():
    board = chess.Board(ROOK_BOARD)
    assert legal_targets(board, "z9") == []

def test_explain_no_piece_on_square():
    board = chess.Board(ROOK_BOARD)
    assert "no piece" in explain_illegal_move(board, "e4e5").lower()

def test_explain_not_your_piece():
    board = chess.Board(ROOK_BOARD)
    assert "not your piece" in explain_illegal_move(board, "c8c7").lower()

def test_explain_own_piece_on_destination():
    board = chess.Board(ROOK_BOARD)
    assert "already have a piece" in explain_illegal_move(board, "a1h1").lower()

def test_explain_blocked_path():
    # Black pawn on a3 blocks the rook's a1-a8 file (a8 itself is an empty target).
    board = chess.Board("7k/8/8/8/8/p7/8/R3K3 w - - 0 1")
    reason = explain_illegal_move(board, "a1a8")
    assert "blocking" in reason.lower()

def test_explain_piece_cant_move_like_that():
    board = chess.Board(ROOK_BOARD)
    # King on h1 cannot jump three squares to h4 (empty, not adjacent).
    reason = explain_illegal_move(board, "h1h4")
    assert "king" in reason.lower()

def test_explain_unparseable_move():
    board = chess.Board(ROOK_BOARD)
    assert "real move" in explain_illegal_move(board, "not-a-move").lower()

def test_suboptimal_missed_mate():
    board = chess.Board("6k1/5ppp/8/8/8/8/7K/R7 w - - 0 1")
    reason = explain_suboptimal_move(board, "h2h3", "a1a8")
    assert "mate" in reason.lower()

def test_suboptimal_hangs_a_piece():
    board = chess.Board("7k/8/8/1n6/8/8/8/3Q3K w - - 0 1")
    reason = explain_suboptimal_move(board, "d1d4", "d1d2")
    assert "hangs your queen" in reason.lower()

def test_suboptimal_misses_bigger_capture():
    # Knight can take a pawn on c6 (safely) or a rook on d7 (also safely, and worth
    # more) — playing the pawn capture should be flagged against the rook capture.
    board = chess.Board("7k/3r4/2p5/4N3/8/8/8/7K w - - 0 1")
    reason = explain_suboptimal_move(board, "e5c6", "e5d7")
    assert "rook" in reason.lower()

def test_suboptimal_misses_a_check():
    board = chess.Board("7k/8/8/8/8/8/8/R6K w - - 0 1")
    reason = explain_suboptimal_move(board, "a1a4", "a1a8")
    assert "check" in reason.lower()

def test_suboptimal_generic_fallback_names_solution_move():
    board = chess.Board("7k/8/8/8/8/8/8/N6K w - - 0 1")
    reason = explain_suboptimal_move(board, "a1b3", "a1c2")
    assert "better move" in reason.lower()
    assert "c2" in reason.lower()

def test_score_move_is_100_at_zero_eval_loss():
    result = score_move(0.0)
    assert result == {"score": 100, "label": "Best"}

def test_score_move_bands_match_familiar_chess_labels():
    assert score_move(10.0)["label"] == "Excellent"
    assert score_move(50.0)["label"] == "Good"
    assert score_move(100.0)["label"] == "Inaccuracy"
    assert score_move(250.0)["label"] == "Mistake"
    assert score_move(500.0)["label"] == "Blunder"

def test_score_move_is_floored_at_zero_for_severe_blunders():
    result = score_move(1000.0)
    assert result["score"] == 0
    assert result["label"] == "Blunder"

def test_describe_move_reports_piece_and_squares():
    board = chess.Board()
    description = describe_move(board, "e2e4")
    assert "pawn" in description.lower()
    assert "e2" in description
    assert "e4" in description

def test_describe_move_reports_a_capture():
    board = chess.Board("7k/8/8/8/8/8/1p6/N6K w - - 0 1")
    description = describe_move(board, "a1b2")
    assert "capturing" in description.lower()
    assert "pawn" in description.lower()

def test_describe_move_reports_check():
    board = chess.Board("7k/8/8/8/8/8/8/R6K w - - 0 1")
    description = describe_move(board, "a1a8")
    assert "check" in description.lower()

def test_suboptimal_high_tier_is_more_verbose_than_low_tier():
    board = chess.Board("7k/8/8/1n6/8/8/8/3Q3K w - - 0 1")
    high = explain_suboptimal_move(board, "d1d4", "d1d2", tier="high")
    low = explain_suboptimal_move(board, "d1d4", "d1d2", tier="low")
    assert "undefended" in high.lower() and "queen" in high.lower()
    assert len(high) > len(low)
    assert "queen" in low.lower()

def test_suboptimal_default_tier_matches_medium():
    board = chess.Board("7k/8/8/1n6/8/8/8/3Q3K w - - 0 1")
    default = explain_suboptimal_move(board, "d1d4", "d1d2")
    medium = explain_suboptimal_move(board, "d1d4", "d1d2", tier="medium")
    assert default == medium

def test_suboptimal_missed_mate_tiers_all_name_the_mate():
    board = chess.Board("6k1/5ppp/8/8/8/8/7K/R7 w - - 0 1")
    for tier in ("high", "medium", "low"):
        reason = explain_suboptimal_move(board, "h2h3", "a1a8", tier=tier)
        assert "mate" in reason.lower()

def test_reveal_solution_false_redacts_the_solution_notation():
    board = chess.Board("6k1/5ppp/8/8/8/8/7K/R7 w - - 0 1")
    hidden = explain_suboptimal_move(board, "h2h3", "a1a8", reveal_solution=False)
    assert "a8" not in hidden.lower()
    assert "solution move" in hidden.lower()
    assert "mate" in hidden.lower()  # the WHY is still explained

def test_reveal_solution_true_still_names_the_move_by_default():
    board = chess.Board("6k1/5ppp/8/8/8/8/7K/R7 w - - 0 1")
    shown = explain_suboptimal_move(board, "h2h3", "a1a8")
    assert "a8" in shown.lower()

def test_reveal_solution_false_does_not_affect_the_hangs_piece_branch():
    # This branch never names the solution move to begin with (it describes
    # the PLAYED move's flaw), so redaction should leave it untouched.
    board = chess.Board("7k/8/8/1n6/8/8/8/3Q3K w - - 0 1")
    shown = explain_suboptimal_move(board, "d1d4", "d1d2", reveal_solution=True)
    hidden = explain_suboptimal_move(board, "d1d4", "d1d2", reveal_solution=False)
    assert shown == hidden

def test_reveal_solution_false_redacts_across_all_tiers():
    board = chess.Board("7k/3r4/2p5/4N3/8/8/8/7K w - - 0 1")
    for tier in ("high", "medium", "low"):
        hidden = explain_suboptimal_move(board, "e5c6", "e5d7", tier=tier, reveal_solution=False)
        assert "d7" not in hidden.lower()
