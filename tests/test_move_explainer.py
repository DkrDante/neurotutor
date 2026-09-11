import chess
from chess_task.move_explainer import explain_illegal_move, legal_targets

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
