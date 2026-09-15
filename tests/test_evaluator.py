import chess
from chess_task.evaluator import MATERIAL_VALUES, MoveEvaluator, win_probability

class _MaterialOnlyEvaluator(MoveEvaluator):
    """Exercises MoveEvaluator.evaluate_position()'s base (material-only)
    implementation without needing a Stockfish binary."""
    def eval_loss(self, board, played_move, best_move) -> float:
        return 0.0

def test_evaluate_position_starting_position_is_balanced():
    evaluator = _MaterialOnlyEvaluator()
    assert evaluator.evaluate_position(chess.Board()) == 0.0

def test_evaluate_position_reflects_material_deficit():
    # White is down a queen relative to the starting position (Black's is intact).
    board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1")
    evaluator = _MaterialOnlyEvaluator()
    assert evaluator.evaluate_position(board) == -MATERIAL_VALUES[chess.QUEEN]

def test_win_probability_is_50_50_at_even_material():
    assert abs(win_probability(0.0) - 0.5) < 1e-9

def test_win_probability_favors_white_with_positive_eval():
    assert win_probability(500.0) > 0.5

def test_win_probability_favors_black_with_negative_eval():
    assert win_probability(-500.0) < 0.5

def test_win_probability_stays_within_bounds():
    assert 0.0 < win_probability(-3000.0) < 0.01
    assert 0.99 < win_probability(3000.0) < 1.0

def test_best_move_prefers_the_highest_value_capture():
    # White can capture either a pawn (b6) or a queen (d7) with the same knight.
    board = chess.Board("7k/3q4/1p6/4N3/8/8/8/7K w - - 0 1")
    evaluator = _MaterialOnlyEvaluator()
    assert evaluator.best_move(board) == chess.Move.from_uci("e5d7")

def test_best_move_falls_back_to_a_legal_move_with_no_captures_available():
    board = chess.Board("7k/8/8/8/8/8/8/R6K w - - 0 1")
    evaluator = _MaterialOnlyEvaluator()
    move = evaluator.best_move(board)
    assert move in board.legal_moves
