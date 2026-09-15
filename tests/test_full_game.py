import random
import chess
from chess_task.evaluator import MoveEvaluator
from chess_task.full_game import (
    CORRECT_EVAL_LOSS_THRESHOLD, FullGameEngine, _MAX_THINKING_SECONDS, _MIN_THINKING_SECONDS,
)

class _NoOpEvaluator(MoveEvaluator):
    """Never charges a real eval_loss — isolates FullGameEngine's own logic
    (grading threshold, board bookkeeping, opponent-strength scaling) from the
    base MoveEvaluator's material-only heuristics, which have their own tests."""
    def eval_loss(self, board, played_move, best_move) -> float:
        return 999.0  # always "wrong" unless the move equals best_move exactly

def test_get_puzzle_reflects_the_live_board_not_a_fixed_position():
    engine = FullGameEngine(_NoOpEvaluator())
    puzzle = engine.get_puzzle(difficulty=1000.0)
    assert puzzle.fen == chess.Board().fen()
    assert chess.Move.from_uci(puzzle.solution_move) in chess.Board().legal_moves
    assert puzzle.rating == 1000

def test_submit_move_is_correct_when_it_matches_best_move_exactly():
    engine = FullGameEngine(_NoOpEvaluator())
    puzzle = engine.get_puzzle(difficulty=1000.0)
    event = engine.submit_move(puzzle, puzzle.solution_move, time_to_move=2.0)
    assert event.correct is True
    assert event.eval_loss == 0.0

def test_submit_move_is_wrong_when_eval_loss_exceeds_threshold():
    engine = FullGameEngine(_NoOpEvaluator())
    puzzle = engine.get_puzzle(difficulty=1000.0)
    board = chess.Board(puzzle.fen)
    other_move = next(m for m in board.legal_moves if m.uci() != puzzle.solution_move)
    event = engine.submit_move(puzzle, other_move.uci(), time_to_move=2.0)
    assert event.correct is False
    assert event.eval_loss > CORRECT_EVAL_LOSS_THRESHOLD

def test_push_player_move_advances_the_live_board_and_ply_count():
    engine = FullGameEngine(_NoOpEvaluator())
    engine.push_player_move("e2e4")
    assert engine.board.fen().split(" ")[0] != chess.Board().fen().split(" ")[0]
    assert engine.board.turn == chess.BLACK
    next_puzzle = engine.get_puzzle(difficulty=1000.0)
    assert next_puzzle.puzzle_id == "game-ply-1"

def test_pick_opponent_move_returns_the_only_legal_move_when_forced():
    # King and rook vs lone king, black to move with exactly one legal reply.
    engine = FullGameEngine(_NoOpEvaluator(), starting_fen="7k/8/8/8/8/8/8/R3K3 b - - 0 1")
    move = engine.pick_opponent_move(difficulty=1000.0)
    assert move in engine.board.legal_moves

def test_pick_opponent_move_always_plays_best_move_at_max_strength():
    # A position with exactly one capture available makes best_move()
    # deterministic (the base evaluator falls back to a RANDOM legal move
    # whenever there's no capture to prefer, which starting-position boards
    # don't have) — needed so the two best_move() calls below are comparable.
    engine = FullGameEngine(_NoOpEvaluator(), starting_fen="7k/8/8/8/4P3/2n5/8/7K b - - 0 1", rng=random.Random(0))
    move = engine.pick_opponent_move(difficulty=2600.0)  # strength clamps to 1.0
    assert move == engine.evaluator.best_move(engine.board)
    assert move == chess.Move.from_uci("c3e4")

def test_pick_opponent_move_never_forced_to_best_move_at_floor_difficulty():
    # At the difficulty floor, strength is 0.0 — pick_opponent_move must never
    # take the `< strength` branch regardless of the RNG draw.
    engine = FullGameEngine(_NoOpEvaluator(), rng=random.Random(0))
    for _ in range(20):
        move = engine.pick_opponent_move(difficulty=400.0)
        assert move in engine.board.legal_moves

def test_push_opponent_move_advances_board_and_ply_count():
    engine = FullGameEngine(_NoOpEvaluator())
    move = engine.pick_opponent_move(difficulty=1000.0)
    engine.push_opponent_move(move)
    assert engine.board.turn == chess.BLACK
    assert engine.get_puzzle(difficulty=1000.0).puzzle_id == "game-ply-1"

def test_thinking_time_is_at_minimum_at_the_difficulty_floor():
    engine = FullGameEngine(_NoOpEvaluator())
    assert engine.thinking_time(difficulty=400.0) == _MIN_THINKING_SECONDS

def test_thinking_time_is_at_maximum_at_full_strength():
    engine = FullGameEngine(_NoOpEvaluator())
    assert engine.thinking_time(difficulty=2600.0) == _MAX_THINKING_SECONDS

def test_thinking_time_increases_with_difficulty():
    engine = FullGameEngine(_NoOpEvaluator())
    assert engine.thinking_time(difficulty=1500.0) > engine.thinking_time(difficulty=800.0)
