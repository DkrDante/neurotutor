import chess
import pytest
from chess_task.base import Puzzle, BehaviorEvent
from chess_task.evaluator import MoveEvaluator
from chess_task.puzzles import PuzzleTaskEngine

class FakeEvaluator(MoveEvaluator):
    def eval_loss(self, board, played_move, best_move) -> float:
        return 250.0

def test_load_puzzles_from_csv():
    engine = PuzzleTaskEngine(evaluator=FakeEvaluator())
    assert len(engine.puzzles) == 10

def test_get_puzzle_picks_nearest_rating():
    engine = PuzzleTaskEngine(evaluator=FakeEvaluator())
    puzzle = engine.get_puzzle(difficulty=1000)
    assert puzzle.puzzle_id == "rb03"  # rating 960, closest to 1000

def test_get_puzzle_does_not_repeat_until_exhausted():
    engine = PuzzleTaskEngine(evaluator=FakeEvaluator())
    seen = {engine.get_puzzle(1000).puzzle_id for _ in range(10)}
    assert len(seen) == 10

def test_submit_move_correct():
    engine = PuzzleTaskEngine(evaluator=FakeEvaluator())
    puzzle = Puzzle(puzzle_id="rb01", fen="2k5/1ppp4/8/8/8/8/8/R6K w - - 0 1", solution_move="a1a8", rating=700)
    event = engine.submit_move(puzzle, "a1a8", time_to_move=3.0)
    assert isinstance(event, BehaviorEvent)
    assert event.correct is True
    assert event.eval_loss == 0.0

def test_submit_move_incorrect_uses_evaluator():
    engine = PuzzleTaskEngine(evaluator=FakeEvaluator())
    puzzle = Puzzle(puzzle_id="rb01", fen="2k5/1ppp4/8/8/8/8/8/R6K w - - 0 1", solution_move="a1a8", rating=700)
    event = engine.submit_move(puzzle, "h1g1", time_to_move=3.0)
    assert event.correct is False
    assert event.eval_loss == 250.0

class ExplodingEvaluator(MoveEvaluator):
    def eval_loss(self, board, played_move, best_move) -> float:
        raise AssertionError("evaluator must never be called with an illegal move")

@pytest.mark.parametrize("bad_move", ["not-a-move", "", "z9z9", "a1a1a1a1", "a1"])
def test_submit_move_invalid_uci_is_incorrect_and_does_not_raise(bad_move):
    engine = PuzzleTaskEngine(evaluator=ExplodingEvaluator())
    puzzle = Puzzle(puzzle_id="rb01", fen="2k5/1ppp4/8/8/8/8/8/R6K w - - 0 1", solution_move="a1a8", rating=700)
    event = engine.submit_move(puzzle, bad_move, time_to_move=3.0)
    assert event.correct is False
    assert event.eval_loss == PuzzleTaskEngine.ILLEGAL_MOVE_PENALTY

@pytest.mark.parametrize("illegal_move", ["a1a1", "h1a8", "e7e5"])
def test_submit_move_legal_uci_but_illegal_move_is_incorrect(illegal_move):
    """Syntactically valid UCI that is not legal in this position (a1a1 is the
    'clicked the same square twice' case). Must not reach the evaluator, whose
    _score() would assert inside board.push()."""
    engine = PuzzleTaskEngine(evaluator=ExplodingEvaluator())
    puzzle = Puzzle(puzzle_id="rb01", fen="2k5/1ppp4/8/8/8/8/8/R6K w - - 0 1", solution_move="a1a8", rating=700)
    event = engine.submit_move(puzzle, illegal_move, time_to_move=3.0)
    assert event.correct is False
    assert event.eval_loss == PuzzleTaskEngine.ILLEGAL_MOVE_PENALTY

def test_null_evaluator_close_is_a_noop():
    FakeEvaluator().close()  # MoveEvaluator provides a default no-op close()
