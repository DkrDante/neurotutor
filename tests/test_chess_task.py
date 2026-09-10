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
