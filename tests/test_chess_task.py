import chess
import chess.engine
import pytest
from chess_task.base import Puzzle, BehaviorEvent
from chess_task.evaluator import MoveEvaluator
from chess_task.puzzles import PuzzleTaskEngine

class FakeEvaluator(MoveEvaluator):
    def eval_loss(self, board, played_move, best_move) -> float:
        return 250.0

def test_load_puzzles_from_csv():
    engine = PuzzleTaskEngine(evaluator=FakeEvaluator())
    # 10 original hand-built mate-in-1 puzzles + a real sample pulled from the public
    # Lichess puzzle database (single-move puzzles only, see chess_task/puzzle_data/).
    assert len(engine.puzzles) > 300

def test_get_puzzle_picks_nearest_rating():
    engine = PuzzleTaskEngine(evaluator=FakeEvaluator())
    expected = min(engine.puzzles, key=lambda p: abs(p.rating - 1000))
    puzzle = engine.get_puzzle(difficulty=1000)
    assert puzzle.puzzle_id == expected.puzzle_id

def test_get_puzzle_does_not_repeat_until_exhausted():
    engine = PuzzleTaskEngine(evaluator=FakeEvaluator())
    total = len(engine.puzzles)
    seen = {engine.get_puzzle(1000).puzzle_id for _ in range(total)}
    assert len(seen) == total

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

class _StubEngine:
    """Stands in for chess.engine.SimpleEngine so this runs without the binary."""
    def __init__(self):
        self.analyse_calls = 0
        self.quit_calls = 0

    def analyse(self, board, limit):
        self.analyse_calls += 1
        return {"score": chess.engine.PovScore(chess.engine.Cp(10), chess.WHITE)}

    def quit(self):
        self.quit_calls += 1

def test_stockfish_evaluator_opens_one_engine_and_reuses_it(monkeypatch):
    """Regression guard: _score() used to spawn a fresh subprocess on every call."""
    import chess_task.evaluator as evaluator_module

    spawns = []
    stub = _StubEngine()

    monkeypatch.setattr(evaluator_module.shutil, "which", lambda _binary: "/fake/stockfish")
    monkeypatch.setattr(
        evaluator_module.chess.engine.SimpleEngine, "popen_uci",
        staticmethod(lambda binary: (spawns.append(binary), stub)[1]),
    )

    evaluator = evaluator_module.StockfishEvaluator()
    assert len(spawns) == 1  # opened once in __init__

    board = chess.Board("2k5/1ppp4/8/8/8/8/8/R6K w - - 0 1")
    for _ in range(3):
        evaluator.eval_loss(board, chess.Move.from_uci("h1g1"), chess.Move.from_uci("a1a8"))

    assert len(spawns) == 1  # ...and never reopened, despite 6 analyse() calls
    assert stub.analyse_calls == 6

    evaluator.close()
    assert stub.quit_calls == 1
