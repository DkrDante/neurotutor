from __future__ import annotations
import csv
from pathlib import Path
import chess
from chess_task.base import TaskEngine, Puzzle, BehaviorEvent
from chess_task.evaluator import MoveEvaluator

DEFAULT_PUZZLE_CSV = Path(__file__).parent / "puzzle_data" / "sample_puzzles.csv"

class PuzzleTaskEngine(TaskEngine):
    def __init__(self, evaluator: MoveEvaluator, puzzle_csv: Path = DEFAULT_PUZZLE_CSV):
        self.evaluator = evaluator
        self.puzzles = self._load_puzzles(puzzle_csv)
        self._served: set[str] = set()

    @staticmethod
    def _load_puzzles(path: Path) -> list[Puzzle]:
        puzzles = []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                puzzles.append(Puzzle(
                    puzzle_id=row["puzzle_id"],
                    fen=row["fen"],
                    solution_move=row["solution_move"],
                    rating=int(row["rating"]),
                ))
        return puzzles

    def get_puzzle(self, difficulty: float) -> Puzzle:
        available = [p for p in self.puzzles if p.puzzle_id not in self._served]
        if not available:
            self._served.clear()
            available = self.puzzles
        best = min(available, key=lambda p: abs(p.rating - difficulty))
        self._served.add(best.puzzle_id)
        return best

    def submit_move(self, puzzle: Puzzle, move_uci: str, time_to_move: float) -> BehaviorEvent:
        board = chess.Board(puzzle.fen)
        played_move = chess.Move.from_uci(move_uci)
        best_move = chess.Move.from_uci(puzzle.solution_move)
        correct = played_move == best_move
        eval_loss = 0.0 if correct else self.evaluator.eval_loss(board, played_move, best_move)
        return BehaviorEvent(
            puzzle_id=puzzle.puzzle_id,
            correct=correct,
            time_to_move=time_to_move,
            eval_loss=eval_loss,
            puzzle_rating=puzzle.rating,
        )
