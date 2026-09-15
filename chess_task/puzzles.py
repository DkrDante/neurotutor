from __future__ import annotations
import csv
from pathlib import Path
import chess
from chess_task.base import TaskEngine, Puzzle, BehaviorEvent
from chess_task.evaluator import MoveEvaluator

DEFAULT_PUZZLE_CSV = Path(__file__).parent / "puzzle_data" / "sample_puzzles.csv"

class PuzzleTaskEngine(TaskEngine):
    # Eval loss charged for an unparseable or illegal move. Deliberately larger than a
    # realistic Stockfish centipawn loss so it reads as "worst possible attempt".
    ILLEGAL_MOVE_PENALTY = 500.0

    def __init__(
        self, evaluator: MoveEvaluator, puzzle_csv: Path = DEFAULT_PUZZLE_CSV,
        already_served: set[str] | None = None,
    ):
        self.evaluator = evaluator
        self.puzzles = self._load_puzzles(puzzle_csv)
        # Seeded from a returning learner's persisted profile (web/server.py)
        # so a reconnect — a page refresh included — doesn't immediately
        # re-serve puzzles already shown this "cycle": without this, a fresh
        # PuzzleTaskEngine per connection means the no-repeat tracking below
        # only ever worked within a single connection, not across one.
        self._served: set[str] = set(already_served) if already_served else set()

    @property
    def served_puzzle_ids(self) -> set[str]:
        return self._served

    @staticmethod
    def _load_puzzles(path: Path) -> list[Puzzle]:
        puzzles = []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # "solution_move" holds one or more space-separated UCI moves: a single
                # move for the original single-ply puzzles, or the full solver/opponent/
                # solver/... sequence for a multi-move puzzle. No CSV schema change needed.
                solution_moves = row["solution_move"].split()
                puzzles.append(Puzzle(
                    puzzle_id=row["puzzle_id"],
                    fen=row["fen"],
                    solution_move=solution_moves[0],
                    solution_moves=solution_moves,
                    rating=int(row["rating"]),
                ))
        return puzzles

    def get_puzzle(self, difficulty: float, rating_min: float | None = None, rating_max: float | None = None) -> Puzzle:
        candidates = self.puzzles
        if rating_min is not None or rating_max is not None:
            lo = rating_min if rating_min is not None else 0
            hi = rating_max if rating_max is not None else float("inf")
            # Falls back to the full set if the band is thinner than expected
            # (e.g. very few 2200+ puzzles) rather than ever raising.
            candidates = [p for p in self.puzzles if lo <= p.rating <= hi] or self.puzzles
        available = [p for p in candidates if p.puzzle_id not in self._served]
        if not available:
            self._served.clear()
            available = candidates
        best = min(available, key=lambda p: abs(p.rating - difficulty))
        self._served.add(best.puzzle_id)
        return best

    def submit_move(self, puzzle: Puzzle, move_uci: str, time_to_move: float) -> BehaviorEvent:
        board = chess.Board(puzzle.fen)
        try:
            played_move = chess.Move.from_uci(move_uci)
            is_legal = played_move in board.legal_moves
        except (ValueError, TypeError):
            played_move = None
            is_legal = False

        best_move = chess.Move.from_uci(puzzle.solution_move)
        correct = is_legal and played_move == best_move
        if not is_legal:
            # The evaluator's _score() pushes the move onto the board, which asserts on
            # illegal moves — so never hand it one; charge a flat penalty instead.
            eval_loss = self.ILLEGAL_MOVE_PENALTY
        elif correct:
            eval_loss = 0.0
        else:
            eval_loss = self.evaluator.eval_loss(board, played_move, best_move)
        return BehaviorEvent(
            puzzle_id=puzzle.puzzle_id,
            correct=correct,
            time_to_move=time_to_move,
            eval_loss=eval_loss,
            puzzle_rating=puzzle.rating,
        )
