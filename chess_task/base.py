from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

@dataclass
class Puzzle:
    puzzle_id: str
    fen: str
    solution_move: str  # UCI, e.g. "e2e4" — the solver's FIRST move
    rating: int
    # Full remaining move list after the puzzle's starting position: solver move,
    # opponent reply, solver move, ... (opponent replies are pre-scripted, not engine
    # play). Defaults to empty for single-ply puzzles built without this field (e.g.
    # constructed directly in tests) — callers should treat an empty list the same as
    # [solution_move].
    solution_moves: list[str] = field(default_factory=list)

@dataclass
class BehaviorEvent:
    puzzle_id: str
    correct: bool
    time_to_move: float
    eval_loss: float
    puzzle_rating: int

class TaskEngine(ABC):
    @abstractmethod
    def get_puzzle(self, difficulty: float) -> Puzzle:
        raise NotImplementedError

    @abstractmethod
    def submit_move(self, puzzle: Puzzle, move_uci: str, time_to_move: float) -> BehaviorEvent:
        raise NotImplementedError
