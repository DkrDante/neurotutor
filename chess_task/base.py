from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class Puzzle:
    puzzle_id: str
    fen: str
    solution_move: str  # UCI, e.g. "e2e4"
    rating: int

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
