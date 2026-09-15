from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class Action:
    difficulty_delta: float
    show_hint: bool
    pacing_delay: float

class Policy(ABC):
    @abstractmethod
    def decide(self, state: str, confidence: float, correct: bool) -> Action:
        raise NotImplementedError
