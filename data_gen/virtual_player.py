from __future__ import annotations
import numpy as np
from chess_task.base import BehaviorEvent
from common.states import STATES

STATE_BEHAVIOR_PARAMS = {
    "Focused":    {"correct_prob": 0.85, "time_mean": 6.0, "time_std": 1.5, "eval_loss_mean": 80.0},
    "Overloaded": {"correct_prob": 0.35, "time_mean": 14.0, "time_std": 4.0, "eval_loss_mean": 300.0},
    "Confused":   {"correct_prob": 0.40, "time_mean": 12.0, "time_std": 3.5, "eval_loss_mean": 260.0},
    "Fatigued":   {"correct_prob": 0.55, "time_mean": 16.0, "time_std": 5.0, "eval_loss_mean": 180.0},
    "Engaged":    {"correct_prob": 0.75, "time_mean": 7.5, "time_std": 2.0, "eval_loss_mean": 100.0},
}
assert set(STATE_BEHAVIOR_PARAMS) == set(STATES)

class VirtualPlayer:
    def __init__(self, target_state: str, seed: int | None = None):
        if target_state not in STATE_BEHAVIOR_PARAMS:
            raise ValueError(f"Unknown state: {target_state}")
        self.target_state = target_state
        self._rng = np.random.default_rng(seed)

    def attempt(self, puzzle_id: str, puzzle_rating: int) -> BehaviorEvent:
        params = STATE_BEHAVIOR_PARAMS[self.target_state]
        correct = bool(self._rng.random() < params["correct_prob"])
        time_to_move = float(max(0.5, self._rng.normal(params["time_mean"], params["time_std"])))
        eval_loss = 0.0
        if not correct:
            eval_loss = float(max(0.0, self._rng.normal(params["eval_loss_mean"], params["eval_loss_mean"] * 0.3)))
        return BehaviorEvent(
            puzzle_id=puzzle_id, correct=correct, time_to_move=time_to_move,
            eval_loss=eval_loss, puzzle_rating=puzzle_rating,
        )
