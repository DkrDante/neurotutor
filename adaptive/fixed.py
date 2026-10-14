"""A policy that never adapts anything. Used while COLLECTING labelled training
data (/collect): the experimental condition (puzzle difficulty band, time
pressure, ...) must be the only thing shaping the participant's experience, so
the tutor must not also be changing difficulty, injecting hints or adding
pacing delays off its own (still-unvalidated) state predictions."""
from __future__ import annotations
from adaptive.policy import Action, Policy


class FixedPolicy(Policy):
    def decide(self, state: str, confidence: float, correct: bool) -> Action:
        return Action(difficulty_delta=0.0, show_hint=False, pacing_delay=0.0)
