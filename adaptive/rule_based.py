from __future__ import annotations
from adaptive.policy import Policy, Action
from common.states import STATES

CONFIDENCE_THRESHOLD = 0.4

STATE_RULES = {
    "Focused":    Action(difficulty_delta=100.0, show_hint=False, pacing_delay=0.0),
    "Engaged":    Action(difficulty_delta=75.0, show_hint=False, pacing_delay=0.0),
    "Overloaded": Action(difficulty_delta=-150.0, show_hint=True, pacing_delay=3.0),
    "Confused":   Action(difficulty_delta=-100.0, show_hint=True, pacing_delay=2.0),
    "Fatigued":   Action(difficulty_delta=-50.0, show_hint=False, pacing_delay=5.0),
}
assert set(STATE_RULES) == set(STATES)
NO_OP_ACTION = Action(difficulty_delta=0.0, show_hint=False, pacing_delay=0.0)

class RuleBasedPolicy(Policy):
    def decide(self, state: str, confidence: float) -> Action:
        if state not in STATES:
            raise ValueError(f"Unknown state: {state}")
        if confidence < CONFIDENCE_THRESHOLD:
            return NO_OP_ACTION
        return STATE_RULES[state]
