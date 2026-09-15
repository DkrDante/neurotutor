from __future__ import annotations
from adaptive.policy import Policy, Action
from common.states import STATES

CONFIDENCE_THRESHOLD = 0.4

# Whether THIS attempt was right or wrong is ground truth from the game itself —
# it always moves difficulty, independent of the EEG signal's confidence.
CORRECT_DIFFICULTY_DELTA = 100.0
WRONG_DIFFICULTY_DELTA = -100.0

# The predicted cognitive state's own push on difficulty, added on top of the
# correctness delta above (only once confidence clears CONFIDENCE_THRESHOLD —
# below that the state prediction isn't trusted enough to act on, and hints/
# pacing stay off, but correctness still adapts difficulty on its own).
STATE_RULES = {
    "Focused":    Action(difficulty_delta=100.0, show_hint=False, pacing_delay=0.0),
    "Engaged":    Action(difficulty_delta=75.0, show_hint=False, pacing_delay=0.0),
    "Overloaded": Action(difficulty_delta=-150.0, show_hint=True, pacing_delay=3.0),
    "Confused":   Action(difficulty_delta=-100.0, show_hint=True, pacing_delay=2.0),
    "Fatigued":   Action(difficulty_delta=-50.0, show_hint=False, pacing_delay=5.0),
}
assert set(STATE_RULES) == set(STATES)

class RuleBasedPolicy(Policy):
    def decide(self, state: str, confidence: float, correct: bool) -> Action:
        if state not in STATES:
            raise ValueError(f"Unknown state: {state}")

        correctness_delta = CORRECT_DIFFICULTY_DELTA if correct else WRONG_DIFFICULTY_DELTA
        if confidence < CONFIDENCE_THRESHOLD:
            return Action(difficulty_delta=correctness_delta, show_hint=False, pacing_delay=0.0)

        state_action = STATE_RULES[state]
        return Action(
            difficulty_delta=correctness_delta + state_action.difficulty_delta,
            show_hint=state_action.show_hint,
            pacing_delay=state_action.pacing_delay,
        )
