"""Proactive struggle detection — a closed adaptive loop that acts on a
TREND across recent attempts, not just the outcome of the current one.

adaptive.rule_based / adaptive.rl_policy are both reactive: they only adjust
difficulty/hints after THIS attempt's correctness is already known. A learner
who is still answering correctly but has been read as Overloaded, Confused,
or Fatigued for several attempts in a row is trending toward a mistake (or
toward disengaging) before that shows up as a concrete error — this module
is what lets the tutor intervene at that point instead of waiting for it.
"""
from __future__ import annotations

STRUGGLE_STATES = {"Overloaded", "Confused", "Fatigued"}
TREND_WINDOW = 3
TREND_STATE_THRESHOLD = 2  # at least this many of the window read as a struggle state
TREND_CONFIDENCE_THRESHOLD = 0.4  # only trust a state read the model itself is confident about

# Applied ON TOP OF whatever the active policy (rule-based or RL) already
# decided for this attempt, only when a trend is newly detected.
PROACTIVE_DIFFICULTY_NUDGE = -75.0


def detect_struggle_trend(recent_states: list[tuple[str, float]]) -> bool:
    """recent_states: the last few (predicted_state, confidence) pairs, oldest
    first. True if enough of the recent window reads as a struggle state with
    real confidence — regardless of whether those attempts were correct."""
    if len(recent_states) < TREND_WINDOW:
        return False
    window = recent_states[-TREND_WINDOW:]
    struggling = sum(
        1 for state, confidence in window
        if state in STRUGGLE_STATES and confidence >= TREND_CONFIDENCE_THRESHOLD
    )
    return struggling >= TREND_STATE_THRESHOLD
