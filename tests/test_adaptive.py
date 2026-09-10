import pytest
from adaptive.rule_based import RuleBasedPolicy, CONFIDENCE_THRESHOLD

def test_overloaded_lowers_difficulty_and_shows_hint():
    policy = RuleBasedPolicy()
    action = policy.decide("Overloaded", confidence=0.9)
    assert action.difficulty_delta < 0
    assert action.show_hint is True

def test_focused_raises_difficulty():
    policy = RuleBasedPolicy()
    action = policy.decide("Focused", confidence=0.9)
    assert action.difficulty_delta > 0

def test_fatigued_increases_pacing_delay():
    policy = RuleBasedPolicy()
    action = policy.decide("Fatigued", confidence=0.9)
    assert action.pacing_delay > 0

def test_low_confidence_returns_no_op():
    policy = RuleBasedPolicy()
    action = policy.decide("Overloaded", confidence=CONFIDENCE_THRESHOLD - 0.01)
    assert action.difficulty_delta == 0.0
    assert action.show_hint is False

def test_unknown_state_raises():
    policy = RuleBasedPolicy()
    with pytest.raises(ValueError):
        policy.decide("NotAState", confidence=0.9)
