import pytest
from adaptive.rule_based import (
    CONFIDENCE_THRESHOLD, CORRECT_DIFFICULTY_DELTA, RuleBasedPolicy, WRONG_DIFFICULTY_DELTA,
)

def test_overloaded_lowers_difficulty_and_shows_hint():
    policy = RuleBasedPolicy()
    action = policy.decide("Overloaded", confidence=0.9, correct=True)
    assert action.difficulty_delta < 0
    assert action.show_hint is True

def test_focused_raises_difficulty():
    policy = RuleBasedPolicy()
    action = policy.decide("Focused", confidence=0.9, correct=True)
    assert action.difficulty_delta > 0

def test_fatigued_increases_pacing_delay():
    policy = RuleBasedPolicy()
    action = policy.decide("Fatigued", confidence=0.9, correct=True)
    assert action.pacing_delay > 0

def test_low_confidence_falls_back_to_correctness_only():
    policy = RuleBasedPolicy()
    below_threshold = CONFIDENCE_THRESHOLD - 0.01

    correct_action = policy.decide("Overloaded", confidence=below_threshold, correct=True)
    assert correct_action.difficulty_delta == CORRECT_DIFFICULTY_DELTA
    assert correct_action.show_hint is False
    assert correct_action.pacing_delay == 0.0

    wrong_action = policy.decide("Overloaded", confidence=below_threshold, correct=False)
    assert wrong_action.difficulty_delta == WRONG_DIFFICULTY_DELTA

def test_unknown_state_raises():
    policy = RuleBasedPolicy()
    with pytest.raises(ValueError):
        policy.decide("NotAState", confidence=0.9, correct=True)

def test_correct_move_raises_difficulty_more_than_a_wrong_move_in_the_same_state():
    # The core adaptive-difficulty behavior: getting it right should always push
    # difficulty higher than getting it wrong, holding cognitive state fixed.
    policy = RuleBasedPolicy()
    correct_action = policy.decide("Focused", confidence=0.9, correct=True)
    wrong_action = policy.decide("Focused", confidence=0.9, correct=False)
    assert correct_action.difficulty_delta > wrong_action.difficulty_delta

def test_wrong_move_can_lower_difficulty_even_in_a_positive_state():
    # A wrong move's own -100 penalty can outweigh a merely-Engaged (+75) state,
    # so even a "good" cognitive state doesn't paper over a missed puzzle.
    policy = RuleBasedPolicy()
    action = policy.decide("Engaged", confidence=0.9, correct=False)
    assert action.difficulty_delta < 0

def test_correct_move_still_lowers_difficulty_when_overloaded():
    # Correctness alone isn't enough to ramp difficulty up on a struggling learner:
    # Overloaded's -150 outweighs a correct answer's +100.
    policy = RuleBasedPolicy()
    action = policy.decide("Overloaded", confidence=0.9, correct=True)
    assert action.difficulty_delta == CORRECT_DIFFICULTY_DELTA + (-150.0)
    assert action.difficulty_delta < 0
