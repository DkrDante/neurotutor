from adaptive.proactive import TREND_WINDOW, detect_struggle_trend

def test_no_trend_with_too_few_attempts():
    assert detect_struggle_trend([("Overloaded", 0.9)]) is False

def test_no_trend_when_mostly_focused():
    recent = [("Focused", 0.9), ("Focused", 0.8), ("Overloaded", 0.9)]
    assert detect_struggle_trend(recent) is False

def test_trend_detected_with_two_of_three_struggling():
    recent = [("Focused", 0.9), ("Overloaded", 0.9), ("Confused", 0.7)]
    assert detect_struggle_trend(recent) is True

def test_trend_ignores_low_confidence_struggle_reads():
    # Both "struggle" reads are below the confidence threshold — shouldn't trust them.
    recent = [("Overloaded", 0.2), ("Confused", 0.3), ("Focused", 0.9)]
    assert detect_struggle_trend(recent) is False

def test_trend_only_looks_at_the_most_recent_window():
    # Two struggle reads exist, but they're outside the trailing window.
    long_history = [("Overloaded", 0.9), ("Confused", 0.9)] + [("Focused", 0.9)] * TREND_WINDOW
    assert detect_struggle_trend(long_history) is False

def test_trend_detected_regardless_of_correctness():
    # This module only looks at (state, confidence) pairs — correctness isn't
    # even part of the signature, confirming the trend fires independent of it.
    recent = [("Fatigued", 0.6), ("Fatigued", 0.6), ("Fatigued", 0.6)]
    assert detect_struggle_trend(recent) is True
