from coaching.trace import build_aggregate_calculations, build_attempt_trace, build_calculation_trace

def _attempt(**overrides):
    base = {
        "puzzle_id": "p1", "correct": True, "time_to_move": 5.0, "eval_loss": 0.0,
        "puzzle_rating": 1000, "predicted_state": "Focused", "confidence": 0.9,
        "difficulty_delta": 100.0, "show_hint": False, "pacing_delay": 0.0,
        "sense_to_adapt_latency": 0.02, "created_at": 0.0,
    }
    base.update(overrides)
    return base

def test_empty_attempts_returns_empty_trace():
    result = build_calculation_trace([])
    assert result == {"num_attempts": 0, "aggregate": [], "attempts": []}

def test_attempt_trace_tracks_running_accuracy():
    attempts = [_attempt(correct=True), _attempt(correct=False), _attempt(correct=True)]
    trace = build_attempt_trace(attempts)
    assert [t["running_accuracy"] for t in trace] == [1.0, 0.5, 2 / 3]

def test_attempt_trace_walks_difficulty_from_1000_with_floor():
    attempts = [_attempt(difficulty_delta=100.0), _attempt(difficulty_delta=-2000.0), _attempt(difficulty_delta=50.0)]
    trace = build_attempt_trace(attempts)
    assert trace[0]["difficulty_before"] == 1000.0
    assert trace[0]["difficulty_after"] == 1100.0
    assert trace[1]["difficulty_before"] == 1100.0
    assert trace[1]["difficulty_after"] == 400.0  # floored
    assert trace[2]["difficulty_before"] == 400.0
    assert trace[2]["difficulty_after"] == 450.0

def test_rule_based_reconstruction_matches_when_confidence_is_low():
    # Below the 0.4 threshold: only the correctness delta applies, regardless of state.
    a = _attempt(correct=True, confidence=0.1, predicted_state="Overloaded", difficulty_delta=100.0)
    trace = build_attempt_trace([a])
    recon = trace[0]["rule_based_reconstruction"]
    assert recon["confidence_trusted"] is False
    assert recon["state_delta_applied"] == 0.0
    assert recon["predicted_delta"] == 100.0
    assert recon["matches_actual"] is True

def test_rule_based_reconstruction_adds_state_delta_when_confidence_trusted():
    # Overloaded + correct: 100 (correctness) + (-150) (state) = -50.
    a = _attempt(correct=True, confidence=0.9, predicted_state="Overloaded", difficulty_delta=-50.0)
    trace = build_attempt_trace([a])
    recon = trace[0]["rule_based_reconstruction"]
    assert recon["confidence_trusted"] is True
    assert recon["state_delta_applied"] == -150.0
    assert recon["predicted_delta"] == -50.0
    assert recon["matches_actual"] is True

def test_rule_based_reconstruction_flags_divergence_from_rl_policy():
    # Recorded delta doesn't match what the rule-based formula would produce —
    # signals this attempt was actually decided by the RL policy instead.
    a = _attempt(correct=True, confidence=0.9, predicted_state="Overloaded", difficulty_delta=999.0)
    trace = build_attempt_trace([a])
    recon = trace[0]["rule_based_reconstruction"]
    assert recon["predicted_delta"] == -50.0
    assert recon["matches_actual"] is False

def test_aggregate_calculations_includes_accuracy_and_difficulty():
    attempts = [_attempt(correct=True), _attempt(correct=False)]
    calcs = build_aggregate_calculations(attempts)
    titles = {c["title"] for c in calcs}
    assert "Overall accuracy" in titles
    assert "Difficulty trajectory" in titles
    accuracy_calc = next(c for c in calcs if c["title"] == "Overall accuracy")
    assert accuracy_calc["substitution"] == "1 / 2"
    assert accuracy_calc["result"] == "50.0%"

def test_aggregate_calculations_empty_for_no_attempts():
    assert build_aggregate_calculations([]) == []

def test_aggregate_calculations_omits_weakest_entries_when_none():
    attempts = [_attempt(correct=True) for _ in range(3)]
    calcs = build_aggregate_calculations(attempts)
    titles = {c["title"] for c in calcs}
    assert "Weakest puzzle-rating band" not in titles
    assert "Weakest cognitive state" not in titles

def test_build_calculation_trace_shape():
    attempts = [_attempt()]
    result = build_calculation_trace(attempts)
    assert result["num_attempts"] == 1
    assert len(result["attempts"]) == 1
    assert isinstance(result["aggregate"], list)
