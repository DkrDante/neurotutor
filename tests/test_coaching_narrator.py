import httpx
from coaching.narrator import ai_narrative, build_coaching_report, rule_based_tips

def test_rule_based_tips_for_no_attempts():
    tips = rule_based_tips({"num_attempts": 0})
    assert len(tips) == 1
    assert "play a few puzzles" in tips[0].lower()

def test_rule_based_tips_names_weakest_rating_band():
    stats = {
        "num_attempts": 10, "accuracy_rate": 0.7,
        "weakest_rating_band": {"key": "1800-2199", "accuracy": 0.2, "count": 5},
        "weakest_state": None, "avg_time_correct": None, "avg_time_wrong": None,
        "avg_eval_loss_on_misses": None, "hint_rate": 0.0, "difficulty_change": 0,
    }
    tips = rule_based_tips(stats)
    assert any("1800-2199" in t and "20%" in t for t in tips)

def test_rule_based_tips_names_weakest_state_with_advice():
    stats = {
        "num_attempts": 10, "accuracy_rate": 0.8,
        "weakest_rating_band": None,
        "weakest_state": {"key": "Overloaded", "accuracy": 0.3, "count": 4},
        "avg_time_correct": None, "avg_time_wrong": None,
        "avg_eval_loss_on_misses": None, "hint_rate": 0.0, "difficulty_change": 0,
    }
    tips = rule_based_tips(stats)
    assert any("Overloaded" in t and "re-scan the whole board" in t for t in tips)

def test_rule_based_tips_flags_rushing():
    stats = {
        "num_attempts": 10, "accuracy_rate": 0.8, "weakest_rating_band": None, "weakest_state": None,
        "avg_time_correct": 10.0, "avg_time_wrong": 3.0,  # wrong is much faster than correct
        "avg_eval_loss_on_misses": None, "hint_rate": 0.0, "difficulty_change": 0,
    }
    tips = rule_based_tips(stats)
    assert any("rushing" in t.lower() for t in tips)

def test_rule_based_tips_flags_overthinking():
    stats = {
        "num_attempts": 10, "accuracy_rate": 0.8, "weakest_rating_band": None, "weakest_state": None,
        "avg_time_correct": 3.0, "avg_time_wrong": 12.0,  # wrong takes much longer than correct
        "avg_eval_loss_on_misses": None, "hint_rate": 0.0, "difficulty_change": 0,
    }
    tips = rule_based_tips(stats)
    assert any("calculation errors" in t.lower() for t in tips)

def test_rule_based_tips_flags_severe_blunders():
    stats = {
        "num_attempts": 10, "accuracy_rate": 0.8, "weakest_rating_band": None, "weakest_state": None,
        "avg_time_correct": None, "avg_time_wrong": None,
        "avg_eval_loss_on_misses": 300.0, "hint_rate": 0.0, "difficulty_change": 0,
    }
    tips = rule_based_tips(stats)
    assert any("blunder" in t.lower() for t in tips)

def test_rule_based_tips_flags_high_hint_rate():
    stats = {
        "num_attempts": 10, "accuracy_rate": 0.8, "weakest_rating_band": None, "weakest_state": None,
        "avg_time_correct": None, "avg_time_wrong": None,
        "avg_eval_loss_on_misses": None, "hint_rate": 0.6, "difficulty_change": 0,
    }
    tips = rule_based_tips(stats)
    assert any("hint" in t.lower() for t in tips)

def test_rule_based_tips_falls_back_to_encouragement_when_nothing_stands_out():
    stats = {
        "num_attempts": 10, "accuracy_rate": 0.8, "weakest_rating_band": None, "weakest_state": None,
        "avg_time_correct": 5.0, "avg_time_wrong": 5.0,
        "avg_eval_loss_on_misses": 20.0, "hint_rate": 0.1, "difficulty_change": 10,
    }
    tips = rule_based_tips(stats)
    assert len(tips) == 1
    assert "solid" in tips[0].lower()

def test_ai_narrative_returns_none_for_empty_session():
    assert ai_narrative({"num_attempts": 0}) is None

def test_ai_narrative_returns_generated_text_on_success(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"response": "  Great session! Keep drilling those endgames.  "}

    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: FakeResponse())
    result = ai_narrative({"num_attempts": 5, "accuracy_rate": 0.5})
    assert result == "Great session! Keep drilling those endgames."

def test_ai_narrative_never_raises_when_ollama_errors(monkeypatch):
    def raise_connection_error(*args, **kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "post", raise_connection_error)
    result = ai_narrative({"num_attempts": 5, "accuracy_rate": 0.5})
    assert result is None

def test_ai_narrative_returns_none_on_empty_generated_text(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"response": "   "}

    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: FakeResponse())
    assert ai_narrative({"num_attempts": 5, "accuracy_rate": 0.5}) is None

def test_build_coaching_report_shape():
    report = build_coaching_report({"num_attempts": 0})
    assert set(report) == {"statistics", "tips", "narrative"}
    assert report["narrative"] is None
    assert len(report["tips"]) >= 1
