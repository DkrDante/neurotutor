from storage.db import SessionStore

def test_create_session_and_log_attempt_round_trip():
    store = SessionStore(":memory:")
    session_id = store.create_session()
    attempt_id = store.log_attempt(
        session_id=session_id, puzzle_id="p1", correct=True, time_to_move=5.0,
        eval_loss=0.0, puzzle_rating=1000, predicted_state="Focused", confidence=0.9,
        difficulty_delta=50.0, show_hint=False, pacing_delay=0.0, sense_to_adapt_latency=0.05,
    )
    assert attempt_id
    summary = store.get_session_summary(session_id)
    assert summary["num_attempts"] == 1
    assert summary["accuracy_rate"] == 1.0
    store.close()

def test_session_summary_aggregates_multiple_attempts():
    store = SessionStore(":memory:")
    session_id = store.create_session()
    store.log_attempt(session_id, "p1", True, 5.0, 0.0, 1000, "Focused", 0.9, 50.0, False, 0.0, 0.05)
    store.log_attempt(session_id, "p2", False, 8.0, 150.0, 1000, "Confused", 0.7, -100.0, True, 2.0, 0.08)
    summary = store.get_session_summary(session_id)
    assert summary["num_attempts"] == 2
    assert summary["accuracy_rate"] == 0.5
    assert summary["state_trend"] == ["Focused", "Confused"]
    store.close()

def test_summary_of_unknown_session_returns_empty():
    store = SessionStore(":memory:")
    summary = store.get_session_summary("does-not-exist")
    assert summary["num_attempts"] == 0
    store.close()
