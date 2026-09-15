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

def test_list_sessions_returns_newest_first_with_summary_stats():
    store = SessionStore(":memory:")
    first_session = store.create_session()
    store.log_attempt(first_session, "p1", True, 5.0, 0.0, 1000, "Focused", 0.9, 50.0, False, 0.0, 0.05)
    second_session = store.create_session()
    store.log_attempt(second_session, "p1", False, 8.0, 150.0, 1000, "Confused", 0.7, -100.0, True, 2.0, 0.08)
    store.log_attempt(second_session, "p2", True, 4.0, 0.0, 1100, "Focused", 0.85, 75.0, False, 0.0, 0.06)

    sessions = store.list_sessions()

    assert [s["session_id"] for s in sessions] == [second_session, first_session]
    assert sessions[0]["num_attempts"] == 2
    assert sessions[0]["accuracy_rate"] == 0.5
    assert sessions[1]["num_attempts"] == 1
    assert sessions[1]["accuracy_rate"] == 1.0
    store.close()

def test_list_sessions_excludes_sessions_with_no_attempts():
    store = SessionStore(":memory:")
    store.create_session()  # never plays a puzzle
    assert store.list_sessions() == []
    store.close()

def test_get_session_attempts_returns_full_time_series_in_order():
    store = SessionStore(":memory:")
    session_id = store.create_session()
    store.log_attempt(session_id, "p1", True, 5.0, 0.0, 1000, "Focused", 0.9, 50.0, False, 0.0, 0.05)
    store.log_attempt(session_id, "p2", False, 8.0, 150.0, 1100, "Confused", 0.7, -100.0, True, 2.0, 0.08)

    attempts = store.get_session_attempts(session_id)

    assert [a["puzzle_id"] for a in attempts] == ["p1", "p2"]
    assert attempts[0]["correct"] is True
    assert attempts[1]["correct"] is False
    assert attempts[1]["show_hint"] is True
    assert attempts[1]["puzzle_rating"] == 1100
    store.close()

def test_get_session_attempts_for_unknown_session_is_empty():
    store = SessionStore(":memory:")
    assert store.get_session_attempts("does-not-exist") == []
    store.close()

def test_learner_profile_unknown_learner_returns_none():
    store = SessionStore(":memory:")
    assert store.get_learner_profile("does-not-exist") is None
    store.close()

def test_save_and_get_learner_profile_round_trips():
    store = SessionStore(":memory:")
    store.save_learner_profile("learner-1", difficulty=1250.0, policy_state={"q_table": [{"state": ["Focused", "high"], "values": [1.0, 2.0]}]})

    profile = store.get_learner_profile("learner-1")

    assert profile["difficulty"] == 1250.0
    assert profile["policy_state"]["q_table"][0]["state"] == ["Focused", "high"]
    store.close()

def test_save_learner_profile_without_policy_state():
    store = SessionStore(":memory:")
    store.save_learner_profile("learner-2", difficulty=900.0)
    profile = store.get_learner_profile("learner-2")
    assert profile == {"difficulty": 900.0, "policy_state": None, "served_puzzle_ids": set()}
    store.close()

def test_save_learner_profile_upserts_existing_learner():
    store = SessionStore(":memory:")
    store.save_learner_profile("learner-3", difficulty=1000.0)
    store.save_learner_profile("learner-3", difficulty=1400.0)
    profile = store.get_learner_profile("learner-3")
    assert profile["difficulty"] == 1400.0
    store.close()

def test_served_puzzle_ids_round_trip():
    store = SessionStore(":memory:")
    store.save_learner_profile("learner-4", difficulty=1000.0, served_puzzle_ids={"p1", "p2", "p3"})
    profile = store.get_learner_profile("learner-4")
    assert profile["served_puzzle_ids"] == {"p1", "p2", "p3"}
    store.close()

def test_served_puzzle_ids_updates_on_upsert():
    store = SessionStore(":memory:")
    store.save_learner_profile("learner-5", difficulty=1000.0, served_puzzle_ids={"p1"})
    store.save_learner_profile("learner-5", difficulty=1000.0, served_puzzle_ids={"p1", "p2"})
    profile = store.get_learner_profile("learner-5")
    assert profile["served_puzzle_ids"] == {"p1", "p2"}
    store.close()

def test_served_puzzle_ids_untouched_when_omitted():
    # Game mode saves difficulty/policy_state without ever passing
    # served_puzzle_ids (it has no puzzle pool) — that must never wipe out
    # puzzle-mode's tracking for the same learner.
    store = SessionStore(":memory:")
    store.save_learner_profile("learner-6", difficulty=1000.0, served_puzzle_ids={"p1", "p2"})
    store.save_learner_profile("learner-6", difficulty=1100.0)
    profile = store.get_learner_profile("learner-6")
    assert profile["difficulty"] == 1100.0
    assert profile["served_puzzle_ids"] == {"p1", "p2"}
    store.close()

def test_aggregate_stats_of_empty_store_reports_zero_without_raising():
    store = SessionStore(":memory:")
    stats = store.get_aggregate_stats("puzzle")
    assert stats["mode"] == "puzzle"
    assert stats["total_sessions"] == 0
    assert stats["total_attempts"] == 0
    assert stats["overall_accuracy"] == 0.0
    assert stats["state_distribution"] == []
    assert stats["engagement_trend_avg"] is None
    store.close()

def test_aggregate_stats_across_multiple_sessions():
    store = SessionStore(":memory:")
    s1 = store.create_session()
    store.log_attempt(s1, "p1", True, 5.0, 0.0, 1000, "Focused", 0.9, 50.0, False, 0.0, 0.05)
    store.log_attempt(s1, "p2", False, 8.0, 150.0, 1000, "Confused", 0.7, -100.0, True, 2.0, 0.08)
    s2 = store.create_session()
    store.log_attempt(s2, "p3", True, 4.0, 0.0, 1100, "Focused", 0.85, 75.0, False, 0.0, 0.06)

    stats = store.get_aggregate_stats("puzzle")

    assert stats["total_sessions"] == 2
    assert stats["total_attempts"] == 3
    assert abs(stats["overall_accuracy"] - (2 / 3)) < 1e-9
    states = {row["state"]: row for row in stats["state_distribution"]}
    assert states["Focused"]["count"] == 2
    assert states["Confused"]["count"] == 1
    store.close()

def test_aggregate_stats_engagement_trend_detects_improvement_within_a_session():
    store = SessionStore(":memory:")
    session_id = store.create_session()
    # First half all wrong, second half all correct — a clear improving trend.
    for _ in range(2):
        store.log_attempt(session_id, "p1", False, 5.0, 100.0, 1000, "Confused", 0.9, -50.0, True, 1.0, 0.05)
    for _ in range(2):
        store.log_attempt(session_id, "p2", True, 5.0, 0.0, 1000, "Focused", 0.9, 50.0, False, 0.0, 0.05)

    stats = store.get_aggregate_stats("puzzle")

    assert stats["engagement_trend_sessions_counted"] == 1
    assert stats["engagement_trend_avg"] == 1.0  # 100% second half - 0% first half
    store.close()

def test_aggregate_stats_scoped_by_mode_never_mixes_puzzle_and_game():
    store = SessionStore(":memory:")
    puzzle_session = store.create_session()
    store.log_attempt(
        puzzle_session, "p1", True, 5.0, 0.0, 1000, "Focused", 0.9, 50.0, False, 0.0, 0.05, mode="puzzle",
    )
    game_session = store.create_session()
    # Full-game mode repurposes puzzle_rating to mean the live difficulty
    # number — a wildly different scale from a real puzzle rating, which is
    # exactly why these two modes must never be aggregated together.
    store.log_attempt(
        game_session, "game-ply-0", False, 12.0, 0.0, 400, "Confused", 0.8, -50.0, True, 1.0, 0.03, mode="game",
    )

    puzzle_stats = store.get_aggregate_stats("puzzle")
    game_stats = store.get_aggregate_stats("game")

    assert puzzle_stats["total_attempts"] == 1
    assert puzzle_stats["overall_accuracy"] == 1.0
    assert game_stats["total_attempts"] == 1
    assert game_stats["overall_accuracy"] == 0.0
    store.close()

def test_log_attempt_defaults_to_puzzle_mode_when_unspecified():
    store = SessionStore(":memory:")
    session_id = store.create_session()
    store.log_attempt(session_id, "p1", True, 5.0, 0.0, 1000, "Focused", 0.9, 50.0, False, 0.0, 0.05)
    attempts = store.get_session_attempts(session_id)
    assert attempts[0]["mode"] == "puzzle"
    store.close()

def test_list_sessions_can_filter_by_mode():
    store = SessionStore(":memory:")
    puzzle_session = store.create_session()
    store.log_attempt(puzzle_session, "p1", True, 5.0, 0.0, 1000, "Focused", 0.9, 50.0, False, 0.0, 0.05, mode="puzzle")
    game_session = store.create_session()
    store.log_attempt(game_session, "game-ply-0", True, 5.0, 0.0, 500, "Focused", 0.9, 50.0, False, 0.0, 0.05, mode="game")

    puzzle_only = store.list_sessions(mode="puzzle")
    assert [s["session_id"] for s in puzzle_only] == [puzzle_session]
    assert puzzle_only[0]["mode"] == "puzzle"

    game_only = store.list_sessions(mode="game")
    assert [s["session_id"] for s in game_only] == [game_session]
    store.close()
