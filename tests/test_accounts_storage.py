import pytest
from accounts.storage import AccountStore, InvalidCredentialsError, UsernameTakenError

def _store():
    return AccountStore(":memory:")

def test_create_user_and_verify_login():
    store = _store()
    user_id = store.create_user("alice", "hunter22")
    assert store.verify_login("alice", "hunter22") == user_id

def test_verify_login_rejects_wrong_password():
    store = _store()
    store.create_user("alice", "hunter22")
    with pytest.raises(InvalidCredentialsError):
        store.verify_login("alice", "wrongpass")

def test_verify_login_rejects_unknown_username():
    store = _store()
    with pytest.raises(InvalidCredentialsError):
        store.verify_login("ghost", "whatever")

def test_create_user_rejects_duplicate_username():
    store = _store()
    store.create_user("alice", "hunter22")
    with pytest.raises(UsernameTakenError):
        store.create_user("alice", "differentpass")

def test_create_user_seeds_progress_from_initial_progress():
    store = _store()
    user_id = store.create_user("alice", "hunter22", initial_progress={
        "total_xp": 340, "solved_count": 12, "best_streak": 5,
        "unlocked_achievements": ["first-blood", "streak-2"],
    })
    profile = store.get_profile(user_id)
    assert profile["total_xp"] == 340
    assert profile["solved_count"] == 12
    assert profile["best_streak"] == 5
    assert profile["unlocked_achievements"] == ["first-blood", "streak-2"]

def test_create_user_defaults_progress_to_zero_when_not_given():
    store = _store()
    user_id = store.create_user("alice", "hunter22")
    profile = store.get_profile(user_id)
    assert profile["total_xp"] == 0
    assert profile["unlocked_achievements"] == []
    assert profile["daily_streak"] == {"current": 0, "longest": 0, "last_play_date": None, "freezes_available": 0}

def test_account_session_round_trip():
    store = _store()
    user_id = store.create_user("alice", "hunter22")
    token = store.create_account_session(user_id)
    assert store.get_user_id_for_session(token) == user_id
    store.delete_account_session(token)
    assert store.get_user_id_for_session(token) is None

def test_get_user_id_for_session_returns_none_for_unknown_token():
    store = _store()
    assert store.get_user_id_for_session("not-a-real-token") is None

def test_save_progress_updates_stored_counters():
    store = _store()
    user_id = store.create_user("alice", "hunter22")
    store.save_progress(user_id, {
        "total_xp": 500, "solved_count": 20, "current_streak": 3, "best_streak": 8,
        "no_hint_streak": 2, "no_hint_total_count": 15, "max_rating_solved": 1800,
        "games_played": 4, "games_won": 2, "puzzles_served": 25, "unlocked_achievements": ["century"],
    })
    profile = store.get_profile(user_id)
    assert profile["total_xp"] == 500
    assert profile["games_won"] == 2
    assert profile["puzzles_served"] == 25
    assert profile["unlocked_achievements"] == ["century"]

def test_puzzles_served_defaults_to_zero_for_a_fresh_account():
    store = _store()
    user_id = store.create_user("alice", "hunter22")
    assert store.get_profile(user_id)["puzzles_served"] == 0

def test_record_daily_checkin_persists_across_calls():
    store = _store()
    user_id = store.create_user("alice", "hunter22")
    first = store.record_daily_checkin(user_id, today="2026-01-01")
    assert first["current_streak"] == 1
    second = store.record_daily_checkin(user_id, today="2026-01-02")
    assert second["current_streak"] == 2
    profile = store.get_profile(user_id)
    assert profile["daily_streak"] == {
        "current": 2, "longest": 2, "last_play_date": "2026-01-02", "freezes_available": 0,
    }

def test_record_daily_checkin_rejects_unknown_user():
    store = _store()
    with pytest.raises(ValueError):
        store.record_daily_checkin("no-such-user", today="2026-01-01")

def test_record_daily_checkin_uses_a_banked_freeze_to_bridge_a_missed_day():
    store = _store()
    user_id = store.create_user("alice", "hunter22")
    for i in range(1, 8):
        store.record_daily_checkin(user_id, today=f"2026-01-{i:02d}")
    profile = store.get_profile(user_id)
    assert profile["daily_streak"]["current"] == 7
    assert profile["daily_streak"]["freezes_available"] == 1  # earned at day 7

    # Skip 2026-01-08 entirely, come back on the 9th — a 2-day gap.
    result = store.record_daily_checkin(user_id, today="2026-01-09")
    assert result["freeze_used"] is True
    assert result["current_streak"] == 8
    assert result["freezes_available"] == 0

def test_get_profile_returns_none_for_unknown_user():
    store = _store()
    assert store.get_profile("no-such-user") is None

def test_get_leaderboard_orders_by_total_xp_descending():
    store = _store()
    store.create_user("alice", "hunter22", initial_progress={"total_xp": 100})
    store.create_user("bob", "hunter22", initial_progress={"total_xp": 500})
    store.create_user("carol", "hunter22", initial_progress={"total_xp": 300})
    entries = store.get_leaderboard()
    assert [e["username"] for e in entries] == ["bob", "carol", "alice"]
    assert entries[0]["total_xp"] == 500

def test_get_leaderboard_respects_limit():
    store = _store()
    for i in range(5):
        store.create_user(f"user{i}", "hunter22", initial_progress={"total_xp": i})
    assert len(store.get_leaderboard(limit=2)) == 2

def test_get_rank_reflects_total_xp_position():
    store = _store()
    store.create_user("alice", "hunter22", initial_progress={"total_xp": 100})
    bob_id = store.create_user("bob", "hunter22", initial_progress={"total_xp": 500})
    store.create_user("carol", "hunter22", initial_progress={"total_xp": 300})
    assert store.get_rank(bob_id) == 1

def test_get_rank_returns_none_for_unknown_user():
    store = _store()
    assert store.get_rank("no-such-user") is None
