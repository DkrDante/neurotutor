from accounts.streak import MAX_STREAK_FREEZES, compute_streak_update

def test_first_ever_checkin_starts_streak_at_one():
    result = compute_streak_update(None, 0, 0, 0, "2026-01-01")
    assert result == {
        "current_streak": 1, "longest_streak": 1,
        "last_play_date": "2026-01-01", "changed_today": True,
        "freezes_available": 0, "freeze_used": False,
    }

def test_checking_in_again_same_day_is_a_no_op():
    result = compute_streak_update("2026-01-01", 5, 10, 1, "2026-01-01")
    assert result["current_streak"] == 5
    assert result["longest_streak"] == 10
    assert result["changed_today"] is False
    assert result["freezes_available"] == 1
    assert result["freeze_used"] is False

def test_consecutive_day_extends_streak():
    result = compute_streak_update("2026-01-01", 5, 5, 0, "2026-01-02")
    assert result["current_streak"] == 6
    assert result["longest_streak"] == 6
    assert result["changed_today"] is True
    assert result["freeze_used"] is False

def test_gap_of_two_days_uses_a_freeze_if_available():
    result = compute_streak_update("2026-01-01", 10, 10, 1, "2026-01-03")
    assert result["current_streak"] == 11
    assert result["longest_streak"] == 11
    assert result["freeze_used"] is True
    assert result["freezes_available"] == 0

def test_gap_of_two_days_resets_streak_when_no_freeze_available():
    result = compute_streak_update("2026-01-01", 10, 10, 0, "2026-01-03")
    assert result["current_streak"] == 1
    assert result["longest_streak"] == 10
    assert result["freeze_used"] is False

def test_gap_of_three_or_more_days_resets_streak_even_with_a_freeze_banked():
    result = compute_streak_update("2026-01-01", 10, 10, 2, "2026-01-05")
    assert result["current_streak"] == 1
    assert result["freeze_used"] is False
    assert result["freezes_available"] == 2  # untouched — a freeze only bridges one missed day

def test_longest_streak_updates_once_current_exceeds_it():
    result = compute_streak_update("2026-01-05", 9, 9, 0, "2026-01-06")
    assert result["current_streak"] == 10
    assert result["longest_streak"] == 10

def test_crossing_a_month_boundary_still_counts_as_consecutive():
    result = compute_streak_update("2026-01-31", 3, 3, 0, "2026-02-01")
    assert result["current_streak"] == 4
    assert result["changed_today"] is True

def test_reaching_a_multiple_of_seven_earns_a_freeze():
    result = compute_streak_update("2026-01-06", 6, 6, 0, "2026-01-07")
    assert result["current_streak"] == 7
    assert result["freezes_available"] == 1

def test_earned_freezes_are_capped_at_the_maximum():
    result = compute_streak_update("2026-01-06", 6, 6, MAX_STREAK_FREEZES, "2026-01-07")
    assert result["current_streak"] == 7
    assert result["freezes_available"] == MAX_STREAK_FREEZES
