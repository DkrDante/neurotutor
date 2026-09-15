from coaching.statistics import MIN_SAMPLES, build_statistics

def _attempt(**overrides):
    base = {
        "puzzle_id": "p1", "correct": True, "time_to_move": 5.0, "eval_loss": 0.0,
        "puzzle_rating": 1000, "predicted_state": "Focused", "confidence": 0.9,
        "difficulty_delta": 100.0, "show_hint": False, "pacing_delay": 0.0,
        "sense_to_adapt_latency": 0.02, "created_at": 0.0,
    }
    base.update(overrides)
    return base

def test_empty_attempts_returns_zero_count_only():
    stats = build_statistics([])
    assert stats == {"num_attempts": 0}

def test_basic_accuracy_rate():
    attempts = [_attempt(correct=True), _attempt(correct=True), _attempt(correct=False), _attempt(correct=False)]
    stats = build_statistics(attempts)
    assert stats["num_attempts"] == 4
    assert stats["accuracy_rate"] == 0.5

def test_rating_band_stats_are_skipped_for_game_mode_attempts():
    # Full-game mode repurposes puzzle_rating to mean the live adaptive-
    # difficulty number (chess_task/full_game.py), so bucketing it into
    # "1800-2199" etc. would report a nonsense weakest rating band.
    attempts = (
        [_attempt(puzzle_rating=900, correct=True, mode="game") for _ in range(3)]
        + [_attempt(puzzle_rating=2000, correct=False, mode="game") for _ in range(MIN_SAMPLES)]
    )
    stats = build_statistics(attempts)
    assert stats["accuracy_by_rating_band"] == {}
    assert stats["weakest_rating_band"] is None
    # Everything else (accuracy, streaks, difficulty trajectory) still works —
    # only the puzzle-rating-specific stat is suppressed.
    assert stats["num_attempts"] == 3 + MIN_SAMPLES

def test_weakest_rating_band_identified_with_enough_samples():
    # Perfect at low rating, all wrong at a high rating band with >= MIN_SAMPLES.
    attempts = (
        [_attempt(puzzle_rating=900, correct=True) for _ in range(3)]
        + [_attempt(puzzle_rating=2000, correct=False) for _ in range(MIN_SAMPLES)]
    )
    stats = build_statistics(attempts)
    weakest = stats["weakest_rating_band"]
    assert weakest is not None
    assert weakest["key"] == "1800-2199"
    assert weakest["accuracy"] == 0.0
    assert weakest["count"] == MIN_SAMPLES

def test_weakest_rating_band_none_when_band_too_thin():
    # Only 1 attempt at the "bad" band — below MIN_SAMPLES, so no conclusion.
    attempts = [_attempt(puzzle_rating=900, correct=True) for _ in range(5)] + [_attempt(puzzle_rating=2000, correct=False)]
    stats = build_statistics(attempts)
    assert stats["weakest_rating_band"] is None

def test_weakest_rating_band_none_when_all_bands_equal():
    attempts = [_attempt(puzzle_rating=900, correct=True) for _ in range(MIN_SAMPLES)] + \
               [_attempt(puzzle_rating=1900, correct=True) for _ in range(MIN_SAMPLES)]
    stats = build_statistics(attempts)
    assert stats["weakest_rating_band"] is None

def test_weakest_state_identified():
    attempts = (
        [_attempt(predicted_state="Focused", correct=True) for _ in range(5)]
        + [_attempt(predicted_state="Overloaded", correct=False) for _ in range(MIN_SAMPLES)]
    )
    stats = build_statistics(attempts)
    weakest = stats["weakest_state"]
    assert weakest["key"] == "Overloaded"
    assert weakest["accuracy"] == 0.0

def test_avg_time_correct_and_wrong_are_computed_separately():
    attempts = [
        _attempt(correct=True, time_to_move=2.0), _attempt(correct=True, time_to_move=4.0),
        _attempt(correct=False, time_to_move=10.0),
    ]
    stats = build_statistics(attempts)
    assert stats["avg_time_correct"] == 3.0
    assert stats["avg_time_wrong"] == 10.0

def test_avg_time_wrong_is_none_when_no_wrong_attempts():
    attempts = [_attempt(correct=True) for _ in range(3)]
    stats = build_statistics(attempts)
    assert stats["avg_time_wrong"] is None

def test_avg_eval_loss_on_misses_only_counts_wrong_attempts():
    attempts = [_attempt(correct=True, eval_loss=0.0), _attempt(correct=False, eval_loss=200.0), _attempt(correct=False, eval_loss=100.0)]
    stats = build_statistics(attempts)
    assert stats["avg_eval_loss_on_misses"] == 150.0

def test_hint_rate():
    attempts = [_attempt(show_hint=True), _attempt(show_hint=True), _attempt(show_hint=False), _attempt(show_hint=False)]
    stats = build_statistics(attempts)
    assert stats["hint_rate"] == 0.5

def test_longest_streak_and_breaks():
    attempts = [
        _attempt(correct=True), _attempt(correct=True), _attempt(correct=False),
        _attempt(correct=True), _attempt(correct=True), _attempt(correct=True),
        _attempt(correct=False),
    ]
    stats = build_statistics(attempts)
    assert stats["longest_streak"] == 3
    assert stats["streak_breaks"] == 2

def test_difficulty_trajectory_matches_web_session_formula():
    # Mirrors TutorSession's max(400, difficulty + delta) starting at 1000.
    attempts = [_attempt(difficulty_delta=100.0), _attempt(difficulty_delta=-2000.0), _attempt(difficulty_delta=50.0)]
    stats = build_statistics(attempts)
    assert stats["difficulty_start"] == 1100.0
    assert stats["difficulty_end"] == 450.0  # floored at 400, then +50
    assert stats["difficulty_change"] == 450.0 - 1100.0
