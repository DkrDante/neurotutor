"""Turns a session's raw per-attempt log (storage.db.SessionStore.get_session_attempts)
into the real, computed diagnostics a coaching report is built from — which
puzzle-rating band is actually costing the player accuracy, whether the
EEG-driven cognitive state correlates with more mistakes, whether they rush
or overthink before a blunder, and how severe their misses tend to be.

Every number here is a deterministic aggregation over the attempts passed in
— nothing here calls a model or fabricates a statistic.
"""
from __future__ import annotations

# Puzzle-rating buckets: (low, high inclusive, label).
RATING_BANDS = [
    (0, 999, "under 1000"),
    (1000, 1399, "1000-1399"),
    (1400, 1799, "1400-1799"),
    (1800, 2199, "1800-2199"),
    (2200, 100000, "2200+"),
]
# Don't draw a conclusion from a band/state with too few samples to mean anything.
MIN_SAMPLES = 3


def _band_for_rating(rating: int) -> str:
    for low, high, label in RATING_BANDS:
        if low <= rating <= high:
            return label
    return "unrated"


def _avg(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _accuracy_breakdown(groups: dict[str, list[bool]]) -> dict[str, dict]:
    return {key: {"accuracy": sum(flags) / len(flags), "count": len(flags)} for key, flags in groups.items()}


def _weakest(breakdown: dict[str, dict], overall_accuracy: float) -> dict | None:
    """The lowest-accuracy key with enough samples to trust, or None if every
    group is too thin or none are meaningfully below the overall accuracy."""
    eligible = {key: v for key, v in breakdown.items() if v["count"] >= MIN_SAMPLES}
    if not eligible:
        return None
    key, values = min(eligible.items(), key=lambda kv: kv[1]["accuracy"])
    if values["accuracy"] >= overall_accuracy - 0.001:
        return None
    return {"key": key, **values}


def build_statistics(attempts: list[dict]) -> dict:
    if not attempts:
        return {"num_attempts": 0}

    num_attempts = len(attempts)
    correct_attempts = [a for a in attempts if a["correct"]]
    wrong_attempts = [a for a in attempts if not a["correct"]]
    accuracy_rate = len(correct_attempts) / num_attempts

    # Rating bands only mean something for puzzle mode — full-game attempts
    # (chess_task/full_game.py) repurpose puzzle_rating to mean the live
    # adaptive-difficulty number, so bucketing THAT into "1800-2199" etc.
    # would report a nonsense "weakest rating band" for a game session.
    is_game_mode = attempts[0].get("mode") == "game"
    by_band: dict[str, list[bool]] = {}
    if not is_game_mode:
        for a in attempts:
            by_band.setdefault(_band_for_rating(a["puzzle_rating"]), []).append(a["correct"])
    accuracy_by_rating_band = _accuracy_breakdown(by_band)

    by_state: dict[str, list[bool]] = {}
    for a in attempts:
        by_state.setdefault(a["predicted_state"], []).append(a["correct"])
    accuracy_by_state = _accuracy_breakdown(by_state)

    # Longest correct-answer streak and how many times it broke, in play order.
    longest_streak = 0
    current_streak = 0
    streak_breaks = 0
    for a in attempts:
        if a["correct"]:
            current_streak += 1
            longest_streak = max(longest_streak, current_streak)
        else:
            if current_streak > 0:
                streak_breaks += 1
            current_streak = 0

    # Recomputes the same running calculation TutorSession performs live (see
    # web/session.py) to see the session's overall difficulty trajectory.
    difficulty = 1000.0
    trajectory = []
    for a in attempts:
        difficulty = max(400.0, difficulty + a["difficulty_delta"])
        trajectory.append(difficulty)

    return {
        "num_attempts": num_attempts,
        "accuracy_rate": accuracy_rate,
        "accuracy_by_rating_band": accuracy_by_rating_band,
        "weakest_rating_band": _weakest(accuracy_by_rating_band, accuracy_rate),
        "accuracy_by_state": accuracy_by_state,
        "weakest_state": _weakest(accuracy_by_state, accuracy_rate),
        "avg_time_correct": _avg([a["time_to_move"] for a in correct_attempts]),
        "avg_time_wrong": _avg([a["time_to_move"] for a in wrong_attempts]),
        "avg_eval_loss_on_misses": _avg([a["eval_loss"] for a in wrong_attempts]),
        "hint_rate": sum(1 for a in attempts if a["show_hint"]) / num_attempts,
        "longest_streak": longest_streak,
        "streak_breaks": streak_breaks,
        "difficulty_start": trajectory[0] if trajectory else 1000.0,
        "difficulty_end": trajectory[-1] if trajectory else 1000.0,
        "difficulty_change": (trajectory[-1] - trajectory[0]) if trajectory else 0.0,
    }
