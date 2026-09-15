"""Replays every formula behind the tutor's numbers with the REAL values from
one session plugged in — not just the formula text (see web/server.py's
_calculation_notes for that), but the actual substitution and result for this
session's own attempts.

Two kinds of output:
- `build_aggregate_calculations`: the same summary numbers coaching.statistics
  already computes, paired with their formula and real substituted values.
- `build_attempt_trace`: a per-attempt walk of the running accuracy and the
  difficulty adjustment, including a reconstruction of what the rule-based
  policy (adaptive.rule_based) would have produced for that attempt so a
  divergence from the recorded delta is a visible signal that the session
  actually ran under the RL policy instead (attempts don't record which
  policy was active, so this is inferred rather than read directly).
"""
from __future__ import annotations
from adaptive.rule_based import CONFIDENCE_THRESHOLD, CORRECT_DIFFICULTY_DELTA, STATE_RULES, WRONG_DIFFICULTY_DELTA
from coaching.statistics import build_statistics

DIFFICULTY_FLOOR = 400.0
DIFFICULTY_START = 1000.0


def build_attempt_trace(attempts: list[dict]) -> list[dict]:
    trace = []
    correct_so_far = 0
    difficulty = DIFFICULTY_START
    for i, a in enumerate(attempts):
        correct_so_far += 1 if a["correct"] else 0

        correctness_delta = CORRECT_DIFFICULTY_DELTA if a["correct"] else WRONG_DIFFICULTY_DELTA
        confidence_trusted = a["confidence"] >= CONFIDENCE_THRESHOLD
        state_delta = STATE_RULES[a["predicted_state"]].difficulty_delta if confidence_trusted else 0.0
        rule_based_delta = correctness_delta + state_delta
        actual_delta = a["difficulty_delta"]

        difficulty_before = difficulty
        difficulty = max(DIFFICULTY_FLOOR, difficulty + actual_delta)

        trace.append({
            "index": i,
            "puzzle_id": a["puzzle_id"],
            "correct": a["correct"],
            "predicted_state": a["predicted_state"],
            "confidence": a["confidence"],
            "eval_loss": a["eval_loss"],
            "puzzle_rating": a["puzzle_rating"],
            "time_to_move": a["time_to_move"],
            "show_hint": a["show_hint"],
            "running_accuracy": correct_so_far / (i + 1),
            "difficulty_before": difficulty_before,
            "difficulty_after": difficulty,
            "actual_delta": actual_delta,
            "rule_based_reconstruction": {
                "correctness_delta": correctness_delta,
                "confidence_trusted": confidence_trusted,
                "state_delta_applied": state_delta,
                "predicted_delta": rule_based_delta,
                "matches_actual": abs(rule_based_delta - actual_delta) < 0.01,
            },
        })
    return trace


def build_aggregate_calculations(attempts: list[dict]) -> list[dict]:
    stats = build_statistics(attempts)
    if stats["num_attempts"] == 0:
        return []

    n = stats["num_attempts"]
    correct_n = sum(1 for a in attempts if a["correct"])
    wrong_n = n - correct_n
    hint_n = sum(1 for a in attempts if a["show_hint"])

    calcs = [
        {
            "title": "Overall accuracy",
            "formula": "correct_attempts / total_attempts",
            "substitution": f"{correct_n} / {n}",
            "result": f"{stats['accuracy_rate'] * 100:.1f}%",
        },
        {
            "title": "Hint reliance",
            "formula": "attempts_shown_a_hint / total_attempts",
            "substitution": f"{hint_n} / {n}",
            "result": f"{stats['hint_rate'] * 100:.1f}%",
        },
        {
            "title": "Difficulty trajectory",
            "formula": "max(400, difficulty + delta) applied per attempt, starting at 1000",
            "substitution": f"{stats['difficulty_start']:.0f} → {stats['difficulty_end']:.0f} over {n} attempts",
            "result": f"{stats['difficulty_change']:+.0f}",
        },
        {
            "title": "Longest correct streak",
            "formula": "longest run of consecutive correct attempts, in play order",
            "substitution": f"broke {stats['streak_breaks']} time(s) across {n} attempts",
            "result": str(stats["longest_streak"]),
        },
    ]

    if stats["avg_time_correct"] is not None or stats["avg_time_wrong"] is not None:
        parts = []
        if stats["avg_time_correct"] is not None:
            parts.append(f"{stats['avg_time_correct']:.2f}s correct (n={correct_n})")
        if stats["avg_time_wrong"] is not None:
            parts.append(f"{stats['avg_time_wrong']:.2f}s wrong (n={wrong_n})")
        calcs.append({
            "title": "Decision speed",
            "formula": "mean(time_to_move), grouped by correct vs. wrong",
            "substitution": " vs. ".join(parts),
            "result": (
                "rushing on misses" if stats["avg_time_correct"] and stats["avg_time_wrong"]
                and stats["avg_time_wrong"] < stats["avg_time_correct"]
                else "overthinking on misses" if stats["avg_time_correct"] and stats["avg_time_wrong"]
                else "n/a"
            ),
        })

    if stats["avg_eval_loss_on_misses"] is not None:
        calcs.append({
            "title": "Blunder severity",
            "formula": "mean(eval_loss for wrong attempts)",
            "substitution": f"n={wrong_n} wrong attempt(s)",
            "result": f"{stats['avg_eval_loss_on_misses']:.0f} centipawns",
        })

    if stats["weakest_rating_band"]:
        wb = stats["weakest_rating_band"]
        calcs.append({
            "title": "Weakest puzzle-rating band",
            "formula": "lowest-accuracy rating band with ≥ 3 attempts and accuracy below the session average",
            "substitution": f"{wb['key']}: {wb['count']} attempts",
            "result": f"{wb['accuracy'] * 100:.0f}% (vs {stats['accuracy_rate'] * 100:.0f}% overall)",
        })

    if stats["weakest_state"]:
        ws = stats["weakest_state"]
        calcs.append({
            "title": "Weakest cognitive state",
            "formula": "lowest-accuracy predicted state with ≥ 3 attempts and accuracy below the session average",
            "substitution": f"{ws['key']}: {ws['count']} attempts",
            "result": f"{ws['accuracy'] * 100:.0f}% (vs {stats['accuracy_rate'] * 100:.0f}% overall)",
        })

    return calcs


def build_calculation_trace(attempts: list[dict]) -> dict:
    return {
        "num_attempts": len(attempts),
        "aggregate": build_aggregate_calculations(attempts),
        "attempts": build_attempt_trace(attempts),
    }
