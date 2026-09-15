"""Daily play-streak calculation — pure and deterministic so it's unit
testable without a database: given today's date and the account's current
streak/freeze state, decides how the streak changes.

This is a Duolingo-style "played at least once today" calendar streak tied
to the account, deliberately separate from the puzzle-solve streak (tracked
client-side in app.js, reset by any wrong answer) — the two answer different
questions: "are you on a hot run right now" vs. "have you shown up daily."

Streak freezes protect against missing a SINGLE day (a 2-day gap between
last_play_date and today) — one freeze bridges exactly one missed day, not
multiple; a longer gap still resets the streak even with freezes banked.
Freezes are earned automatically, one per full week of streak reached, up to
a small cap, rather than requiring a separate "buy a freeze" action.
"""
from __future__ import annotations
import datetime

MAX_STREAK_FREEZES = 3
STREAK_FREEZE_EARN_INTERVAL_DAYS = 7


def compute_streak_update(
    last_play_date: str | None, current_streak: int, longest_streak: int,
    freezes_available: int, today: str,
) -> dict:
    """today/last_play_date are "YYYY-MM-DD" strings."""
    if last_play_date == today:
        return {
            "current_streak": current_streak, "longest_streak": longest_streak,
            "last_play_date": today, "changed_today": False,
            "freezes_available": freezes_available, "freeze_used": False,
        }

    freeze_used = False
    if last_play_date is not None:
        gap_days = (datetime.date.fromisoformat(today) - datetime.date.fromisoformat(last_play_date)).days
        if gap_days == 1:
            new_current = current_streak + 1
        elif gap_days == 2 and freezes_available > 0:
            new_current = current_streak + 1
            freeze_used = True
            freezes_available -= 1
        else:
            new_current = 1
    else:
        new_current = 1

    new_longest = max(longest_streak, new_current)
    if new_current % STREAK_FREEZE_EARN_INTERVAL_DAYS == 0 and freezes_available < MAX_STREAK_FREEZES:
        freezes_available += 1

    return {
        "current_streak": new_current, "longest_streak": new_longest,
        "last_play_date": today, "changed_today": True,
        "freezes_available": freezes_available, "freeze_used": freeze_used,
    }
