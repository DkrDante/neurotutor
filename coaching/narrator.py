"""Turns coaching.statistics.build_statistics()'s output into readable
feedback, two ways:

  - rule_based_tips(): always available, no external dependency — a short,
    deterministic list of tips, each one naming the real numbers behind it.
  - ai_narrative(): asks a locally-running Ollama model (no API key, no
    cloud call — see OLLAMA_HOST/OLLAMA_MODEL below) to turn the SAME
    computed statistics into a short coaching paragraph. Returns None on
    any failure (Ollama not running, model not pulled, network error) so
    callers always have the rule-based tips to fall back to — this feature
    never depends on Ollama being installed or running.
"""
from __future__ import annotations
import json
import os

STATE_ADVICE = {
    "Overloaded": "slow down and re-scan the whole board before moving",
    "Confused": "break the position down — look for checks, captures, and threats one at a time",
    "Fatigued": "consider a short break; solving while fatigued is measurably costing you accuracy",
}

# Local Ollama server (https://ollama.com) — `ollama serve` (or the desktop
# app) must already be running with OLLAMA_MODEL pulled (`ollama pull
# <model>`); both are overridable so this works with whatever's already on
# the machine instead of assuming one specific model.
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
OLLAMA_TIMEOUT_SECONDS = 20.0


def rule_based_tips(stats: dict) -> list[str]:
    if stats.get("num_attempts", 0) == 0:
        return ["Play a few puzzles first — there's nothing to analyze yet."]

    tips: list[str] = []
    overall_pct = stats["accuracy_rate"] * 100

    weakest_band = stats.get("weakest_rating_band")
    if weakest_band:
        tips.append(
            f"Your accuracy drops to {weakest_band['accuracy'] * 100:.0f}% on puzzles rated "
            f"{weakest_band['key']} (vs {overall_pct:.0f}% overall, over {weakest_band['count']} attempts) "
            "— that's the rating band to drill next."
        )

    weakest_state = stats.get("weakest_state")
    if weakest_state:
        state = weakest_state["key"]
        advice = STATE_ADVICE.get(state, "pay extra attention to your process in this state")
        tips.append(
            f"When the model reads you as {state}, your accuracy falls to {weakest_state['accuracy'] * 100:.0f}% "
            f"(vs {overall_pct:.0f}% overall) — try to {advice}."
        )

    avg_correct, avg_wrong = stats.get("avg_time_correct"), stats.get("avg_time_wrong")
    if avg_correct is not None and avg_wrong is not None:
        if avg_wrong < avg_correct * 0.7:
            tips.append(
                f"You move noticeably faster right before a mistake ({avg_wrong:.1f}s) than before a correct "
                f"answer ({avg_correct:.1f}s) — you're likely rushing rather than missing the idea."
            )
        elif avg_wrong > avg_correct * 1.5:
            tips.append(
                f"Your wrong moves actually take longer ({avg_wrong:.1f}s) than your correct ones "
                f"({avg_correct:.1f}s) — that points to calculation errors, not carelessness; double-check "
                "candidate lines before committing."
            )

    avg_loss = stats.get("avg_eval_loss_on_misses")
    if avg_loss is not None and avg_loss > 150:
        tips.append(
            f"Your misses average a {avg_loss:.0f}-centipawn loss — these are outright blunders, not close "
            "calls. Run a hanging-piece / one-move-check scan before every move."
        )

    if stats.get("hint_rate", 0) > 0.4:
        tips.append(
            f"You requested a hint on {stats['hint_rate'] * 100:.0f}% of attempts — try committing to "
            "2-3 candidate moves yourself before asking for one."
        )

    change = stats.get("difficulty_change", 0)
    if change > 100:
        tips.append(f"Difficulty climbed by {change:.0f} points this session — solid upward trajectory.")
    elif change < -100:
        tips.append(
            f"Difficulty eased by {abs(change):.0f} points this session — the adaptive engine backed off "
            "after a rough patch; a good time to consolidate fundamentals before pushing again."
        )

    if not tips:
        tips.append("No single weak spot stands out yet — a solid, consistent session. Keep playing to surface patterns.")

    return tips


def ai_narrative(stats: dict) -> str | None:
    if stats.get("num_attempts", 0) == 0:
        return None
    try:
        import httpx
    except ImportError:
        return None

    prompt = (
        "You are a chess coach reviewing one student's practice session. Here are the "
        f"real, computed statistics from their session, as JSON:\n{json.dumps(stats, default=str)}\n\n"
        "Write a short (120-180 word) coaching summary: name the one or two clearest patterns "
        "in this data (a weak rating band, a weak cognitive state, rushing vs overthinking, blunder "
        "severity, hint reliance), and give 2-4 concrete, specific tips. Only reference numbers that "
        "appear in the JSON above — never invent a statistic. Be direct and encouraging, not generic."
    )
    try:
        response = httpx.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=OLLAMA_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        text = response.json().get("response", "").strip()
        return text or None
    except Exception:
        # Ollama not running, model not pulled, a slow/failed generation —
        # this must never break the coaching report; the rule-based tips
        # above always work regardless.
        return None


def build_coaching_report(stats: dict) -> dict:
    return {
        "statistics": stats,
        "tips": rule_based_tips(stats),
        "narrative": ai_narrative(stats),
    }
