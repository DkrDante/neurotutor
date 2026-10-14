"""web/static/protocol.js is the experimental design; test its guarantees by
executing the real JavaScript under node (skipped if node isn't installed)."""
from __future__ import annotations
import json
import shutil
import subprocess
from pathlib import Path
import pytest

PROTOCOL_JS = Path(__file__).resolve().parent.parent / "web" / "static" / "protocol.js"
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


def _build(**opts):
    script = (
        f"const p = require({json.dumps(str(PROTOCOL_JS))});"
        f"console.log(JSON.stringify(p.buildProtocol({json.dumps(opts)})));"
    )
    out = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True, timeout=30)
    return json.loads(out.stdout)


def test_structure_baselines_practice_and_eight_scored_blocks():
    steps = _build(participant="P01", skill=1000)
    conditions = [s["condition"] for s in steps]
    assert conditions[:4] == ["Baseline_EO", "Baseline_EC", "Practice", "Practice_Dual"]
    assert conditions[-1] == "Baseline_EO_end"
    scored = [s for s in steps if s["type"] == "task" and s["rated"]]
    assert len(scored) == 8
    assert sorted(c["condition"] for c in scored[:6]) == ["Confused"] * 2 + ["Focused"] * 2 + ["Overloaded"] * 2
    assert [c["condition"] for c in scored[6:]] == ["Late_Focused"] * 2   # fatigue blocks come last


def test_no_condition_twice_in_a_row_for_many_participants():
    for i in range(60):
        order = [s["condition"] for s in _build(participant=f"P{i:02d}") if s["type"] == "task" and s["rated"]][:6]
        assert all(a != b for a, b in zip(order, order[1:])), order


def test_order_is_deterministic_per_participant_but_varies_across_them():
    assert _build(participant="P01") == _build(participant="P01")
    orders = {tuple(s["condition"] for s in _build(participant=f"P{i}") if s.get("rated") and s["type"] == "task") for i in range(20)}
    assert len(orders) > 3


def test_difficulty_bands_scale_with_skill_and_stay_inside_the_puzzle_set():
    for skill in (700, 1000, 1600, 2200):
        for s in _build(participant="P", skill=skill):
            if "band" in s:
                lo, hi = s["band"]
                assert 604 <= lo < hi <= 2472 and hi - lo >= 150
    by_cond = {s["condition"]: s["band"] for s in _build(participant="P", skill=1000) if "band" in s}
    assert by_cond["Focused"][1] < by_cond["Confused"][1] <= by_cond["Overloaded"][1]   # harder = higher band
    assert by_cond["Overloaded"][0] > by_cond["Focused"][0]


def test_only_overloaded_blocks_carry_the_digit_task():
    steps = _build(participant="P")
    assert {s["condition"] for s in steps if s.get("dual")} == {"Overloaded", "Practice_Dual"}


def test_quick_mode_is_short_and_block_length_is_configurable():
    quick = _build(participant="P", quick=True)
    assert all(s["seconds"] <= 20 for s in quick)
    long_blocks = [s["seconds"] for s in _build(participant="P", blockSeconds=300) if s.get("rated")]
    assert set(long_blocks) == {300}
