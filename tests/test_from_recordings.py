"""Recording -> labelled dataset (data_gen/from_recordings.py), end to end on a
synthetic /collect-style recording written by the real recorder."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pytest
from common.states import STATE_TO_INDEX, STATES
from data_gen.fake_recording import build_fake_recording
from data_gen.from_recordings import (
    Block, Thresholds, build_from_recording, decide_blocks, find_recordings, fit_baseline, load, merge,
    parse_blocks, save,
)
from eeg.recorder import load_recording


@pytest.fixture(scope="module")
def recording(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("rec")
    return build_fake_recording("SIM01", root, block_seconds=240, seed=1, inconsistent_condition_index=2)


def _block(condition, ratings, **kw):
    return Block(block=kw.pop("block", "b"), condition=condition, kind="task", start=0.0, end=240.0, ratings=ratings, **kw)


# --------------------------------------------------------------- block logic
def test_parse_blocks_pairs_markers_and_attaches_ratings(recording):
    rec = load_recording(recording)
    blocks = parse_blocks(rec["markers"], float(rec["times"][-1]))
    scored = [b for b in blocks if b.condition in ("Focused", "Confused", "Overloaded", "Late_Focused")]
    assert len(scored) == 8 and all(b.ratings and "mind" in b.ratings for b in scored)
    assert all(b.end > b.start and not b.incomplete for b in blocks)


def test_unfinished_block_is_flagged_incomplete():
    blocks = parse_blocks([{"kind": "block_start", "t": 10.0, "block": "x", "condition": "Focused"}], 50.0)
    assert blocks[0].incomplete and blocks[0].end == 50.0


@pytest.mark.parametrize("condition,ratings,expected", [
    ("Overloaded", dict(mental_demand=6, frustration=2), True),
    ("Overloaded", dict(mental_demand=3, frustration=2), False),      # "overloaded" but felt easy
    ("Confused", dict(confusion=5), True),
    ("Confused", dict(confusion=2), False),
    ("Focused", dict(concentration=6, mental_demand=3), True),
    ("Focused", dict(concentration=2, mental_demand=3), False),
    ("Focused", dict(concentration=6, mental_demand=7), False),       # that was not an easy block
])
def test_manipulation_checks(condition, ratings, expected):
    (decision,) = decide_blocks([_block(condition, ratings)], Thresholds())
    assert decision.used is expected
    assert ("passed" in decision.reason) is expected or not expected


def test_fatigue_needs_absolute_level_and_a_rise_over_early_blocks():
    early = [_block("Focused", dict(concentration=6, mental_demand=3, fatigue=2), block="e1"),
             _block("Confused", dict(confusion=5, fatigue=3), block="e2")]
    tired = _block("Late_Focused", dict(fatigue=6), block="l1")
    barely = _block("Late_Focused", dict(fatigue=5), block="l2")
    not_tired = _block("Late_Focused", dict(fatigue=3), block="l3")
    decisions = {d.block: d for d in decide_blocks(early + [tired, barely, not_tired], Thresholds())}
    assert decisions["l1"].used and decisions["l1"].state == "Fatigued"
    assert not decisions["l3"].used
    # mean early fatigue is 2.5; 5 >= 2.5 + 1 passes, so "barely" is accepted
    assert decisions["l2"].used


def test_ignored_conditions_and_missing_ratings_are_never_used():
    decisions = decide_blocks([_block("Practice", None), _block("Baseline_EO", None), _block("Focused", None)], Thresholds())
    assert not any(d.used for d in decisions)
    assert "no self-report" in decisions[2].reason
    lenient = decide_blocks([_block("Focused", None)], Thresholds(require_ratings=False))
    assert lenient[0].used


# ------------------------------------------------------------- dataset build
def test_attempt_aligned_dataset_labels_shapes_and_block_drop(recording):
    ds = build_from_recording(recording, alignment="attempts")
    assert ds.X_eeg.shape[1:] == (5, 8, 3) and ds.X_behavior.shape[1:] == (5, 4)
    assert len(ds) == len(ds.y) == len(ds.group) > 50
    used = {r["block"]: r for r in ds.report if r["used"]}
    dropped = [r for r in ds.report if not r["used"] and r["condition"] in ("Focused", "Confused", "Overloaded", "Late_Focused")]
    assert any("FAILED" in r["reason"] for r in dropped)                    # the deliberately inconsistent block
    assert set(ds.y) <= {STATE_TO_INDEX[s] for s in ("Focused", "Overloaded", "Confused", "Fatigued")}
    assert STATE_TO_INDEX["Engaged"] not in set(ds.y)                       # no induction condition for Engaged
    for cond, state in {"Focused": "Focused", "Confused": "Confused", "Overloaded": "Overloaded", "Late_Focused": "Fatigued"}.items():
        sel = ds.condition == cond
        if sel.any():
            assert set(ds.y[sel]) == {STATE_TO_INDEX[state]}
    assert set(ds.condition) <= {"Focused", "Confused", "Overloaded", "Late_Focused"}   # no practice / baseline / rest
    assert all(g.startswith("SIM01/") for g in ds.group) and set(g.split("/")[-1] for g in ds.group) <= set(used)
    assert np.isfinite(ds.X_eeg).all() and (ds.X_eeg >= 0).all()           # band powers


def test_features_are_in_the_scale_the_model_was_built_for(recording):
    """After baseline normalisation real-path band power should sit in the same
    ballpark as the synthetic training data's (not µV-sized, not ~0)."""
    ds = build_from_recording(recording, alignment="attempts")
    synthetic = np.load("data/synthetic/train.npz")["X_eeg"] if Path("data/synthetic/train.npz").exists() else None
    mean_theta_alpha = ds.X_eeg[..., :2].mean()
    assert 0.005 < mean_theta_alpha < 5.0
    if synthetic is not None:
        ratio = mean_theta_alpha / synthetic[..., :2].mean()
        assert 0.05 < ratio < 20


def test_behavior_vector_matches_the_attempts(recording):
    ds = build_from_recording(recording, alignment="attempts")
    beh = ds.X_behavior
    assert set(np.unique(beh[..., 0])) <= {0.0, 1.0}                         # correct flag
    assert (beh[..., 1] > 0).all() and (beh[..., 1] <= 1).all()              # time_to_move / 30
    assert (beh[..., 2] >= 0).all() and (beh[..., 3] > 0).all()


def test_time_aligned_dataset_has_zero_behavior_and_more_examples(recording):
    attempts = build_from_recording(recording, alignment="attempts")
    timed = build_from_recording(recording, alignment="time")
    assert len(timed) > len(attempts)
    assert np.all(timed.X_behavior == 0)
    assert timed.alignment == "time"


def test_disturbance_note_removes_overlapping_windows(recording, tmp_path):
    import shutil
    copy = tmp_path / "copy"
    shutil.copytree(recording, copy)
    before = build_from_recording(copy, alignment="time")
    rec = load_recording(copy)
    block = next(b for b in parse_blocks(rec["markers"], float(rec["times"][-1])) if b.block == next(r["block"] for r in before.report if r["used"]))
    middle = (block.start + block.end) / 2
    with open(copy / "markers.jsonl", "a") as f:
        f.write(json.dumps({"t": middle, "kind": "note", "note": "disturbance"}) + "\n")
    after = build_from_recording(copy, alignment="time")
    assert len(after) < len(before)


def test_baseline_comes_from_the_eyes_open_block_and_falls_back_gracefully(recording):
    rec = load_recording(recording)
    blocks = parse_blocks(rec["markers"], float(rec["times"][-1]))
    baseline, source = fit_baseline(rec, blocks)
    assert source == "eyes-open baseline block" and len(baseline["mean"]) == 8
    baseline2, source2 = fit_baseline(rec, [b for b in blocks if b.condition != "Baseline_EO"])
    assert "whole recording" in source2


def test_save_load_round_trip_and_merge(recording, tmp_path):
    ds = build_from_recording(recording, alignment="attempts")
    path = tmp_path / "ds.npz"
    save(ds, path)
    again = load(path)
    assert np.array_equal(again.y, ds.y) and np.array_equal(again.group, ds.group)
    assert np.allclose(again.X_eeg, ds.X_eeg) and again.alignment == "attempts" and again.report
    assert len(merge([ds, again])) == 2 * len(ds)


def test_find_recordings_accepts_a_directory_or_a_participant_folder(recording):
    assert find_recordings([recording]) == [recording]
    assert find_recordings([recording.parent]) == [recording]
    assert find_recordings([recording.parent.parent]) == [recording]


# ------------------------------------------- the three-question report (current /collect)
def R(difficulty, mind, sleepiness=2):
    return dict(difficulty=difficulty, mind=mind, sleepiness=sleepiness)


@pytest.mark.parametrize("condition,ratings,expected", [
    ("Focused", R(2, "sharp"), True),
    ("Focused", R(3, "pushing"), True),
    ("Focused", R(4, "sharp"), False),            # "too hard" is not a Focused block
    ("Focused", R(2, "lost"), False),
    ("Focused", R(2, "drifting"), False),
    ("Confused", R(3, "lost"), True),
    ("Confused", R(2, "overwhelmed"), True),
    ("Confused", R(4, "pushing"), True),          # "too hard" counts even if they kept up
    ("Confused", R(2, "sharp"), False),           # "confused" but felt sharp and easy
    ("Overloaded", R(3, "pushing"), True),
    ("Overloaded", R(2, "overwhelmed"), True),
    ("Overloaded", R(2, "sharp"), False),         # overloaded block that felt easy: dropped
])
def test_manipulation_checks_three_question_report(condition, ratings, expected):
    (decision,) = decide_blocks([_block(condition, ratings)], Thresholds())
    assert decision.used is expected


def test_fatigue_check_three_question_report():
    early = [_block("Focused", R(2, "sharp", 1), block="e1"), _block("Confused", R(3, "lost", 2), block="e2")]  # mean 1.5
    cases = {
        "tired": R(2, "drifting", 4), "very": R(2, "sharp", 5), "drifty": R(2, "drifting", 3),
        "alert": R(2, "sharp", 2), "sleepy_but_sharp": R(2, "sharp", 3),
    }
    blocks = [_block("Late_Focused", r, block=k) for k, r in cases.items()]
    decisions = {d.block: d for d in decide_blocks(early + blocks, Thresholds())}
    assert decisions["tired"].used and decisions["tired"].state == "Fatigued"
    assert decisions["very"].used and decisions["drifty"].used
    assert not decisions["alert"].used and not decisions["sleepy_but_sharp"].used


def test_fatigue_needs_a_rise_over_early_blocks_in_three_question_report():
    already_sleepy = [_block("Focused", R(2, "sharp", 4), block="e1"), _block("Confused", R(3, "lost", 4), block="e2")]
    late = _block("Late_Focused", R(2, "drifting", 4), block="l1")      # tired, but no more than at the start
    (d,) = [x for x in decide_blocks(already_sleepy + [late], Thresholds()) if x.block == "l1"]
    assert not d.used


def test_fake_recording_in_both_report_formats_is_understood(tmp_path):
    from data_gen.fake_recording import build_fake_recording
    for legacy in (False, True):
        d = build_fake_recording(f"L{int(legacy)}", tmp_path, block_seconds=20, seed=3, legacy_sliders=legacy)
        rec = load_recording(d)
        blocks = parse_blocks(rec["markers"], float(rec["times"][-1]))
        decisions = decide_blocks(blocks, Thresholds())
        scored = [x for x in decisions if x.condition in ("Focused", "Confused", "Overloaded", "Late_Focused")]
        assert sum(x.used for x in scored) >= 6                         # a consistent fake participant passes the checks
