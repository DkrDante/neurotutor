"""model/train_real.py: block-wise CV, baselines, the CLI, and the serving-side
feature-mask hook."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pytest
from common.states import STATE_TO_INDEX, STATES
from data_gen.fake_recording import build_fake_recording
from data_gen.from_recordings import build_from_recording, merge, save
from model import train_real
from model.inference import StatePredictor


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("rec")
    parts = [build_from_recording(build_fake_recording(p, root, block_seconds=240, seed=20 + i, separation=0.6), alignment="attempts")
             for i, p in enumerate(["SIM01", "SIM02"])]
    return merge(parts)


def test_block_folds_never_mix_a_block_across_train_and_test_and_cover_every_class(dataset):
    folds = train_real.block_folds(dataset.group, dataset.y)
    assert len(folds) >= 2
    all_test = np.concatenate([t for _, t in folds])
    assert sorted(all_test) == list(range(len(dataset)))                      # every example tested exactly once
    for train, test in folds:
        assert not set(dataset.group[train]) & set(dataset.group[test])       # no block on both sides
        assert set(dataset.y[test]) == set(dataset.y)                         # every state present in every test fold


def test_block_folds_refuse_a_state_with_a_single_block():
    groups = np.array(["a"] * 5 + ["b"] * 5 + ["c"] * 5)
    y = np.array([0] * 5 + [0] * 5 + [1] * 5)       # state 1 has only block c
    with pytest.raises(ValueError, match="at least|>= 2"):
        train_real.block_folds(groups, y)


def test_participant_folds_need_two_participants(dataset):
    folds = train_real.participant_folds(dataset.participant, dataset.y)
    assert len(folds) == 2
    for train, test in folds:
        assert not set(dataset.participant[train]) & set(dataset.participant[test])
    one = dataset.participant.copy(); one[:] = "SIM01"
    with pytest.raises(ValueError):
        train_real.participant_folds(one, dataset.y)


def test_cross_validation_reports_models_and_baselines(dataset):
    folds = train_real.block_folds(dataset.group, dataset.y)
    results = train_real.cross_validate(dataset, folds, behavior_mask=train_real.DEFAULT_BEHAVIOR_MASK, epochs=8, seed=0, ablation=True, log=lambda *_: None)
    assert set(results) == {"fused", "eeg_only", "behavior_only", "majority", "logreg_behavior", "logreg_eeg", "logreg_both"}
    for r in results.values():
        assert r["n"] == len(dataset) and 0 <= r["accuracy"] <= 1 and 0 <= r["balanced_accuracy"] <= 1
        assert len(r["per_block"]) == len(set(dataset.group))
    assert results["majority"]["balanced_accuracy"] <= results["fused"]["balanced_accuracy"] + 1e-9
    assert results["logreg_eeg"]["balanced_accuracy"] > 0.5                   # the signal is learnable at all
    report = train_real.format_report(dataset, results, "blocks-wise", len(folds), 8)
    assert "Blocks right" in report and "Caveats" in report and "logreg_eeg" in report


def test_the_default_mask_removes_the_protocol_confounded_rating_feature():
    assert train_real.DEFAULT_BEHAVIOR_MASK[3] == 0.0 and sum(train_real.EEG_ONLY_MASK) == 0.0


def test_cli_trains_saves_checkpoint_with_config_and_the_predictor_honours_the_mask(dataset, tmp_path, monkeypatch):
    npz = tmp_path / "real.npz"
    save(dataset, npz)
    ckpt = tmp_path / "real_fused.pt"
    monkeypatch.setattr(sys, "argv", ["train_real", str(npz), "--checkpoint", str(ckpt), "--epochs", "5"])
    train_real.main()
    config = json.loads(Path(str(ckpt) + ".config.json").read_text())
    assert config["behavior_mask"] == [1.0, 1.0, 1.0, 0.0] and config["trained_on"]["participants"] == ["SIM01", "SIM02"]
    assert "Engaged" not in config["trained_on"]["states_present"]
    assert ckpt.with_name(ckpt.name + ".report.md").exists()

    predictor = StatePredictor(ckpt)
    assert predictor.behavior_mask is not None and predictor.behavior_mask[3] == 0.0
    eeg = dataset.X_eeg[0]
    beh_a = dataset.X_behavior[0].copy()
    beh_b = beh_a.copy(); beh_b[:, 3] = 0.99          # wildly different puzzle rating
    state_a, conf_a, _ = predictor.predict(eeg, beh_a)
    state_b, conf_b, _ = predictor.predict(eeg, beh_b)
    assert state_a == state_b and abs(conf_a - conf_b) < 1e-6                  # rating is ignored, as trained


def test_original_synthetic_checkpoint_still_loads_without_a_mask():
    ckpt = Path("model/checkpoints/best.pt")
    if not ckpt.exists():
        pytest.skip("no synthetic checkpoint in this checkout")
    assert StatePredictor(ckpt).behavior_mask is None


def test_eeg_only_cli_masks_all_behaviour(dataset, tmp_path, monkeypatch):
    npz = tmp_path / "time.npz"
    root = tmp_path / "rec"
    timed = merge([build_from_recording(build_fake_recording(p, root, block_seconds=240, seed=30 + i, separation=0.6), alignment="time")
                   for i, p in enumerate(["SIM01", "SIM02"])])
    save(timed, npz)
    ckpt = tmp_path / "live_eeg.pt"
    monkeypatch.setattr(sys, "argv", ["train_real", str(npz), "--checkpoint", str(ckpt), "--eeg-only", "--epochs", "4", "--skip-cv"])
    train_real.main()
    assert json.loads(Path(str(ckpt) + ".config.json").read_text())["behavior_mask"] == [0.0, 0.0, 0.0, 0.0]
