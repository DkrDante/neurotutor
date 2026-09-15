import torch
from model.evidence import compute_feature_importance
from model.fusion import FusionLSTMClassifier
from common.config import NUM_CHANNELS, BAND_NAMES, NUM_BEHAVIOR_FEATS
from common.states import STATES

def _fresh_checkpoint(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.pt"
    model = FusionLSTMClassifier(
        num_channels=NUM_CHANNELS, num_bands=len(BAND_NAMES),
        num_behavior_feats=NUM_BEHAVIOR_FEATS, num_classes=len(STATES),
    )
    torch.save(model.state_dict(), checkpoint_path)
    return checkpoint_path

def test_feature_importance_covers_every_behavior_feature_and_eeg_channel(tmp_path):
    from model.evidence import load_model
    model = load_model(_fresh_checkpoint(tmp_path))

    result = compute_feature_importance(model, split="test")

    assert 0.0 <= result["baseline_accuracy"] <= 1.0
    behavior_names = {r["feature"] for r in result["behavior_importance"]}
    assert behavior_names == {"correct", "time_to_move", "eval_loss", "puzzle_rating"}
    assert len(result["eeg_channel_importance"]) == NUM_CHANNELS
    assert {r["channel"] for r in result["eeg_channel_importance"]} == set(range(NUM_CHANNELS))

def test_feature_importance_is_reproducible_with_a_fixed_seed(tmp_path):
    from model.evidence import load_model
    checkpoint_path = _fresh_checkpoint(tmp_path)
    model_a = load_model(checkpoint_path)
    model_b = load_model(checkpoint_path)

    result_a = compute_feature_importance(model_a, split="test", seed=3)
    result_b = compute_feature_importance(model_b, split="test", seed=3)

    assert result_a == result_b

def test_feature_importance_rankings_are_sorted_descending(tmp_path):
    from model.evidence import load_model
    model = load_model(_fresh_checkpoint(tmp_path))
    result = compute_feature_importance(model, split="test")

    behavior_scores = [r["importance"] for r in result["behavior_importance"]]
    assert behavior_scores == sorted(behavior_scores, reverse=True)
    eeg_scores = [r["importance"] for r in result["eeg_channel_importance"]]
    assert eeg_scores == sorted(eeg_scores, reverse=True)
