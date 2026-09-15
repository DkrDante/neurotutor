import numpy as np
import torch
from model.evidence import evaluate_metrics, inspect_weights, load_model, trace_forward_pass
from model.fusion import FusionLSTMClassifier
from common.config import NUM_CHANNELS, BAND_NAMES, NUM_BEHAVIOR_FEATS, SEQ_LEN
from common.states import STATES

def _fresh_checkpoint(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.pt"
    model = FusionLSTMClassifier(
        num_channels=NUM_CHANNELS, num_bands=len(BAND_NAMES),
        num_behavior_feats=NUM_BEHAVIOR_FEATS, num_classes=len(STATES),
    )
    torch.save(model.state_dict(), checkpoint_path)
    return checkpoint_path

def test_load_model_returns_eval_mode_model_matching_checkpoint(tmp_path):
    model = load_model(_fresh_checkpoint(tmp_path))
    assert isinstance(model, FusionLSTMClassifier)
    assert not model.training

def test_inspect_weights_covers_every_parameter_with_real_statistics(tmp_path):
    model = load_model(_fresh_checkpoint(tmp_path))
    weights = inspect_weights(model)
    named = dict(model.named_parameters())
    assert {w["name"] for w in weights} == set(named)
    for w in weights:
        param = named[w["name"]]
        assert w["shape"] == list(param.shape)
        assert w["num_params"] == param.numel()
        assert len(w["sample_values"]) == min(8, param.numel())
        # A freshly-initialized nn.Linear weight is never all exactly zero.
        # Biases and the GCN's adjacency_logits are legitimately zero-initialized
        # (see model/graph.py), so they're excluded from this check.
        if "bias" not in w["name"] and "adjacency_logits" not in w["name"]:
            assert w["std"] > 0.0

def test_evaluate_metrics_returns_every_state_and_valid_confusion_matrix(tmp_path):
    model = load_model(_fresh_checkpoint(tmp_path))
    metrics = evaluate_metrics(model, split="test")
    assert metrics["split"] == "test"
    assert set(metrics["per_class"]) == set(STATES)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    cm = metrics["confusion_matrix"]
    assert len(cm) == len(STATES)
    assert all(len(row) == len(STATES) for row in cm)
    assert sum(sum(row) for row in cm) == metrics["num_samples"]
    for state in STATES:
        assert metrics["per_class"][state]["support"] >= 0

def _write_single_class_split(data_dir, label: int, num_samples: int = 6):
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    X_eeg = rng.normal(size=(num_samples, SEQ_LEN, NUM_CHANNELS, len(BAND_NAMES))).astype(np.float32)
    X_behavior = rng.random(size=(num_samples, SEQ_LEN, NUM_BEHAVIOR_FEATS)).astype(np.float32)
    y = np.full(num_samples, label, dtype=np.int64)
    np.savez(data_dir / "test.npz", X_eeg=X_eeg, X_behavior=X_behavior, y=y)

def test_evaluate_metrics_degrades_gracefully_when_a_split_has_only_one_class(tmp_path):
    # roc_auc_score raises ValueError when y_true contains a single class —
    # evaluate_metrics must catch that and report None rather than crash,
    # since a tiny/skewed held-out split can genuinely look like this.
    data_dir = tmp_path / "data"
    _write_single_class_split(data_dir, label=STATES.index("Focused"))
    model = load_model(_fresh_checkpoint(tmp_path))

    metrics = evaluate_metrics(model, data_dir=data_dir, split="test")

    assert metrics["roc_auc_macro"] is None
    assert all(m["roc_auc"] is None for m in metrics["per_class"].values())
    assert metrics["num_samples"] == 6
    assert 0.0 <= metrics["accuracy"] <= 1.0

def test_trace_forward_pass_differs_across_samples(tmp_path):
    # The strongest evidence against "hardcoded": two different real dataset
    # examples must produce genuinely different numbers at every stage.
    model = load_model(_fresh_checkpoint(tmp_path))
    trace_a = trace_forward_pass(model, split="test", sample_index=0)
    trace_b = trace_forward_pass(model, split="test", sample_index=1)
    assert trace_a["input_behavior_last_timestep"] != trace_b["input_behavior_last_timestep"]
    assert trace_a["logits"] != trace_b["logits"]
    assert trace_a["softmax_probabilities"] != trace_b["softmax_probabilities"]

def test_trace_forward_pass_softmax_probabilities_sum_to_one(tmp_path):
    model = load_model(_fresh_checkpoint(tmp_path))
    trace = trace_forward_pass(model, split="test", sample_index=0)
    assert abs(sum(trace["softmax_probabilities"].values()) - 1.0) < 1e-5
    assert trace["predicted_label"] == max(trace["softmax_probabilities"], key=trace["softmax_probabilities"].get)

def test_trace_forward_pass_rejects_out_of_range_index(tmp_path):
    model = load_model(_fresh_checkpoint(tmp_path))
    try:
        trace_forward_pass(model, split="test", sample_index=10**6)
        assert False, "expected ValueError"
    except ValueError:
        pass
