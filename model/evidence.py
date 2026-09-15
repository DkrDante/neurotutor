"""Real evidence that the fusion classifier is a genuinely trained model doing
real computation — not a stub, not hardcoded, not synthetic output. Three
things, computed fresh from the actual checkpoint and dataset every call:

1. `inspect_weights`: every learnable parameter's real shape and value
   statistics, straight from the trained checkpoint's state_dict.
2. `evaluate_metrics`: precision/recall/F1 (per class + macro/weighted) and
   multi-class ROC-AUC (one-vs-rest), computed independently of
   model/train.py's accuracy+macro-F1-only training-loop metric — this is not
   the number the checkpoint was selected by, it's a separate check against it.
3. `trace_forward_pass`: a full, real forward pass on one real dataset
   example, computed submodule-by-submodule (same technique as
   model/inference.py's predict_with_internals) so every intermediate
   tensor's actual numbers are visible. Pointing this at a different
   sample_index produces genuinely different numbers at every stage — the
   strongest evidence against "hardcoded" there is.

Both model/evaluate_model.py (offline script) and web/server.py's
/api/model-evidence endpoint import this module, so a script run and the live
page report numbers computed by the exact same code.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from model.baselines import run_baselines
from model.dataset import NpzSequenceDataset
from model.fusion import FusionLSTMClassifier
from model.train import history_path_for
from common.config import NUM_CHANNELS, BAND_NAMES, NUM_BEHAVIOR_FEATS
from common.states import STATES

DEFAULT_CHECKPOINT = Path("model/checkpoints/best.pt")
DEFAULT_DATA_DIR = Path("data/synthetic")


def load_training_history(checkpoint_path: Path = DEFAULT_CHECKPOINT) -> list[dict] | None:
    """Per-epoch train loss / val accuracy / val F1, if this checkpoint was
    trained with `model.train.train(..., save_history=True)` — None for an
    older checkpoint trained before that option existed, rather than raising."""
    history_file = history_path_for(checkpoint_path)
    if not history_file.exists():
        return None
    return json.loads(history_file.read_text())


def compute_baselines(data_dir: Path = DEFAULT_DATA_DIR) -> list[dict]:
    """Classical, non-neural baselines (majority-class, logistic regression
    on behavior alone) on the same held-out test split — see
    model/baselines.py for why this question matters independently of the
    ablation study's fused-vs-modality comparison."""
    return run_baselines(data_dir)


BEHAVIOR_FEATURE_NAMES = ["correct", "time_to_move", "eval_loss", "puzzle_rating"]


def compute_feature_importance(
    model: FusionLSTMClassifier, data_dir: Path = DEFAULT_DATA_DIR, split: str = "test", seed: int = 0,
) -> dict:
    """Permutation importance: shuffle one input dimension across the batch
    (breaking its real relationship with the label while leaving everything
    else untouched) and measure how much accuracy drops — a bigger drop
    means the model actually relies on that input, computed empirically by
    re-running the real model, not estimated or asserted. Covers both the
    4 behavior features and each of the 8 EEG channels (all 3 bands shuffled
    together per channel, since a channel is the natural unit here)."""
    dataset = NpzSequenceDataset(Path(data_dir) / f"{split}.npz")
    labels = dataset.y.numpy()
    rng = np.random.default_rng(seed)

    with torch.no_grad():
        baseline_preds = model(dataset.X_eeg, dataset.X_behavior).argmax(dim=-1).numpy()
    baseline_accuracy = float(accuracy_score(labels, baseline_preds))

    def importance_of(eeg: torch.Tensor, behavior: torch.Tensor) -> float:
        with torch.no_grad():
            preds = model(eeg, behavior).argmax(dim=-1).numpy()
        return baseline_accuracy - float(accuracy_score(labels, preds))

    behavior_importance = []
    for feat_idx, name in enumerate(BEHAVIOR_FEATURE_NAMES):
        shuffled = dataset.X_behavior.clone()
        perm = rng.permutation(shuffled.shape[0])
        shuffled[:, :, feat_idx] = shuffled[perm][:, :, feat_idx]
        behavior_importance.append({"feature": name, "importance": importance_of(dataset.X_eeg, shuffled)})

    eeg_importance = []
    for channel_idx in range(NUM_CHANNELS):
        shuffled = dataset.X_eeg.clone()
        perm = rng.permutation(shuffled.shape[0])
        shuffled[:, :, channel_idx, :] = shuffled[perm][:, :, channel_idx, :]
        eeg_importance.append({"channel": channel_idx, "importance": importance_of(shuffled, dataset.X_behavior)})

    return {
        "baseline_accuracy": baseline_accuracy,
        "behavior_importance": sorted(behavior_importance, key=lambda r: r["importance"], reverse=True),
        "eeg_channel_importance": sorted(eeg_importance, key=lambda r: r["importance"], reverse=True),
    }


def load_model(checkpoint_path: Path = DEFAULT_CHECKPOINT) -> FusionLSTMClassifier:
    model = FusionLSTMClassifier(
        num_channels=NUM_CHANNELS, num_bands=len(BAND_NAMES),
        num_behavior_feats=NUM_BEHAVIOR_FEATS, num_classes=len(STATES),
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def inspect_weights(model: FusionLSTMClassifier) -> list[dict]:
    report = []
    for name, param in model.named_parameters():
        values = param.detach().flatten()
        report.append({
            "name": name,
            "shape": list(param.shape),
            "num_params": values.numel(),
            "mean": float(values.mean()),
            "std": float(values.std()) if values.numel() > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
            # First 8 real values, flattened — not a summary statistic, the
            # actual learned floats, so a reader can see they're not all
            # zero/identical/round numbers a placeholder would have.
            "sample_values": [float(v) for v in values[:8]],
        })
    return report


def evaluate_metrics(model: FusionLSTMClassifier, data_dir: Path = DEFAULT_DATA_DIR, split: str = "test") -> dict:
    dataset = NpzSequenceDataset(Path(data_dir) / f"{split}.npz")
    labels = dataset.y.numpy()
    class_indices = list(range(len(STATES)))

    with torch.no_grad():
        logits = model(dataset.X_eeg, dataset.X_behavior)
        probs = torch.softmax(logits, dim=-1).numpy()
    preds = probs.argmax(axis=-1)

    report = classification_report(
        labels, preds, labels=class_indices, target_names=STATES, output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(labels, preds, labels=class_indices)

    # roc_auc_score needs at least 2 classes actually present in y_true — a
    # tiny/skewed split could fail this, so degrade to None rather than crash.
    try:
        roc_auc_macro = roc_auc_score(labels, probs, multi_class="ovr", average="macro", labels=class_indices)
        roc_auc_per_class = roc_auc_score(labels, probs, multi_class="ovr", average=None, labels=class_indices)
    except ValueError:
        roc_auc_macro = None
        roc_auc_per_class = [None] * len(STATES)

    per_class = {
        state: {
            "precision": report[state]["precision"],
            "recall": report[state]["recall"],
            "f1": report[state]["f1-score"],
            "support": int(report[state]["support"]),
            "roc_auc": float(roc_auc_per_class[i]) if roc_auc_per_class[i] is not None else None,
        }
        for i, state in enumerate(STATES)
    }

    return {
        "split": split,
        "num_samples": len(labels),
        "accuracy": report["accuracy"],
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "roc_auc_macro": float(roc_auc_macro) if roc_auc_macro is not None else None,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "state_order": STATES,
    }


def trace_forward_pass(
    model: FusionLSTMClassifier, data_dir: Path = DEFAULT_DATA_DIR, split: str = "test", sample_index: int = 0,
) -> dict:
    dataset = NpzSequenceDataset(Path(data_dir) / f"{split}.npz")
    if not 0 <= sample_index < len(dataset):
        raise ValueError(f"sample_index {sample_index} out of range for {split} split ({len(dataset)} samples)")

    eeg_seq = dataset.X_eeg[sample_index].unsqueeze(0)
    behavior_seq = dataset.X_behavior[sample_index].unsqueeze(0)
    true_label = STATES[int(dataset.y[sample_index])]

    with torch.no_grad():
        batch, seq_len, num_channels, _num_bands = eeg_seq.shape
        eeg_flat = eeg_seq.reshape(batch * seq_len, num_channels, -1)

        gcn1_out = model.eeg_encoder.gcn1(eeg_flat)
        gcn2_out = model.eeg_encoder.gcn2(gcn1_out)
        eeg_embed = gcn2_out.mean(dim=1).reshape(batch, seq_len, -1)

        behavior_linear1 = model.behavior_encoder.net[0]
        behavior_linear2 = model.behavior_encoder.net[2]
        behavior_hidden = torch.relu(behavior_linear1(behavior_seq))
        behavior_embed = behavior_linear2(behavior_hidden)

        fused = torch.cat([eeg_embed, behavior_embed], dim=-1)
        lstm_out, _ = model.lstm(fused)
        last_hidden = lstm_out[:, -1, :]
        logits = model.classifier(last_hidden)
        probs = torch.softmax(logits, dim=-1).squeeze(0)

        # Both GCN layers have their own learned adjacency — gcn1's shapes the
        # raw per-band input, gcn2's shapes gcn1's already-aggregated output.
        # Exposing both (not just gcn1's) lets the graph-structure visualization
        # show a real, sometimes surprising fact: gcn1's near-uniform adjacency
        # collapses every channel to nearly identical values, which zeroes
        # gcn2's gradient and leaves ITS adjacency at its zero-initialization.
        gcn1_adjacency = torch.softmax(model.eeg_encoder.gcn1.adjacency_logits, dim=-1)
        gcn2_adjacency = torch.softmax(model.eeg_encoder.gcn2.adjacency_logits, dim=-1)
        gcn1_last_step = gcn1_out.reshape(batch, seq_len, num_channels, -1)[0, -1]
        gcn2_last_step = gcn2_out.reshape(batch, seq_len, num_channels, -1)[0, -1]

    best_idx = int(torch.argmax(probs).item())

    return {
        "sample_index": sample_index,
        "split": split,
        "true_label": true_label,
        "predicted_label": STATES[best_idx],
        "correct": STATES[best_idx] == true_label,
        "input_eeg_last_timestep": eeg_seq[0, -1].tolist(),  # (channels, bands)
        "input_behavior_last_timestep": behavior_seq[0, -1].tolist(),
        "gcn1_adjacency_weights": gcn1_adjacency.tolist(),  # (channels, channels)
        "gcn2_adjacency_weights": gcn2_adjacency.tolist(),  # (channels, channels)
        "gcn1_output_last_timestep": gcn1_last_step.tolist(),  # (channels, hidden_dim)
        "gcn2_output_last_timestep": gcn2_last_step.tolist(),  # (channels, out_dim)
        "eeg_embedding": eeg_embed[0, -1].tolist(),
        "behavior_hidden": behavior_hidden[0, -1].tolist(),
        "behavior_embedding": behavior_embed[0, -1].tolist(),
        "fused_vector": fused[0, -1].tolist(),
        "lstm_hidden_state": last_hidden[0].tolist(),
        "logits": logits[0].tolist(),
        "softmax_probabilities": {state: float(probs[i]) for i, state in enumerate(STATES)},
    }
