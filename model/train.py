from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from model.dataset import NpzSequenceDataset
from model.fusion import FusionLSTMClassifier
from common.config import NUM_CHANNELS, BAND_NAMES, NUM_BEHAVIOR_FEATS
from common.states import STATES

def history_path_for(checkpoint_path: Path) -> Path:
    """Per-epoch training history lives next to its checkpoint — evidence.py's
    load_training_history() and /model-evidence read it from exactly here."""
    return Path(checkpoint_path).with_name(Path(checkpoint_path).name + ".history.json")

def evaluate(model, loader, mask_fn=None):
    """mask_fn, if given, is applied to (eeg_seq, behavior_seq) before the forward pass —
    used by model/ablation.py to zero out one modality's input for ablation runs."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for eeg_seq, behavior_seq, labels in loader:
            if mask_fn is not None:
                eeg_seq, behavior_seq = mask_fn(eeg_seq, behavior_seq)
            logits = model(eeg_seq, behavior_seq)
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return accuracy, f1

def train(
    data_dir: Path, checkpoint_path: Path, epochs: int = 20, batch_size: int = 16, lr: float = 1e-3,
    mask_fn=None, seed: int | None = None, save_history: bool = False,
) -> dict:
    # A fixed seed makes a training run reproducible — without one, two runs
    # of the same command produce genuinely different weights (different
    # init, different minibatch order), which is why the ablation table's
    # numbers could never be reproduced exactly before this. Left as None by
    # default so existing callers (model/ablation.py, tests) keep their
    # current non-deterministic behavior unless they explicitly opt in.
    if seed is not None:
        torch.manual_seed(seed)

    data_dir = Path(data_dir)
    checkpoint_path = Path(checkpoint_path)
    train_ds = NpzSequenceDataset(data_dir / "train.npz")
    val_ds = NpzSequenceDataset(data_dir / "val.npz")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = FusionLSTMClassifier(
        num_channels=NUM_CHANNELS, num_bands=len(BAND_NAMES),
        num_behavior_feats=NUM_BEHAVIOR_FEATS, num_classes=len(STATES),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_val_accuracy = -1.0
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(epochs):
        model.train()
        epoch_loss_total = 0.0
        num_batches = 0
        for eeg_seq, behavior_seq, labels in train_loader:
            if mask_fn is not None:
                eeg_seq, behavior_seq = mask_fn(eeg_seq, behavior_seq)
            optimizer.zero_grad()
            logits = model(eeg_seq, behavior_seq)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss_total += loss.item()
            num_batches += 1
        val_accuracy, val_f1 = evaluate(model, val_loader, mask_fn=mask_fn)
        history.append({
            "epoch": epoch + 1, "train_loss": epoch_loss_total / max(1, num_batches),
            "val_accuracy": val_accuracy, "val_f1": val_f1,
        })
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), checkpoint_path)

    if save_history:
        history_path_for(checkpoint_path).write_text(json.dumps(history, indent=2))

    test_ds = NpzSequenceDataset(data_dir / "test.npz")
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    test_accuracy, test_f1 = evaluate(model, test_loader, mask_fn=mask_fn)
    return {
        "best_val_accuracy": best_val_accuracy, "test_accuracy": test_accuracy, "test_f1": test_f1,
        "history": history,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--checkpoint-path", type=Path, default=Path("model/checkpoints/best.pt"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save-history", action="store_true")
    args = parser.parse_args()
    metrics = train(
        args.data_dir, args.checkpoint_path, epochs=args.epochs, seed=args.seed, save_history=args.save_history,
    )
    print({k: v for k, v in metrics.items() if k != "history"})

if __name__ == "__main__":
    main()
