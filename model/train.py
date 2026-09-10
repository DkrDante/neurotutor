from __future__ import annotations
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from model.dataset import NpzSequenceDataset
from model.fusion import FusionLSTMClassifier
from common.config import NUM_CHANNELS, BAND_NAMES, NUM_BEHAVIOR_FEATS
from common.states import STATES

def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for eeg_seq, behavior_seq, labels in loader:
            logits = model(eeg_seq, behavior_seq)
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return accuracy, f1

def train(data_dir: Path, checkpoint_path: Path, epochs: int = 20, batch_size: int = 16, lr: float = 1e-3) -> dict:
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
    for _ in range(epochs):
        model.train()
        for eeg_seq, behavior_seq, labels in train_loader:
            optimizer.zero_grad()
            logits = model(eeg_seq, behavior_seq)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
        val_accuracy, _ = evaluate(model, val_loader)
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), checkpoint_path)

    test_ds = NpzSequenceDataset(data_dir / "test.npz")
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    model.load_state_dict(torch.load(checkpoint_path))
    test_accuracy, test_f1 = evaluate(model, test_loader)
    return {"best_val_accuracy": best_val_accuracy, "test_accuracy": test_accuracy, "test_f1": test_f1}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--checkpoint-path", type=Path, default=Path("model/checkpoints/best.pt"))
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    metrics = train(args.data_dir, args.checkpoint_path, epochs=args.epochs)
    print(metrics)

if __name__ == "__main__":
    main()
