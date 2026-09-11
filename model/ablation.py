"""Ablation study for the fusion classifier.

The synthetic data generator (data_gen/virtual_player.py, eeg/simulated.py) gives
each of the 5 cognitive states a distinct behavior profile AND a distinct EEG
profile, so a high fused-model accuracy alone doesn't prove the EEG/GCN branch is
contributing anything — the behavior features could be doing all the work. This
script trains three variants on the SAME dataset splits: the real fused model,
a behavior-only variant (EEG input zeroed out), and an EEG-only variant (behavior
input zeroed out), so their test accuracies can be compared directly.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from model.train import train

def mask_behavior_only(eeg_seq: torch.Tensor, behavior_seq: torch.Tensor):
    """Zeroes the EEG input, leaving only behavior features for the model to use."""
    return torch.zeros_like(eeg_seq), behavior_seq

def mask_eeg_only(eeg_seq: torch.Tensor, behavior_seq: torch.Tensor):
    """Zeroes the behavior input, leaving only EEG features for the model to use."""
    return eeg_seq, torch.zeros_like(behavior_seq)

VARIANTS = {
    "fused": None,
    "behavior_only": mask_behavior_only,
    "eeg_only": mask_eeg_only,
}

def run_ablation(data_dir: Path, checkpoint_dir: Path = Path("model/checkpoints"), epochs: int = 20) -> dict:
    data_dir = Path(data_dir)
    checkpoint_dir = Path(checkpoint_dir)
    results = {}
    for variant_name, mask_fn in VARIANTS.items():
        checkpoint_path = checkpoint_dir / f"ablation-{variant_name}.pt"
        results[variant_name] = train(data_dir, checkpoint_path, epochs=epochs, mask_fn=mask_fn)
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("model/checkpoints"))
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    results = run_ablation(args.data_dir, checkpoint_dir=args.checkpoint_dir, epochs=args.epochs)

    print(f"{'variant':<15}{'val_acc':>10}{'test_acc':>10}{'test_f1':>10}")
    for variant, metrics in results.items():
        print(
            f"{variant:<15}{metrics['best_val_accuracy']:>10.4f}"
            f"{metrics['test_accuracy']:>10.4f}{metrics['test_f1']:>10.4f}"
        )

if __name__ == "__main__":
    main()
