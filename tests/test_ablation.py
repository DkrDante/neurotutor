from pathlib import Path
import torch
from data_gen.generate_dataset import save_splits
from model.train import train
from model.ablation import VARIANTS, mask_behavior_only, mask_eeg_only, run_ablation

def test_mask_behavior_only_zeros_eeg_not_behavior():
    eeg_seq = torch.randn(2, 5, 8, 3)
    behavior_seq = torch.randn(2, 5, 4)
    masked_eeg, masked_behavior = mask_behavior_only(eeg_seq, behavior_seq)
    assert torch.all(masked_eeg == 0)
    assert torch.equal(masked_behavior, behavior_seq)

def test_mask_eeg_only_zeros_behavior_not_eeg():
    eeg_seq = torch.randn(2, 5, 8, 3)
    behavior_seq = torch.randn(2, 5, 4)
    masked_eeg, masked_behavior = mask_eeg_only(eeg_seq, behavior_seq)
    assert torch.equal(masked_eeg, eeg_seq)
    assert torch.all(masked_behavior == 0)

def test_train_accepts_mask_fn_and_still_produces_a_checkpoint(tmp_path: Path):
    data_dir = tmp_path / "data"
    save_splits(data_dir, num_examples_per_state=20, seed=1)
    checkpoint_path = tmp_path / "checkpoint.pt"
    metrics = train(data_dir, checkpoint_path, epochs=3, mask_fn=mask_eeg_only)
    assert checkpoint_path.exists()
    assert 0.0 <= metrics["test_accuracy"] <= 1.0

def test_run_ablation_produces_all_three_variants(tmp_path: Path):
    data_dir = tmp_path / "data"
    save_splits(data_dir, num_examples_per_state=20, seed=2)
    results = run_ablation(data_dir, checkpoint_dir=tmp_path / "checkpoints", epochs=3)
    assert set(results.keys()) == set(VARIANTS.keys())
    for metrics in results.values():
        assert 0.0 <= metrics["test_accuracy"] <= 1.0
        assert 0.0 <= metrics["test_f1"] <= 1.0
