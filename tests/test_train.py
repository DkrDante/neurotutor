import json
from pathlib import Path
import torch
from data_gen.generate_dataset import save_splits
from model.train import history_path_for, train

def test_train_reaches_reasonable_accuracy_on_synthetic_data(tmp_path: Path):
    torch.manual_seed(42)
    data_dir = tmp_path / "data"
    save_splits(data_dir, num_examples_per_state=100, seed=7)
    checkpoint_path = tmp_path / "checkpoints" / "best.pt"
    metrics = train(data_dir, checkpoint_path, epochs=15)
    assert checkpoint_path.exists()
    assert metrics["test_accuracy"] > 0.5

def test_train_returns_one_history_entry_per_epoch(tmp_path: Path):
    data_dir = tmp_path / "data"
    save_splits(data_dir, num_examples_per_state=20, seed=1)
    checkpoint_path = tmp_path / "checkpoints" / "best.pt"
    metrics = train(data_dir, checkpoint_path, epochs=5, seed=0)

    history = metrics["history"]
    assert len(history) == 5
    assert [h["epoch"] for h in history] == [1, 2, 3, 4, 5]
    for entry in history:
        assert entry["train_loss"] >= 0.0
        assert 0.0 <= entry["val_accuracy"] <= 1.0
        assert 0.0 <= entry["val_f1"] <= 1.0

def test_same_seed_produces_identical_training_history(tmp_path: Path):
    # The whole point of the seed parameter: two runs of the same command
    # must be reproducible, not just "close" — real evidence a checkpoint's
    # reported numbers can be regenerated exactly, not just claimed once.
    data_dir = tmp_path / "data"
    save_splits(data_dir, num_examples_per_state=20, seed=1)

    metrics_a = train(data_dir, tmp_path / "a.pt", epochs=3, seed=99)
    metrics_b = train(data_dir, tmp_path / "b.pt", epochs=3, seed=99)

    assert metrics_a["history"] == metrics_b["history"]
    assert metrics_a["test_accuracy"] == metrics_b["test_accuracy"]

def test_save_history_writes_a_json_file_next_to_the_checkpoint(tmp_path: Path):
    data_dir = tmp_path / "data"
    save_splits(data_dir, num_examples_per_state=20, seed=1)
    checkpoint_path = tmp_path / "checkpoints" / "best.pt"

    metrics = train(data_dir, checkpoint_path, epochs=3, seed=0, save_history=True)

    history_file = history_path_for(checkpoint_path)
    assert history_file.exists()
    assert json.loads(history_file.read_text()) == metrics["history"]

def test_history_not_saved_by_default(tmp_path: Path):
    data_dir = tmp_path / "data"
    save_splits(data_dir, num_examples_per_state=20, seed=1)
    checkpoint_path = tmp_path / "checkpoints" / "best.pt"

    train(data_dir, checkpoint_path, epochs=3, seed=0)

    assert not history_path_for(checkpoint_path).exists()
