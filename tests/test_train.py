from pathlib import Path
import torch
from data_gen.generate_dataset import save_splits
from model.train import train

def test_train_reaches_reasonable_accuracy_on_synthetic_data(tmp_path: Path):
    torch.manual_seed(42)
    data_dir = tmp_path / "data"
    save_splits(data_dir, num_examples_per_state=100, seed=7)
    checkpoint_path = tmp_path / "checkpoints" / "best.pt"
    metrics = train(data_dir, checkpoint_path, epochs=15)
    assert checkpoint_path.exists()
    assert metrics["test_accuracy"] > 0.5
