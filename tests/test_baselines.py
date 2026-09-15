from pathlib import Path
from data_gen.generate_dataset import save_splits
from model.baselines import logistic_regression_baseline, majority_class_baseline, run_baselines

def test_majority_class_baseline_never_beats_reasonable_accuracy(tmp_path: Path):
    data_dir = tmp_path / "data"
    save_splits(data_dir, num_examples_per_state=40, seed=3)
    result = majority_class_baseline(data_dir)
    assert result["name"] == "majority_class"
    # 5 balanced classes -> a majority-class guess should land near 1/5, and
    # can never legitimately exceed what the largest single class holds.
    assert 0.0 <= result["test_accuracy"] <= 0.5
    assert 0.0 <= result["test_f1"] <= 1.0

def test_logistic_regression_baseline_beats_majority_class(tmp_path: Path):
    data_dir = tmp_path / "data"
    save_splits(data_dir, num_examples_per_state=40, seed=3)
    majority = majority_class_baseline(data_dir)
    logistic = logistic_regression_baseline(data_dir, seed=0)
    assert logistic["name"] == "logistic_regression_behavior_only"
    # It has real behavior features to learn from — it should do
    # meaningfully better than always guessing the same class.
    assert logistic["test_accuracy"] > majority["test_accuracy"]

def test_logistic_regression_baseline_is_reproducible_with_a_fixed_seed(tmp_path: Path):
    data_dir = tmp_path / "data"
    save_splits(data_dir, num_examples_per_state=30, seed=5)
    first = logistic_regression_baseline(data_dir, seed=7)
    second = logistic_regression_baseline(data_dir, seed=7)
    assert first == second

def test_run_baselines_returns_both(tmp_path: Path):
    data_dir = tmp_path / "data"
    save_splits(data_dir, num_examples_per_state=20, seed=1)
    results = run_baselines(data_dir)
    names = {r["name"] for r in results}
    assert names == {"majority_class", "logistic_regression_behavior_only"}
