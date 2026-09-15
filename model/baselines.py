"""Classical, non-neural baselines for the cognitive-state classifier —
model/ablation.py already proves the EEG branch does necessary work relative
to the model's OWN behavior-only variant, but that doesn't answer a more
basic question a research reviewer asks first: does this whole fusion
architecture beat something trivial? These two baselines answer that
directly, evaluated on the exact same held-out splits as the real model.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from model.dataset import NpzSequenceDataset


def _flatten_behavior(dataset: NpzSequenceDataset) -> np.ndarray:
    # (N, seq_len, num_behavior_feats) -> (N, seq_len * num_behavior_feats):
    # a classical model has no notion of "sequence", so give it every
    # timestep's behavior features concatenated rather than just the last.
    X = dataset.X_behavior.numpy()
    return X.reshape(X.shape[0], -1)


def majority_class_baseline(data_dir: Path) -> dict:
    """Always predicts whichever class was most common in the TRAINING
    split — the floor any real model must clear. A fused model scoring only
    a little above this would mean it learned almost nothing."""
    data_dir = Path(data_dir)
    train_ds = NpzSequenceDataset(data_dir / "train.npz")
    test_ds = NpzSequenceDataset(data_dir / "test.npz")

    train_labels = train_ds.y.numpy()
    majority_class = int(np.bincount(train_labels).argmax())

    test_labels = test_ds.y.numpy()
    predictions = np.full_like(test_labels, majority_class)

    return {
        "name": "majority_class",
        "test_accuracy": float(accuracy_score(test_labels, predictions)),
        "test_f1": float(f1_score(test_labels, predictions, average="macro", zero_division=0)),
    }


def logistic_regression_baseline(data_dir: Path, seed: int = 0) -> dict:
    """A standard classical-ML baseline (multinomial logistic regression) on
    the flattened behavior features alone — no EEG, no sequence modeling, no
    neural network. Trained and evaluated on the exact same splits as the
    real fused model, so its accuracy is directly comparable."""
    data_dir = Path(data_dir)
    train_ds = NpzSequenceDataset(data_dir / "train.npz")
    test_ds = NpzSequenceDataset(data_dir / "test.npz")

    X_train, y_train = _flatten_behavior(train_ds), train_ds.y.numpy()
    X_test, y_test = _flatten_behavior(test_ds), test_ds.y.numpy()

    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    return {
        "name": "logistic_regression_behavior_only",
        "test_accuracy": float(accuracy_score(y_test, predictions)),
        "test_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
    }


def run_baselines(data_dir: Path) -> list[dict]:
    return [majority_class_baseline(data_dir), logistic_regression_baseline(data_dir)]
