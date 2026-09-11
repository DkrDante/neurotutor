from pathlib import Path
import numpy as np
from data_gen.generate_dataset import save_splits
from model.train import train
from model.inference import StatePredictor
from common.config import NUM_CHANNELS, BAND_NAMES, NUM_BEHAVIOR_FEATS, SEQ_LEN
from common.states import STATES

def test_predict_round_trip(tmp_path: Path):
    data_dir = tmp_path / "data"
    save_splits(data_dir, num_examples_per_state=20, seed=3)
    checkpoint_path = tmp_path / "checkpoints" / "best.pt"
    train(data_dir, checkpoint_path, epochs=5)

    predictor = StatePredictor(checkpoint_path)
    eeg_seq = np.random.default_rng(0).normal(size=(SEQ_LEN, NUM_CHANNELS, len(BAND_NAMES))).astype(np.float32)
    behavior_seq = np.random.default_rng(0).random(size=(SEQ_LEN, NUM_BEHAVIOR_FEATS)).astype(np.float32)
    state, confidence, probs = predictor.predict(eeg_seq, behavior_seq)

    assert state in STATES
    assert 0.0 <= confidence <= 1.0
    assert abs(sum(probs.values()) - 1.0) < 1e-4
    assert set(probs.keys()) == set(STATES)

def test_predict_with_internals_shapes_and_ranges(tmp_path: Path):
    data_dir = tmp_path / "data"
    save_splits(data_dir, num_examples_per_state=20, seed=4)
    checkpoint_path = tmp_path / "checkpoints" / "best.pt"
    train(data_dir, checkpoint_path, epochs=5)

    predictor = StatePredictor(checkpoint_path)
    eeg_seq = np.random.default_rng(1).normal(size=(SEQ_LEN, NUM_CHANNELS, len(BAND_NAMES))).astype(np.float32)
    behavior_seq = np.random.default_rng(1).random(size=(SEQ_LEN, NUM_BEHAVIOR_FEATS)).astype(np.float32)
    result = predictor.predict_with_internals(eeg_seq, behavior_seq)

    assert result["state"] in STATES
    assert 0.0 <= result["confidence"] <= 1.0
    assert set(result["probs"].keys()) == set(STATES)

    assert len(result["gcn1_node_activity"]) == NUM_CHANNELS
    assert len(result["gcn2_node_activity"]) == NUM_CHANNELS
    assert all(v >= 0.0 for v in result["gcn1_node_activity"])  # vector norms are non-negative

    adjacency = result["gcn_adjacency"]
    assert len(adjacency) == NUM_CHANNELS
    assert all(len(row) == NUM_CHANNELS for row in adjacency)
    for row in adjacency:
        assert abs(sum(row) - 1.0) < 1e-4  # each row is a softmax over that node's neighbors

    assert result["eeg_embedding_norm"] >= 0.0
    assert result["behavior_embedding_norm"] >= 0.0
    assert result["lstm_hidden_norm"] >= 0.0
    assert len(result["behavior_activity"]) == NUM_BEHAVIOR_FEATS
    assert result["behavior_activity"] == [float(v) for v in behavior_seq[-1]]

def test_predict_delegates_to_predict_with_internals(tmp_path: Path):
    """predict() must stay a thin wrapper — same state/confidence/probs either way."""
    data_dir = tmp_path / "data"
    save_splits(data_dir, num_examples_per_state=20, seed=5)
    checkpoint_path = tmp_path / "checkpoints" / "best.pt"
    train(data_dir, checkpoint_path, epochs=5)

    predictor = StatePredictor(checkpoint_path)
    eeg_seq = np.random.default_rng(2).normal(size=(SEQ_LEN, NUM_CHANNELS, len(BAND_NAMES))).astype(np.float32)
    behavior_seq = np.random.default_rng(2).random(size=(SEQ_LEN, NUM_BEHAVIOR_FEATS)).astype(np.float32)

    state, confidence, probs = predictor.predict(eeg_seq, behavior_seq)
    result = predictor.predict_with_internals(eeg_seq, behavior_seq)

    assert state == result["state"]
    assert confidence == result["confidence"]
    assert probs == result["probs"]
