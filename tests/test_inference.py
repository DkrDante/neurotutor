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
