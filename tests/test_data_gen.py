import numpy as np
from data_gen.virtual_player import VirtualPlayer
from data_gen.generate_dataset import generate_examples, behavior_to_vector, save_splits
from chess_task.base import BehaviorEvent
from common.states import STATES
from common.config import SEQ_LEN, NUM_CHANNELS, BAND_NAMES, NUM_BEHAVIOR_FEATS

def test_virtual_player_correctness_matches_state_tendency():
    focused = VirtualPlayer(target_state="Focused", seed=1)
    overloaded = VirtualPlayer(target_state="Overloaded", seed=1)
    focused_correct = sum(focused.attempt(f"p{i}", 1000).correct for i in range(200))
    overloaded_correct = sum(overloaded.attempt(f"p{i}", 1000).correct for i in range(200))
    assert focused_correct > overloaded_correct

def test_behavior_to_vector_range():
    event = BehaviorEvent(puzzle_id="p1", correct=True, time_to_move=100.0, eval_loss=1000.0, puzzle_rating=3000)
    vec = behavior_to_vector(event)
    assert vec.shape == (NUM_BEHAVIOR_FEATS,)
    assert np.all(vec >= 0.0) and np.all(vec <= 1.0)

def test_generate_examples_shapes():
    X_eeg, X_behavior, y = generate_examples(num_examples_per_state=3, seed=42)
    n = 3 * len(STATES)
    assert X_eeg.shape == (n, SEQ_LEN, NUM_CHANNELS, len(BAND_NAMES))
    assert X_behavior.shape == (n, SEQ_LEN, NUM_BEHAVIOR_FEATS)
    assert y.shape == (n,)
    assert set(np.unique(y).tolist()) == set(range(len(STATES)))

def test_save_splits_writes_npz(tmp_path):
    save_splits(tmp_path, num_examples_per_state=4, seed=0)
    for split in ["train", "val", "test"]:
        data = np.load(tmp_path / f"{split}.npz")
        assert "X_eeg" in data and "X_behavior" in data and "y" in data
