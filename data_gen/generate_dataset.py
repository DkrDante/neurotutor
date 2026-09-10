from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from common.config import NUM_CHANNELS, SAMPLE_RATE, EPOCH_SAMPLES, SEQ_LEN, NUM_BEHAVIOR_FEATS
from common.states import STATES
from eeg.simulated import SimulatedEEGSource
from preprocessing.filters import bandpass_filter
from preprocessing.features import extract_node_features
from data_gen.virtual_player import VirtualPlayer

def behavior_to_vector(event) -> np.ndarray:
    return np.array([
        1.0 if event.correct else 0.0,
        min(event.time_to_move / 30.0, 1.0),
        min(event.eval_loss / 500.0, 1.0),
        min(event.puzzle_rating / 2000.0, 1.0),
    ], dtype=np.float32)

def generate_examples(num_examples_per_state: int, seed: int = 0):
    X_eeg, X_behavior, y = [], [], []
    rng_seed = seed
    for state_idx, state in enumerate(STATES):
        for _ in range(num_examples_per_state):
            eeg_src = SimulatedEEGSource(
                num_channels=NUM_CHANNELS, sample_rate=SAMPLE_RATE,
                chunk_seconds=EPOCH_SAMPLES / SAMPLE_RATE, target_state=state, seed=rng_seed,
            )
            player = VirtualPlayer(target_state=state, seed=rng_seed)
            rating_rng = np.random.default_rng(rng_seed)
            rng_seed += 1

            seq_eeg, seq_behavior = [], []
            for step in range(SEQ_LEN):
                chunk = eeg_src.generate_chunk()
                # Must stay in lockstep with web.session.TutorSession.submit_move, which
                # applies the same bandpass before feature extraction at serving time.
                filtered = bandpass_filter(chunk.samples, SAMPLE_RATE)
                node_features = extract_node_features(filtered, SAMPLE_RATE)
                rating = int(rating_rng.integers(800, 1400))
                event = player.attempt(puzzle_id=f"synthetic-{step}", puzzle_rating=rating)
                seq_eeg.append(node_features)
                seq_behavior.append(behavior_to_vector(event))

            X_eeg.append(np.stack(seq_eeg))
            X_behavior.append(np.stack(seq_behavior))
            y.append(state_idx)

    return (
        np.array(X_eeg, dtype=np.float32),
        np.array(X_behavior, dtype=np.float32),
        np.array(y, dtype=np.int64),
    )

def save_splits(out_dir: Path, num_examples_per_state: int, seed: int = 0, train_frac: float = 0.7, val_frac: float = 0.15) -> None:
    X_eeg, X_behavior, y = generate_examples(num_examples_per_state, seed=seed)
    n = len(y)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)
    splits = {"train": idx[:train_end], "val": idx[train_end:val_end], "test": idx[val_end:]}

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_idx in splits.items():
        np.savez(
            out_dir / f"{split_name}.npz",
            X_eeg=X_eeg[split_idx], X_behavior=X_behavior[split_idx], y=y[split_idx],
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--examples-per-state", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    save_splits(args.out_dir, args.examples_per_state, seed=args.seed)
    print(f"Wrote synthetic dataset splits to {args.out_dir}")

if __name__ == "__main__":
    main()
