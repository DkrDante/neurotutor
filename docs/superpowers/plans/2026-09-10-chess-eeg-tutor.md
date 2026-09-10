# NeuroTutor Chess Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a working, end-to-end chess tutoring web app that fuses a
simulated EEG signal with live chess-puzzle performance through a real
GCN+MLP+LSTM fusion classifier (trained on synthetic data) to predict a
5-class cognitive state and adapt puzzle difficulty/hints/pacing in real
time — with the EEG input abstracted behind an interface real hardware can
implement later without touching anything downstream.

**Architecture:** Single Python repo, modular monolith. A FastAPI +
WebSocket backend wires together: an `EEGSource` (simulator today), a
puzzle-based chess task engine (`python-chess` + Stockfish), a
preprocessing/feature-extraction stage, a PyTorch fusion model loaded from
a checkpoint, a rule-based adaptive policy, and SQLite session logging.
Model training is a separate offline concern (scripts producing a
checkpoint) from serving (the backend loads it read-only). A minimal
browser frontend renders the board and live state.

**Tech Stack:** Python 3.11, FastAPI, WebSockets, PyTorch (no PyTorch
Geometric — a small custom GCN layer is used instead, see Task 6),
python-chess, Stockfish (optional local binary), SQLite, NumPy/SciPy,
scikit-learn (metrics), vanilla JS/HTML/CSS frontend (no external JS
dependency).

**Spec:** `docs/superpowers/specs/2026-09-10-chess-eeg-tutor-design.md`

## Global Constraints

- 5 cognitive states, fixed order: `Focused, Overloaded, Confused, Fatigued, Engaged` (`common/states.py`).
- EEG config: 8 channels, 128 Hz sample rate, 4.0s epochs (512 samples), theta/alpha/beta bands (`common/config.py`). Every module reads these from `common/config.py` — never hardcode them again.
- Sequence length for the fusion model's temporal window: 5 attempts (`SEQ_LEN`).
- `EEGSource` is an abstract interface; `SimulatedEEGSource` is the only implementation in this plan. No real-hardware adapter ships here.
- Chess: puzzles only (`TaskEngine.get_puzzle`/`submit_move`); full-game mode is out of scope — do not implement it.
- Training data is synthetic only, generated in-repo; no external dataset download.
- No ONNX export, no TensorRT, no edge deployment in this plan.
- SQLite is the only storage; no other database.
- `pytest.ini` sets `pythonpath = .` — all imports are plain top-level package imports (`from common.states import STATES`), no `src/` layout, no package installation required.

---

## Task 1: Project Scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `pytest.ini`
- Create: `.gitignore`
- Create: `common/__init__.py`
- Create: `common/config.py`
- Create: `common/states.py`
- Create: `eeg/__init__.py`, `chess_task/__init__.py`, `preprocessing/__init__.py`, `data_gen/__init__.py`, `model/__init__.py`, `adaptive/__init__.py`, `storage/__init__.py`, `web/__init__.py`
- Create: `tests/__init__.py`
- Test: `tests/test_scaffolding.py`

**Interfaces:**
- Produces: `common.config.{NUM_CHANNELS, SAMPLE_RATE, EPOCH_SECONDS, EPOCH_SAMPLES, SEQ_LEN, BANDS, BAND_NAMES, NUM_BEHAVIOR_FEATS, ARTIFACT_THRESHOLD}`; `common.states.{STATES, STATE_TO_INDEX}`. Every later task imports from these two modules.

- [ ] **Step 1: Create `requirements.txt`**

```
fastapi==0.115.0
uvicorn[standard]==0.30.6
websockets==13.1
python-chess==1.11.2
torch==2.4.1
numpy==1.26.4
scipy==1.13.1
scikit-learn==1.5.1
pytest==8.3.2
pytest-asyncio==0.24.0
httpx==0.27.2
```

- [ ] **Step 2: Create `pytest.ini`**

```ini
[pytest]
pythonpath = .
testpaths = tests
```

- [ ] **Step 3: Create `.gitignore`**

```
__pycache__/
*.pyc
.venv/
venv/
data/
model/checkpoints/
*.db
.pytest_cache/
```

- [ ] **Step 4: Create `common/config.py`**

```python
NUM_CHANNELS = 8
SAMPLE_RATE = 128
EPOCH_SECONDS = 4.0
EPOCH_SAMPLES = int(SAMPLE_RATE * EPOCH_SECONDS)
SEQ_LEN = 5
BANDS = {"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}
BAND_NAMES = list(BANDS.keys())
NUM_BEHAVIOR_FEATS = 4
ARTIFACT_THRESHOLD = 8.0
```

- [ ] **Step 5: Create `common/states.py`**

```python
STATES = ["Focused", "Overloaded", "Confused", "Fatigued", "Engaged"]
STATE_TO_INDEX = {state: idx for idx, state in enumerate(STATES)}
```

- [ ] **Step 6: Create empty `__init__.py` files**

Create empty files at: `common/__init__.py`, `eeg/__init__.py`,
`chess_task/__init__.py`, `preprocessing/__init__.py`,
`data_gen/__init__.py`, `model/__init__.py`, `adaptive/__init__.py`,
`storage/__init__.py`, `web/__init__.py`, `tests/__init__.py`.

- [ ] **Step 7: Write a scaffolding smoke test**

```python
# tests/test_scaffolding.py
from common.config import NUM_CHANNELS, SAMPLE_RATE, SEQ_LEN
from common.states import STATES, STATE_TO_INDEX

def test_config_values_present():
    assert NUM_CHANNELS == 8
    assert SAMPLE_RATE == 128
    assert SEQ_LEN == 5

def test_states_indexed_correctly():
    assert STATES == ["Focused", "Overloaded", "Confused", "Fatigued", "Engaged"]
    assert STATE_TO_INDEX["Focused"] == 0
    assert STATE_TO_INDEX["Engaged"] == 4
```

- [ ] **Step 8: Install dependencies and run the test**

Run: `pip install -r requirements.txt && pytest tests/test_scaffolding.py -v`
Expected: 2 passed.

- [ ] **Step 9: Commit**

```bash
git add requirements.txt pytest.ini .gitignore common/ eeg/__init__.py chess_task/__init__.py preprocessing/__init__.py data_gen/__init__.py model/__init__.py adaptive/__init__.py storage/__init__.py web/__init__.py tests/
git commit -m "chore: scaffold project structure and shared config"
```

---

## Task 2: EEG Source Interface + Simulator

**Files:**
- Create: `eeg/source.py`
- Create: `eeg/simulated.py`
- Test: `tests/test_eeg_simulated.py`

**Interfaces:**
- Consumes: `common.config.{NUM_CHANNELS, SAMPLE_RATE}`, `common.states.STATES`.
- Produces: `eeg.source.EEGChunk(timestamp: float, samples: np.ndarray[shape=(chunk_samples, num_channels)])`; `eeg.source.EEGSource` (ABC with `sample_rate: int`, `num_channels: int`, `async def stream()`); `eeg.simulated.SimulatedEEGSource(num_channels=NUM_CHANNELS, sample_rate=SAMPLE_RATE, chunk_seconds=0.25, target_state="Focused", noise_amplitude=0.2, seed=None)` with methods `generate_chunk() -> EEGChunk` (sync, for tests/dataset gen) and `set_target_state(state: str) -> None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eeg_simulated.py
import numpy as np
import pytest
from scipy.signal import welch
from eeg.simulated import SimulatedEEGSource

def _band_power(samples_1d, sample_rate, band):
    freqs, psd = welch(samples_1d, fs=sample_rate, nperseg=min(256, len(samples_1d)))
    mask = (freqs >= band[0]) & (freqs <= band[1])
    return float(np.trapz(psd[mask], freqs[mask]))

def test_generate_chunk_shape():
    src = SimulatedEEGSource(num_channels=8, sample_rate=128, chunk_seconds=0.25, seed=1)
    chunk = src.generate_chunk()
    assert chunk.samples.shape == (32, 8)

def test_overloaded_has_more_theta_power_than_focused():
    focused = SimulatedEEGSource(target_state="Focused", chunk_seconds=4.0, seed=1)
    overloaded = SimulatedEEGSource(target_state="Overloaded", chunk_seconds=4.0, seed=1)
    f_chunk = focused.generate_chunk()
    o_chunk = overloaded.generate_chunk()
    f_power = np.mean([_band_power(f_chunk.samples[:, ch], 128, (4, 8)) for ch in range(8)])
    o_power = np.mean([_band_power(o_chunk.samples[:, ch], 128, (4, 8)) for ch in range(8)])
    assert o_power > f_power

def test_set_target_state_changes_state():
    src = SimulatedEEGSource(target_state="Focused", seed=1)
    src.set_target_state("Fatigued")
    assert src.target_state == "Fatigued"

def test_invalid_state_raises():
    with pytest.raises(ValueError):
        SimulatedEEGSource(target_state="NotAState")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_eeg_simulated.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eeg.simulated'`

- [ ] **Step 3: Write `eeg/source.py`**

```python
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator
import numpy as np

@dataclass
class EEGChunk:
    timestamp: float
    samples: np.ndarray  # shape (chunk_samples, num_channels)

class EEGSource(ABC):
    sample_rate: int
    num_channels: int

    @abstractmethod
    async def stream(self) -> AsyncIterator[EEGChunk]:
        """Yield EEGChunk objects indefinitely until the consumer stops iterating."""
        raise NotImplementedError
        yield  # pragma: no cover
```

- [ ] **Step 4: Write `eeg/simulated.py`**

```python
from __future__ import annotations
import asyncio
import time
from typing import AsyncIterator, Optional
import numpy as np
from eeg.source import EEGSource, EEGChunk
from common.config import NUM_CHANNELS, SAMPLE_RATE
from common.states import STATES

STATE_BAND_PARAMS = {
    "Focused":    {"theta": 0.3, "alpha": 0.5},
    "Overloaded": {"theta": 0.9, "alpha": 0.3},
    "Confused":   {"theta": 0.8, "alpha": 0.35},
    "Fatigued":   {"theta": 0.5, "alpha": 0.8},
    "Engaged":    {"theta": 0.4, "alpha": 0.4},
}
assert set(STATE_BAND_PARAMS) == set(STATES)

class SimulatedEEGSource(EEGSource):
    def __init__(
        self,
        num_channels: int = NUM_CHANNELS,
        sample_rate: int = SAMPLE_RATE,
        chunk_seconds: float = 0.25,
        target_state: str = "Focused",
        noise_amplitude: float = 0.2,
        seed: Optional[int] = None,
    ):
        if target_state not in STATE_BAND_PARAMS:
            raise ValueError(f"Unknown state: {target_state}")
        self.num_channels = num_channels
        self.sample_rate = sample_rate
        self.chunk_seconds = chunk_seconds
        self.chunk_samples = int(sample_rate * chunk_seconds)
        self.target_state = target_state
        self.noise_amplitude = noise_amplitude
        self._rng = np.random.default_rng(seed)
        self._t = 0.0

    def set_target_state(self, state: str) -> None:
        if state not in STATE_BAND_PARAMS:
            raise ValueError(f"Unknown state: {state}")
        self.target_state = state

    def generate_chunk(self) -> EEGChunk:
        params = STATE_BAND_PARAMS[self.target_state]
        t = self._t + np.arange(self.chunk_samples) / self.sample_rate
        theta = params["theta"] * np.sin(2 * np.pi * 6.0 * t)
        alpha = params["alpha"] * np.sin(2 * np.pi * 10.0 * t)
        base_signal = theta + alpha
        samples = np.empty((self.chunk_samples, self.num_channels), dtype=np.float64)
        for ch in range(self.num_channels):
            gain = 0.8 + 0.4 * self._rng.random()
            noise = self._rng.normal(0, self.noise_amplitude, size=self.chunk_samples)
            samples[:, ch] = gain * base_signal + noise
        self._t += self.chunk_seconds
        return EEGChunk(timestamp=time.monotonic(), samples=samples)

    async def stream(self) -> AsyncIterator[EEGChunk]:
        while True:
            yield self.generate_chunk()
            await asyncio.sleep(self.chunk_seconds)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_eeg_simulated.py -v`
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add eeg/source.py eeg/simulated.py tests/test_eeg_simulated.py
git commit -m "feat: add EEGSource interface and simulated EEG backend"
```

---

## Task 3: Preprocessing (Filtering, Epoching, Band-Power Features)

**Files:**
- Create: `preprocessing/filters.py`
- Create: `preprocessing/features.py`
- Create: `preprocessing/epoching.py`
- Test: `tests/test_preprocessing.py`

**Interfaces:**
- Consumes: `eeg.source.EEGChunk`; `common.config.{BANDS, BAND_NAMES, EPOCH_SECONDS, ARTIFACT_THRESHOLD}`.
- Produces: `preprocessing.filters.bandpass_filter(samples, sample_rate, low=1.0, high=40.0, order=4) -> np.ndarray`; `preprocessing.features.band_power(channel_samples, sample_rate, band) -> float` and `extract_node_features(epoch, sample_rate) -> np.ndarray[shape=(num_channels, len(BAND_NAMES))]`; `preprocessing.epoching.Epoch(samples, is_artifact)` and `EpochBuffer(num_channels, sample_rate, epoch_seconds=EPOCH_SECONDS, artifact_threshold=ARTIFACT_THRESHOLD)` with `add_chunk(chunk)` and `extract_epoch() -> Epoch`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_preprocessing.py
import numpy as np
from preprocessing.filters import bandpass_filter
from preprocessing.features import band_power, extract_node_features
from preprocessing.epoching import EpochBuffer
from eeg.source import EEGChunk
from common.config import BAND_NAMES

def test_bandpass_filter_attenuates_out_of_band():
    sample_rate = 128
    t = np.arange(sample_rate * 2) / sample_rate
    low_freq = np.sin(2 * np.pi * 0.2 * t)
    in_band = np.sin(2 * np.pi * 10 * t)
    signal = (low_freq + in_band).reshape(-1, 1)
    filtered = bandpass_filter(signal, sample_rate)
    assert np.std(filtered) < np.std(signal)

def test_band_power_higher_in_target_band():
    sample_rate = 128
    t = np.arange(sample_rate * 4) / sample_rate
    theta_signal = np.sin(2 * np.pi * 6 * t)
    gamma_signal = np.sin(2 * np.pi * 45 * t)
    theta_power = band_power(theta_signal, sample_rate, (4, 8))
    gamma_power_in_theta_band = band_power(gamma_signal, sample_rate, (4, 8))
    assert theta_power > gamma_power_in_theta_band

def test_extract_node_features_shape():
    sample_rate = 128
    epoch = np.random.default_rng(0).normal(size=(512, 8))
    features = extract_node_features(epoch, sample_rate)
    assert features.shape == (8, len(BAND_NAMES))

def test_epoch_buffer_pads_short_input():
    buf = EpochBuffer(num_channels=8, sample_rate=128, epoch_seconds=4.0)
    buf.add_chunk(EEGChunk(timestamp=0.0, samples=np.ones((32, 8))))
    epoch = buf.extract_epoch()
    assert epoch.samples.shape == (512, 8)
    assert np.all(epoch.samples[:480] == 0)
    assert np.all(epoch.samples[480:] == 1)

def test_epoch_buffer_truncates_long_input():
    buf = EpochBuffer(num_channels=8, sample_rate=128, epoch_seconds=1.0)
    long_input = np.arange(256 * 8).reshape(256, 8).astype(float)
    buf.add_chunk(EEGChunk(timestamp=0.0, samples=long_input))
    epoch = buf.extract_epoch()
    assert epoch.samples.shape == (128, 8)
    assert np.array_equal(epoch.samples, long_input[-128:])

def test_epoch_buffer_flags_artifact():
    buf = EpochBuffer(num_channels=8, sample_rate=128, epoch_seconds=1.0, artifact_threshold=5.0)
    samples = np.zeros((128, 8))
    samples[0, 0] = 100.0
    buf.add_chunk(EEGChunk(timestamp=0.0, samples=samples))
    epoch = buf.extract_epoch()
    assert epoch.is_artifact is True

def test_epoch_buffer_clears_after_extract():
    buf = EpochBuffer(num_channels=8, sample_rate=128, epoch_seconds=1.0)
    buf.add_chunk(EEGChunk(timestamp=0.0, samples=np.ones((128, 8))))
    buf.extract_epoch()
    second = buf.extract_epoch()
    assert np.all(second.samples == 0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_preprocessing.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'preprocessing.filters'`

- [ ] **Step 3: Write `preprocessing/filters.py`**

```python
import numpy as np
from scipy.signal import butter, filtfilt

def bandpass_filter(samples: np.ndarray, sample_rate: int, low: float = 1.0, high: float = 40.0, order: int = 4) -> np.ndarray:
    nyq = 0.5 * sample_rate
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, samples, axis=0)
```

- [ ] **Step 4: Write `preprocessing/features.py`**

```python
import numpy as np
from scipy.signal import welch
from common.config import BANDS, BAND_NAMES

def band_power(channel_samples: np.ndarray, sample_rate: int, band: tuple[float, float]) -> float:
    nperseg = min(256, len(channel_samples))
    freqs, psd = welch(channel_samples, fs=sample_rate, nperseg=nperseg)
    mask = (freqs >= band[0]) & (freqs <= band[1])
    return float(np.trapz(psd[mask], freqs[mask]))

def extract_node_features(epoch: np.ndarray, sample_rate: int) -> np.ndarray:
    """epoch: shape (num_samples, num_channels). Returns (num_channels, len(BAND_NAMES))."""
    num_channels = epoch.shape[1]
    features = np.zeros((num_channels, len(BAND_NAMES)), dtype=np.float64)
    for ch in range(num_channels):
        for b_idx, band_name in enumerate(BAND_NAMES):
            features[ch, b_idx] = band_power(epoch[:, ch], sample_rate, BANDS[band_name])
    return features
```

- [ ] **Step 5: Write `preprocessing/epoching.py`**

```python
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from eeg.source import EEGChunk
from common.config import EPOCH_SECONDS, ARTIFACT_THRESHOLD

@dataclass
class Epoch:
    samples: np.ndarray  # shape (num_samples, num_channels)
    is_artifact: bool

class EpochBuffer:
    def __init__(self, num_channels: int, sample_rate: int, epoch_seconds: float = EPOCH_SECONDS, artifact_threshold: float = ARTIFACT_THRESHOLD):
        self.num_channels = num_channels
        self.sample_rate = sample_rate
        self.epoch_samples = int(sample_rate * epoch_seconds)
        self.artifact_threshold = artifact_threshold
        self._buffer: list[np.ndarray] = []

    def add_chunk(self, chunk: EEGChunk) -> None:
        self._buffer.append(chunk.samples)

    def extract_epoch(self) -> Epoch:
        if self._buffer:
            all_samples = np.concatenate(self._buffer, axis=0)
        else:
            all_samples = np.zeros((0, self.num_channels))
        self._buffer = []

        if len(all_samples) >= self.epoch_samples:
            windowed = all_samples[-self.epoch_samples:]
        elif len(all_samples) > 0:
            pad = np.zeros((self.epoch_samples - len(all_samples), self.num_channels))
            windowed = np.concatenate([pad, all_samples], axis=0)
        else:
            windowed = np.zeros((self.epoch_samples, self.num_channels))

        is_artifact = bool(np.max(np.abs(windowed)) > self.artifact_threshold)
        return Epoch(samples=windowed, is_artifact=is_artifact)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_preprocessing.py -v`
Expected: 7 passed.

- [ ] **Step 7: Commit**

```bash
git add preprocessing/ tests/test_preprocessing.py
git commit -m "feat: add bandpass filtering, epoching, and band-power feature extraction"
```

---

## Task 4: Chess Task Engine (Puzzles)

**Files:**
- Create: `chess_task/base.py`
- Create: `chess_task/evaluator.py`
- Create: `chess_task/puzzles.py`
- Create: `chess_task/puzzle_data/sample_puzzles.csv`
- Test: `tests/test_chess_task.py`

**Interfaces:**
- Produces: `chess_task.base.Puzzle(puzzle_id, fen, solution_move, rating)`; `chess_task.base.BehaviorEvent(puzzle_id, correct, time_to_move, eval_loss, puzzle_rating)`; `chess_task.base.TaskEngine` (ABC: `get_puzzle(difficulty) -> Puzzle`, `submit_move(puzzle, move_uci, time_to_move) -> BehaviorEvent`); `chess_task.evaluator.MoveEvaluator` (ABC: `eval_loss(board, played_move, best_move) -> float`), `StockfishEvaluator(binary_path="stockfish", depth=10)`; `chess_task.puzzles.PuzzleTaskEngine(evaluator, puzzle_csv=DEFAULT_PUZZLE_CSV)`.

**Note (deviation from spec's "public Lichess puzzle dataset" wording):** this plan bundles a small, hand-built, engine-verified sample CSV (Step 1 below) instead of downloading the full public Lichess puzzle dataset, so the build stays offline and deterministic (no multi-GB download, no network dependency in tests). `PuzzleTaskEngine` reads any CSV with the same four columns (`puzzle_id,fen,solution_move,rating`), so swapping in the real Lichess dataset later is a matter of reformatting its CSV to these columns and pointing `puzzle_csv` at it — no code changes needed. Note also that a real Lichess puzzle's `solution_move` is the *second* move in its move list (the first move is the opponent's move that sets up the puzzle) — a reformatting step would need to account for that; this plan's puzzles are single-move-to-mate, sidestepping the issue entirely.

- [ ] **Step 1: Write `chess_task/puzzle_data/sample_puzzles.csv`**

These 10 positions are engine-verified mate-in-1 puzzles (validated with
`python-chess`: each `solution_move` is legal and produces checkmate).

```csv
puzzle_id,fen,solution_move,rating
rb01,2k5/1ppp4/8/8/8/8/8/R6K w - - 0 1,a1a8,700
rb02,3k4/2ppp3/8/8/8/8/8/R6K w - - 0 1,a1a8,830
rb03,4k3/3ppp2/8/8/8/8/8/R6K w - - 0 1,a1a8,960
rb04,5k2/4ppp1/8/8/8/8/8/R6K w - - 0 1,a1a8,1090
rb05,6k1/5ppp/8/8/8/8/8/R6K w - - 0 1,a1a8,1220
rb06,7k/5ppp/8/8/8/8/8/R6K w - - 0 1,a1a8,1350
qc01,7k/8/6QK/8/8/8/8/8 w - - 0 1,g6g7,1300
qc02,k7/8/KQ6/8/8/8/8/8 w - - 0 1,b6b7,1350
qc03,8/8/8/8/8/6QK/8/7k w - - 0 1,g3g2,1400
qc04,8/8/8/8/8/KQ6/8/k7 w - - 0 1,b3b2,1450
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_chess_task.py
import chess
import pytest
from chess_task.base import Puzzle, BehaviorEvent
from chess_task.evaluator import MoveEvaluator
from chess_task.puzzles import PuzzleTaskEngine

class FakeEvaluator(MoveEvaluator):
    def eval_loss(self, board, played_move, best_move) -> float:
        return 250.0

def test_load_puzzles_from_csv():
    engine = PuzzleTaskEngine(evaluator=FakeEvaluator())
    assert len(engine.puzzles) == 10

def test_get_puzzle_picks_nearest_rating():
    engine = PuzzleTaskEngine(evaluator=FakeEvaluator())
    puzzle = engine.get_puzzle(difficulty=1000)
    assert puzzle.puzzle_id == "rb03"  # rating 960, closest to 1000

def test_get_puzzle_does_not_repeat_until_exhausted():
    engine = PuzzleTaskEngine(evaluator=FakeEvaluator())
    seen = {engine.get_puzzle(1000).puzzle_id for _ in range(10)}
    assert len(seen) == 10

def test_submit_move_correct():
    engine = PuzzleTaskEngine(evaluator=FakeEvaluator())
    puzzle = Puzzle(puzzle_id="rb01", fen="2k5/1ppp4/8/8/8/8/8/R6K w - - 0 1", solution_move="a1a8", rating=700)
    event = engine.submit_move(puzzle, "a1a8", time_to_move=3.0)
    assert isinstance(event, BehaviorEvent)
    assert event.correct is True
    assert event.eval_loss == 0.0

def test_submit_move_incorrect_uses_evaluator():
    engine = PuzzleTaskEngine(evaluator=FakeEvaluator())
    puzzle = Puzzle(puzzle_id="rb01", fen="2k5/1ppp4/8/8/8/8/8/R6K w - - 0 1", solution_move="a1a8", rating=700)
    event = engine.submit_move(puzzle, "h1g1", time_to_move=3.0)
    assert event.correct is False
    assert event.eval_loss == 250.0
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_chess_task.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'chess_task.base'`

- [ ] **Step 4: Write `chess_task/base.py`**

```python
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class Puzzle:
    puzzle_id: str
    fen: str
    solution_move: str  # UCI, e.g. "e2e4"
    rating: int

@dataclass
class BehaviorEvent:
    puzzle_id: str
    correct: bool
    time_to_move: float
    eval_loss: float
    puzzle_rating: int

class TaskEngine(ABC):
    @abstractmethod
    def get_puzzle(self, difficulty: float) -> Puzzle:
        raise NotImplementedError

    @abstractmethod
    def submit_move(self, puzzle: Puzzle, move_uci: str, time_to_move: float) -> BehaviorEvent:
        raise NotImplementedError
```

- [ ] **Step 5: Write `chess_task/evaluator.py`**

```python
from __future__ import annotations
from abc import ABC, abstractmethod
import shutil
import chess
import chess.engine

class MoveEvaluator(ABC):
    @abstractmethod
    def eval_loss(self, board: chess.Board, played_move: chess.Move, best_move: chess.Move) -> float:
        raise NotImplementedError

class StockfishEvaluator(MoveEvaluator):
    def __init__(self, binary_path: str = "stockfish", depth: int = 10):
        if shutil.which(binary_path) is None:
            raise FileNotFoundError(f"Stockfish binary not found: {binary_path}")
        self.binary_path = binary_path
        self.depth = depth

    def _score(self, board: chess.Board, move: chess.Move) -> int:
        board = board.copy()
        board.push(move)
        with chess.engine.SimpleEngine.popen_uci(self.binary_path) as engine:
            info = engine.analyse(board, chess.engine.Limit(depth=self.depth))
            score = info["score"].pov(not board.turn)
            return score.score(mate_score=10000)

    def eval_loss(self, board: chess.Board, played_move: chess.Move, best_move: chess.Move) -> float:
        best_score = self._score(board, best_move)
        played_score = self._score(board, played_move)
        return float(max(0, best_score - played_score))
```

- [ ] **Step 6: Write `chess_task/puzzles.py`**

```python
from __future__ import annotations
import csv
from pathlib import Path
import chess
from chess_task.base import TaskEngine, Puzzle, BehaviorEvent
from chess_task.evaluator import MoveEvaluator

DEFAULT_PUZZLE_CSV = Path(__file__).parent / "puzzle_data" / "sample_puzzles.csv"

class PuzzleTaskEngine(TaskEngine):
    def __init__(self, evaluator: MoveEvaluator, puzzle_csv: Path = DEFAULT_PUZZLE_CSV):
        self.evaluator = evaluator
        self.puzzles = self._load_puzzles(puzzle_csv)
        self._served: set[str] = set()

    @staticmethod
    def _load_puzzles(path: Path) -> list[Puzzle]:
        puzzles = []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                puzzles.append(Puzzle(
                    puzzle_id=row["puzzle_id"],
                    fen=row["fen"],
                    solution_move=row["solution_move"],
                    rating=int(row["rating"]),
                ))
        return puzzles

    def get_puzzle(self, difficulty: float) -> Puzzle:
        available = [p for p in self.puzzles if p.puzzle_id not in self._served]
        if not available:
            self._served.clear()
            available = self.puzzles
        best = min(available, key=lambda p: abs(p.rating - difficulty))
        self._served.add(best.puzzle_id)
        return best

    def submit_move(self, puzzle: Puzzle, move_uci: str, time_to_move: float) -> BehaviorEvent:
        board = chess.Board(puzzle.fen)
        played_move = chess.Move.from_uci(move_uci)
        best_move = chess.Move.from_uci(puzzle.solution_move)
        correct = played_move == best_move
        eval_loss = 0.0 if correct else self.evaluator.eval_loss(board, played_move, best_move)
        return BehaviorEvent(
            puzzle_id=puzzle.puzzle_id,
            correct=correct,
            time_to_move=time_to_move,
            eval_loss=eval_loss,
            puzzle_rating=puzzle.rating,
        )
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `pytest tests/test_chess_task.py -v`
Expected: 5 passed.

- [ ] **Step 8: Commit**

```bash
git add chess_task/ tests/test_chess_task.py
git commit -m "feat: add puzzle-based chess task engine with pluggable move evaluator"
```

---

## Task 5: Synthetic Dataset Generator

**Files:**
- Create: `data_gen/virtual_player.py`
- Create: `data_gen/generate_dataset.py`
- Test: `tests/test_data_gen.py`

**Interfaces:**
- Consumes: `chess_task.base.BehaviorEvent`; `eeg.simulated.SimulatedEEGSource`; `preprocessing.features.extract_node_features`; `common.config.{NUM_CHANNELS, SAMPLE_RATE, EPOCH_SAMPLES, SEQ_LEN, BAND_NAMES, NUM_BEHAVIOR_FEATS}`; `common.states.STATES`.
- Produces: `data_gen.virtual_player.VirtualPlayer(target_state, seed=None)` with `attempt(puzzle_id, puzzle_rating) -> BehaviorEvent`; `data_gen.generate_dataset.behavior_to_vector(event) -> np.ndarray[shape=(NUM_BEHAVIOR_FEATS,)]`; `generate_examples(num_examples_per_state, seed=0) -> (X_eeg, X_behavior, y)`; `save_splits(out_dir, num_examples_per_state, seed=0, train_frac=0.7, val_frac=0.15)` writing `train.npz`/`val.npz`/`test.npz` each with arrays `X_eeg`, `X_behavior`, `y`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_data_gen.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_data_gen.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'data_gen.virtual_player'`

- [ ] **Step 3: Write `data_gen/virtual_player.py`**

```python
from __future__ import annotations
import numpy as np
from chess_task.base import BehaviorEvent
from common.states import STATES

STATE_BEHAVIOR_PARAMS = {
    "Focused":    {"correct_prob": 0.85, "time_mean": 6.0, "time_std": 1.5, "eval_loss_mean": 80.0},
    "Overloaded": {"correct_prob": 0.35, "time_mean": 14.0, "time_std": 4.0, "eval_loss_mean": 300.0},
    "Confused":   {"correct_prob": 0.40, "time_mean": 12.0, "time_std": 3.5, "eval_loss_mean": 260.0},
    "Fatigued":   {"correct_prob": 0.55, "time_mean": 16.0, "time_std": 5.0, "eval_loss_mean": 180.0},
    "Engaged":    {"correct_prob": 0.75, "time_mean": 7.5, "time_std": 2.0, "eval_loss_mean": 100.0},
}
assert set(STATE_BEHAVIOR_PARAMS) == set(STATES)

class VirtualPlayer:
    def __init__(self, target_state: str, seed: int | None = None):
        if target_state not in STATE_BEHAVIOR_PARAMS:
            raise ValueError(f"Unknown state: {target_state}")
        self.target_state = target_state
        self._rng = np.random.default_rng(seed)

    def attempt(self, puzzle_id: str, puzzle_rating: int) -> BehaviorEvent:
        params = STATE_BEHAVIOR_PARAMS[self.target_state]
        correct = bool(self._rng.random() < params["correct_prob"])
        time_to_move = float(max(0.5, self._rng.normal(params["time_mean"], params["time_std"])))
        eval_loss = 0.0
        if not correct:
            eval_loss = float(max(0.0, self._rng.normal(params["eval_loss_mean"], params["eval_loss_mean"] * 0.3)))
        return BehaviorEvent(
            puzzle_id=puzzle_id, correct=correct, time_to_move=time_to_move,
            eval_loss=eval_loss, puzzle_rating=puzzle_rating,
        )
```

- [ ] **Step 4: Write `data_gen/generate_dataset.py`**

```python
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from common.config import NUM_CHANNELS, SAMPLE_RATE, EPOCH_SAMPLES, SEQ_LEN, NUM_BEHAVIOR_FEATS
from common.states import STATES
from eeg.simulated import SimulatedEEGSource
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
                node_features = extract_node_features(chunk.samples, SAMPLE_RATE)
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_data_gen.py -v`
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add data_gen/ tests/test_data_gen.py
git commit -m "feat: add virtual player and synthetic EEG+behavior dataset generator"
```

---

## Task 6: Fusion Model Architecture

**Files:**
- Create: `model/graph.py`
- Create: `model/encoders.py`
- Create: `model/fusion.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: `common.config.{NUM_CHANNELS, BAND_NAMES, NUM_BEHAVIOR_FEATS, SEQ_LEN}`, `common.states.STATES`.
- Produces: `model.graph.GCNLayer(in_dim, out_dim, num_nodes)`; `model.encoders.EEGGCNEncoder(num_nodes, in_dim, hidden_dim=16, out_dim=16)`, `BehaviorMLPEncoder(in_dim, hidden_dim=16, out_dim=16)`; `model.fusion.FusionLSTMClassifier(num_channels, num_bands, num_behavior_feats, num_classes, eeg_embed_dim=16, behavior_embed_dim=16, lstm_hidden_dim=32)` — `forward(eeg_seq: (batch, seq_len, num_channels, num_bands), behavior_seq: (batch, seq_len, num_behavior_feats)) -> logits (batch, num_classes)`.

**Note (deviation from spec's parenthetical):** the spec mentions "GCN via PyTorch Geometric"; this plan implements a small custom GCN layer in plain PyTorch instead (Step 3 below), to avoid PyTorch Geometric's platform-specific wheel installation risk for an 8-node fixed graph. Behavior and architecture (GCN-encoded EEG nodes + MLP-encoded behavior, fused via LSTM) are unchanged.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_model.py
import torch
from model.fusion import FusionLSTMClassifier
from common.config import NUM_CHANNELS, BAND_NAMES, NUM_BEHAVIOR_FEATS, SEQ_LEN
from common.states import STATES

def _make_model():
    return FusionLSTMClassifier(
        num_channels=NUM_CHANNELS, num_bands=len(BAND_NAMES),
        num_behavior_feats=NUM_BEHAVIOR_FEATS, num_classes=len(STATES),
    )

def test_forward_output_shape():
    model = _make_model()
    eeg_seq = torch.randn(4, SEQ_LEN, NUM_CHANNELS, len(BAND_NAMES))
    behavior_seq = torch.randn(4, SEQ_LEN, NUM_BEHAVIOR_FEATS)
    logits = model(eeg_seq, behavior_seq)
    assert logits.shape == (4, len(STATES))

def test_model_can_overfit_single_batch():
    torch.manual_seed(0)
    model = _make_model()
    eeg_seq = torch.randn(8, SEQ_LEN, NUM_CHANNELS, len(BAND_NAMES))
    behavior_seq = torch.randn(8, SEQ_LEN, NUM_BEHAVIOR_FEATS)
    labels = torch.randint(0, len(STATES), (8,))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()
    losses = []
    for _ in range(50):
        optimizer.zero_grad()
        logits = model(eeg_seq, behavior_seq)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'model.fusion'`

- [ ] **Step 3: Write `model/graph.py`**

```python
from __future__ import annotations
import torch
import torch.nn as nn

class GCNLayer(nn.Module):
    """Minimal graph-conv layer: softmax-normalized learnable adjacency over
    a fixed, small node set (EEG channels), then a linear + ReLU projection.
    Avoids a PyTorch Geometric dependency for this small, fixed channel graph.
    """
    def __init__(self, in_dim: int, out_dim: int, num_nodes: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.adjacency_logits = nn.Parameter(torch.zeros(num_nodes, num_nodes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, num_nodes, in_dim)
        adjacency = torch.softmax(self.adjacency_logits, dim=-1)  # (num_nodes, num_nodes)
        aggregated = torch.einsum("ij,bjf->bif", adjacency, x)
        return torch.relu(self.linear(aggregated))
```

- [ ] **Step 4: Write `model/encoders.py`**

```python
from __future__ import annotations
import torch
import torch.nn as nn
from model.graph import GCNLayer

class EEGGCNEncoder(nn.Module):
    def __init__(self, num_nodes: int, in_dim: int, hidden_dim: int = 16, out_dim: int = 16):
        super().__init__()
        self.gcn1 = GCNLayer(in_dim, hidden_dim, num_nodes)
        self.gcn2 = GCNLayer(hidden_dim, out_dim, num_nodes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, num_nodes, in_dim) -> (batch, out_dim)
        h = self.gcn1(x)
        h = self.gcn2(h)
        return h.mean(dim=1)

class BehaviorMLPEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 16, out_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
```

- [ ] **Step 5: Write `model/fusion.py`**

```python
from __future__ import annotations
import torch
import torch.nn as nn
from model.encoders import EEGGCNEncoder, BehaviorMLPEncoder

class FusionLSTMClassifier(nn.Module):
    def __init__(
        self, num_channels: int, num_bands: int, num_behavior_feats: int, num_classes: int,
        eeg_embed_dim: int = 16, behavior_embed_dim: int = 16, lstm_hidden_dim: int = 32,
    ):
        super().__init__()
        self.eeg_encoder = EEGGCNEncoder(num_channels, num_bands, out_dim=eeg_embed_dim)
        self.behavior_encoder = BehaviorMLPEncoder(num_behavior_feats, out_dim=behavior_embed_dim)
        self.lstm = nn.LSTM(input_size=eeg_embed_dim + behavior_embed_dim, hidden_size=lstm_hidden_dim, batch_first=True)
        self.classifier = nn.Linear(lstm_hidden_dim, num_classes)

    def forward(self, eeg_seq: torch.Tensor, behavior_seq: torch.Tensor) -> torch.Tensor:
        batch, seq_len, num_channels, num_bands = eeg_seq.shape
        eeg_flat = eeg_seq.reshape(batch * seq_len, num_channels, num_bands)
        eeg_embed = self.eeg_encoder(eeg_flat).reshape(batch, seq_len, -1)
        behavior_embed = self.behavior_encoder(behavior_seq)
        fused = torch.cat([eeg_embed, behavior_embed], dim=-1)
        lstm_out, _ = self.lstm(fused)
        last_hidden = lstm_out[:, -1, :]
        return self.classifier(last_hidden)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_model.py -v`
Expected: 2 passed.

- [ ] **Step 7: Commit**

```bash
git add model/graph.py model/encoders.py model/fusion.py tests/test_model.py
git commit -m "feat: add GCN+MLP+LSTM fusion classifier architecture"
```

---

## Task 7: Training Script

**Files:**
- Create: `model/dataset.py`
- Create: `model/train.py`
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: `model.fusion.FusionLSTMClassifier`; `data_gen.generate_dataset.save_splits`; `common.config.{NUM_CHANNELS, BAND_NAMES, NUM_BEHAVIOR_FEATS}`; `common.states.STATES`.
- Produces: `model.dataset.NpzSequenceDataset(npz_path)` (a `torch.utils.data.Dataset` yielding `(eeg_seq, behavior_seq, label)`); `model.train.evaluate(model, loader) -> (accuracy, f1)`; `model.train.train(data_dir, checkpoint_path, epochs=20, batch_size=16, lr=1e-3) -> dict` with keys `best_val_accuracy`, `test_accuracy`, `test_f1`, writing the best-val checkpoint to `checkpoint_path`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_train.py
from pathlib import Path
from data_gen.generate_dataset import save_splits
from model.train import train

def test_train_reaches_reasonable_accuracy_on_synthetic_data(tmp_path: Path):
    data_dir = tmp_path / "data"
    save_splits(data_dir, num_examples_per_state=40, seed=7)
    checkpoint_path = tmp_path / "checkpoints" / "best.pt"
    metrics = train(data_dir, checkpoint_path, epochs=15)
    assert checkpoint_path.exists()
    assert metrics["test_accuracy"] > 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_train.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'model.dataset'`

- [ ] **Step 3: Write `model/dataset.py`**

```python
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

class NpzSequenceDataset(Dataset):
    def __init__(self, npz_path: Path):
        data = np.load(npz_path)
        self.X_eeg = torch.tensor(data["X_eeg"], dtype=torch.float32)
        self.X_behavior = torch.tensor(data["X_behavior"], dtype=torch.float32)
        self.y = torch.tensor(data["y"], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X_eeg[idx], self.X_behavior[idx], self.y[idx]
```

- [ ] **Step 4: Write `model/train.py`**

```python
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from model.dataset import NpzSequenceDataset
from model.fusion import FusionLSTMClassifier
from common.config import NUM_CHANNELS, BAND_NAMES, NUM_BEHAVIOR_FEATS
from common.states import STATES

def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for eeg_seq, behavior_seq, labels in loader:
            logits = model(eeg_seq, behavior_seq)
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return accuracy, f1

def train(data_dir: Path, checkpoint_path: Path, epochs: int = 20, batch_size: int = 16, lr: float = 1e-3) -> dict:
    data_dir = Path(data_dir)
    checkpoint_path = Path(checkpoint_path)
    train_ds = NpzSequenceDataset(data_dir / "train.npz")
    val_ds = NpzSequenceDataset(data_dir / "val.npz")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = FusionLSTMClassifier(
        num_channels=NUM_CHANNELS, num_bands=len(BAND_NAMES),
        num_behavior_feats=NUM_BEHAVIOR_FEATS, num_classes=len(STATES),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_val_accuracy = -1.0
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(epochs):
        model.train()
        for eeg_seq, behavior_seq, labels in train_loader:
            optimizer.zero_grad()
            logits = model(eeg_seq, behavior_seq)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
        val_accuracy, _ = evaluate(model, val_loader)
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), checkpoint_path)

    test_ds = NpzSequenceDataset(data_dir / "test.npz")
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    model.load_state_dict(torch.load(checkpoint_path))
    test_accuracy, test_f1 = evaluate(model, test_loader)
    return {"best_val_accuracy": best_val_accuracy, "test_accuracy": test_accuracy, "test_f1": test_f1}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--checkpoint-path", type=Path, default=Path("model/checkpoints/best.pt"))
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    metrics = train(args.data_dir, args.checkpoint_path, epochs=args.epochs)
    print(metrics)

if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_train.py -v`
Expected: 1 passed (may take up to ~30s on CPU).

- [ ] **Step 6: Commit**

```bash
git add model/dataset.py model/train.py tests/test_train.py
git commit -m "feat: add training loop for the fusion classifier on synthetic data"
```

---

## Task 8: Inference Wrapper

**Files:**
- Create: `model/inference.py`
- Test: `tests/test_inference.py`

**Interfaces:**
- Consumes: `model.fusion.FusionLSTMClassifier`; `common.config.{NUM_CHANNELS, BAND_NAMES, NUM_BEHAVIOR_FEATS}`; `common.states.STATES`.
- Produces: `model.inference.StatePredictor(checkpoint_path)` with `predict(eeg_seq: np.ndarray[shape=(SEQ_LEN, NUM_CHANNELS, len(BAND_NAMES))], behavior_seq: np.ndarray[shape=(SEQ_LEN, NUM_BEHAVIOR_FEATS)]) -> (state: str, confidence: float, probs: dict[str, float])`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_inference.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_inference.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'model.inference'`

- [ ] **Step 3: Write `model/inference.py`**

```python
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from model.fusion import FusionLSTMClassifier
from common.config import NUM_CHANNELS, BAND_NAMES, NUM_BEHAVIOR_FEATS
from common.states import STATES

class StatePredictor:
    def __init__(self, checkpoint_path: Path):
        self.model = FusionLSTMClassifier(
            num_channels=NUM_CHANNELS, num_bands=len(BAND_NAMES),
            num_behavior_feats=NUM_BEHAVIOR_FEATS, num_classes=len(STATES),
        )
        self.model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        self.model.eval()

    def predict(self, eeg_seq: np.ndarray, behavior_seq: np.ndarray):
        eeg_tensor = torch.tensor(eeg_seq, dtype=torch.float32).unsqueeze(0)
        behavior_tensor = torch.tensor(behavior_seq, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(eeg_tensor, behavior_tensor)
            probs = torch.softmax(logits, dim=-1).squeeze(0)
        best_idx = int(torch.argmax(probs).item())
        prob_dict = {state: float(probs[i]) for i, state in enumerate(STATES)}
        return STATES[best_idx], float(probs[best_idx]), prob_dict
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_inference.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add model/inference.py tests/test_inference.py
git commit -m "feat: add checkpoint-loading inference wrapper for the fusion model"
```

---

## Task 9: Adaptive Decision Engine

**Files:**
- Create: `adaptive/policy.py`
- Create: `adaptive/rule_based.py`
- Test: `tests/test_adaptive.py`

**Interfaces:**
- Consumes: `common.states.STATES`.
- Produces: `adaptive.policy.Action(difficulty_delta: float, show_hint: bool, pacing_delay: float)`; `adaptive.policy.Policy` (ABC: `decide(state, confidence) -> Action`); `adaptive.rule_based.RuleBasedPolicy`, `CONFIDENCE_THRESHOLD`, `NO_OP_ACTION`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_adaptive.py
import pytest
from adaptive.rule_based import RuleBasedPolicy, CONFIDENCE_THRESHOLD

def test_overloaded_lowers_difficulty_and_shows_hint():
    policy = RuleBasedPolicy()
    action = policy.decide("Overloaded", confidence=0.9)
    assert action.difficulty_delta < 0
    assert action.show_hint is True

def test_focused_raises_difficulty():
    policy = RuleBasedPolicy()
    action = policy.decide("Focused", confidence=0.9)
    assert action.difficulty_delta > 0

def test_fatigued_increases_pacing_delay():
    policy = RuleBasedPolicy()
    action = policy.decide("Fatigued", confidence=0.9)
    assert action.pacing_delay > 0

def test_low_confidence_returns_no_op():
    policy = RuleBasedPolicy()
    action = policy.decide("Overloaded", confidence=CONFIDENCE_THRESHOLD - 0.01)
    assert action.difficulty_delta == 0.0
    assert action.show_hint is False

def test_unknown_state_raises():
    policy = RuleBasedPolicy()
    with pytest.raises(ValueError):
        policy.decide("NotAState", confidence=0.9)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_adaptive.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'adaptive.rule_based'`

- [ ] **Step 3: Write `adaptive/policy.py`**

```python
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class Action:
    difficulty_delta: float
    show_hint: bool
    pacing_delay: float

class Policy(ABC):
    @abstractmethod
    def decide(self, state: str, confidence: float) -> Action:
        raise NotImplementedError
```

- [ ] **Step 4: Write `adaptive/rule_based.py`**

```python
from __future__ import annotations
from adaptive.policy import Policy, Action
from common.states import STATES

CONFIDENCE_THRESHOLD = 0.4

STATE_RULES = {
    "Focused":    Action(difficulty_delta=100.0, show_hint=False, pacing_delay=0.0),
    "Engaged":    Action(difficulty_delta=75.0, show_hint=False, pacing_delay=0.0),
    "Overloaded": Action(difficulty_delta=-150.0, show_hint=True, pacing_delay=3.0),
    "Confused":   Action(difficulty_delta=-100.0, show_hint=True, pacing_delay=2.0),
    "Fatigued":   Action(difficulty_delta=-50.0, show_hint=False, pacing_delay=5.0),
}
assert set(STATE_RULES) == set(STATES)
NO_OP_ACTION = Action(difficulty_delta=0.0, show_hint=False, pacing_delay=0.0)

class RuleBasedPolicy(Policy):
    def decide(self, state: str, confidence: float) -> Action:
        if state not in STATES:
            raise ValueError(f"Unknown state: {state}")
        if confidence < CONFIDENCE_THRESHOLD:
            return NO_OP_ACTION
        return STATE_RULES[state]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_adaptive.py -v`
Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add adaptive/ tests/test_adaptive.py
git commit -m "feat: add rule-based adaptive decision engine"
```

---

## Task 10: Session Store (SQLite)

**Files:**
- Create: `storage/db.py`
- Test: `tests/test_storage.py`

**Interfaces:**
- Produces: `storage.db.SessionStore(db_path=":memory:")` with `create_session() -> str`, `log_attempt(session_id, puzzle_id, correct, time_to_move, eval_loss, puzzle_rating, predicted_state, confidence, difficulty_delta, show_hint, pacing_delay, sense_to_adapt_latency) -> str`, `get_session_summary(session_id) -> dict` (keys: `num_attempts`, `accuracy_rate`, `avg_latency`, `state_trend`), `close()`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_storage.py
from storage.db import SessionStore

def test_create_session_and_log_attempt_round_trip():
    store = SessionStore(":memory:")
    session_id = store.create_session()
    attempt_id = store.log_attempt(
        session_id=session_id, puzzle_id="p1", correct=True, time_to_move=5.0,
        eval_loss=0.0, puzzle_rating=1000, predicted_state="Focused", confidence=0.9,
        difficulty_delta=50.0, show_hint=False, pacing_delay=0.0, sense_to_adapt_latency=0.05,
    )
    assert attempt_id
    summary = store.get_session_summary(session_id)
    assert summary["num_attempts"] == 1
    assert summary["accuracy_rate"] == 1.0
    store.close()

def test_session_summary_aggregates_multiple_attempts():
    store = SessionStore(":memory:")
    session_id = store.create_session()
    store.log_attempt(session_id, "p1", True, 5.0, 0.0, 1000, "Focused", 0.9, 50.0, False, 0.0, 0.05)
    store.log_attempt(session_id, "p2", False, 8.0, 150.0, 1000, "Confused", 0.7, -100.0, True, 2.0, 0.08)
    summary = store.get_session_summary(session_id)
    assert summary["num_attempts"] == 2
    assert summary["accuracy_rate"] == 0.5
    assert summary["state_trend"] == ["Focused", "Confused"]
    store.close()

def test_summary_of_unknown_session_returns_empty():
    store = SessionStore(":memory:")
    summary = store.get_session_summary("does-not-exist")
    assert summary["num_attempts"] == 0
    store.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_storage.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'storage.db'`

- [ ] **Step 3: Write `storage/db.py`**

```python
from __future__ import annotations
import sqlite3
import time
import uuid

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    started_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS attempts (
    attempt_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    puzzle_id TEXT NOT NULL,
    correct INTEGER NOT NULL,
    time_to_move REAL NOT NULL,
    eval_loss REAL NOT NULL,
    puzzle_rating INTEGER NOT NULL,
    predicted_state TEXT NOT NULL,
    confidence REAL NOT NULL,
    difficulty_delta REAL NOT NULL,
    show_hint INTEGER NOT NULL,
    pacing_delay REAL NOT NULL,
    sense_to_adapt_latency REAL NOT NULL,
    created_at REAL NOT NULL,
    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
);
"""

class SessionStore:
    def __init__(self, db_path: str = ":memory:"):
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        self._conn.execute("INSERT INTO sessions (session_id, started_at) VALUES (?, ?)", (session_id, time.time()))
        self._conn.commit()
        return session_id

    def log_attempt(
        self, session_id: str, puzzle_id: str, correct: bool, time_to_move: float,
        eval_loss: float, puzzle_rating: int, predicted_state: str, confidence: float,
        difficulty_delta: float, show_hint: bool, pacing_delay: float, sense_to_adapt_latency: float,
    ) -> str:
        attempt_id = str(uuid.uuid4())
        self._conn.execute(
            """INSERT INTO attempts (
                attempt_id, session_id, puzzle_id, correct, time_to_move, eval_loss,
                puzzle_rating, predicted_state, confidence, difficulty_delta, show_hint,
                pacing_delay, sense_to_adapt_latency, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                attempt_id, session_id, puzzle_id, int(correct), time_to_move, eval_loss,
                puzzle_rating, predicted_state, confidence, difficulty_delta, int(show_hint),
                pacing_delay, sense_to_adapt_latency, time.time(),
            ),
        )
        self._conn.commit()
        return attempt_id

    def get_session_summary(self, session_id: str) -> dict:
        rows = self._conn.execute(
            "SELECT correct, confidence, sense_to_adapt_latency, predicted_state FROM attempts "
            "WHERE session_id = ? ORDER BY created_at",
            (session_id,),
        ).fetchall()
        if not rows:
            return {"num_attempts": 0, "accuracy_rate": 0.0, "avg_latency": 0.0, "state_trend": []}
        num_attempts = len(rows)
        accuracy_rate = sum(r[0] for r in rows) / num_attempts
        avg_latency = sum(r[2] for r in rows) / num_attempts
        state_trend = [r[3] for r in rows]
        return {"num_attempts": num_attempts, "accuracy_rate": accuracy_rate, "avg_latency": avg_latency, "state_trend": state_trend}

    def close(self) -> None:
        self._conn.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_storage.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add storage/db.py tests/test_storage.py
git commit -m "feat: add SQLite session store for attempts and predictions"
```

---

## Task 11: Session Orchestrator

**Files:**
- Create: `web/session.py`
- Test: `tests/test_integration_session.py`

**Interfaces:**
- Consumes: `eeg.source.EEGSource`; `chess_task.base.{TaskEngine, Puzzle}`; `preprocessing.epoching.EpochBuffer`; `preprocessing.features.extract_node_features`; `model.inference.StatePredictor`; `adaptive.policy.Policy`; `storage.db.SessionStore`; `data_gen.generate_dataset.behavior_to_vector`; `common.config.SEQ_LEN`.
- Produces: `web.session.SessionUpdate(puzzle, correct, predicted_state, confidence, probs, action, sense_to_adapt_latency)`; `web.session.TutorSession(eeg_source, task_engine, predictor, policy, store, initial_difficulty=1000.0)` with `session_id: str`, `difficulty: float`, `next_puzzle() -> Puzzle`, `record_eeg_chunk(chunk) -> None`, `submit_move(puzzle, move_uci, time_to_move) -> SessionUpdate`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_integration_session.py
from pathlib import Path
import torch
from eeg.simulated import SimulatedEEGSource
from chess_task.base import TaskEngine, Puzzle, BehaviorEvent
from model.fusion import FusionLSTMClassifier
from model.inference import StatePredictor
from adaptive.rule_based import RuleBasedPolicy
from storage.db import SessionStore
from web.session import TutorSession
from common.config import NUM_CHANNELS, BAND_NAMES, NUM_BEHAVIOR_FEATS, SEQ_LEN
from common.states import STATES

class FakeTaskEngine(TaskEngine):
    def get_puzzle(self, difficulty: float) -> Puzzle:
        return Puzzle(puzzle_id="fake1", fen="irrelevant", solution_move="e2e4", rating=1000)

    def submit_move(self, puzzle: Puzzle, move_uci: str, time_to_move: float) -> BehaviorEvent:
        return BehaviorEvent(
            puzzle_id=puzzle.puzzle_id, correct=(move_uci == puzzle.solution_move),
            time_to_move=time_to_move, eval_loss=0.0, puzzle_rating=puzzle.rating,
        )

def _make_untrained_checkpoint(tmp_path: Path) -> Path:
    model = FusionLSTMClassifier(
        num_channels=NUM_CHANNELS, num_bands=len(BAND_NAMES),
        num_behavior_feats=NUM_BEHAVIOR_FEATS, num_classes=len(STATES),
    )
    path = tmp_path / "checkpoint.pt"
    torch.save(model.state_dict(), path)
    return path

def test_submit_move_produces_valid_session_update(tmp_path: Path):
    eeg_source = SimulatedEEGSource(target_state="Focused", seed=0)
    task_engine = FakeTaskEngine()
    predictor = StatePredictor(_make_untrained_checkpoint(tmp_path))
    policy = RuleBasedPolicy()
    store = SessionStore(":memory:")
    session = TutorSession(eeg_source, task_engine, predictor, policy, store)

    puzzle = session.next_puzzle()
    session.record_eeg_chunk(eeg_source.generate_chunk())
    update = session.submit_move(puzzle, "e2e4", time_to_move=5.0)

    assert update.correct is True
    assert update.predicted_state in STATES
    assert 0.0 <= update.confidence <= 1.0
    assert update.sense_to_adapt_latency >= 0.0

    summary = store.get_session_summary(session.session_id)
    assert summary["num_attempts"] == 1
    store.close()

def test_multiple_attempts_maintain_rolling_window(tmp_path: Path):
    eeg_source = SimulatedEEGSource(target_state="Overloaded", seed=1)
    task_engine = FakeTaskEngine()
    predictor = StatePredictor(_make_untrained_checkpoint(tmp_path))
    policy = RuleBasedPolicy()
    store = SessionStore(":memory:")
    session = TutorSession(eeg_source, task_engine, predictor, policy, store)

    for _ in range(8):
        puzzle = session.next_puzzle()
        session.record_eeg_chunk(eeg_source.generate_chunk())
        session.submit_move(puzzle, "e2e4", time_to_move=6.0)

    summary = store.get_session_summary(session.session_id)
    assert summary["num_attempts"] == 8
    assert len(session._eeg_history) == SEQ_LEN
    store.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_integration_session.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'web.session'`

- [ ] **Step 3: Write `web/session.py`**

```python
from __future__ import annotations
import time
from dataclasses import dataclass
import numpy as np
from chess_task.base import TaskEngine, Puzzle
from eeg.source import EEGSource
from preprocessing.epoching import EpochBuffer
from preprocessing.features import extract_node_features
from model.inference import StatePredictor
from adaptive.policy import Policy, Action
from storage.db import SessionStore
from common.config import SEQ_LEN
from data_gen.generate_dataset import behavior_to_vector

@dataclass
class SessionUpdate:
    puzzle: Puzzle
    correct: bool
    predicted_state: str
    confidence: float
    probs: dict
    action: Action
    sense_to_adapt_latency: float

class TutorSession:
    def __init__(
        self, eeg_source: EEGSource, task_engine: TaskEngine, predictor: StatePredictor,
        policy: Policy, store: SessionStore, initial_difficulty: float = 1000.0,
    ):
        self.eeg_source = eeg_source
        self.task_engine = task_engine
        self.predictor = predictor
        self.policy = policy
        self.store = store
        self.difficulty = initial_difficulty
        self.session_id = store.create_session()
        self.epoch_buffer = EpochBuffer(num_channels=eeg_source.num_channels, sample_rate=eeg_source.sample_rate)
        self._eeg_history: list[np.ndarray] = []
        self._behavior_history: list[np.ndarray] = []

    def next_puzzle(self) -> Puzzle:
        return self.task_engine.get_puzzle(self.difficulty)

    def record_eeg_chunk(self, chunk) -> None:
        self.epoch_buffer.add_chunk(chunk)

    def submit_move(self, puzzle: Puzzle, move_uci: str, time_to_move: float) -> SessionUpdate:
        start_time = time.monotonic()
        behavior_event = self.task_engine.submit_move(puzzle, move_uci, time_to_move)
        epoch = self.epoch_buffer.extract_epoch()
        node_features = extract_node_features(epoch.samples, self.eeg_source.sample_rate)
        behavior_vector = behavior_to_vector(behavior_event)

        self._eeg_history = (self._eeg_history + [node_features])[-SEQ_LEN:]
        self._behavior_history = (self._behavior_history + [behavior_vector])[-SEQ_LEN:]

        eeg_seq = self._padded_sequence(self._eeg_history, node_features.shape)
        behavior_seq = self._padded_sequence(self._behavior_history, behavior_vector.shape)

        predicted_state, confidence, probs = self.predictor.predict(eeg_seq, behavior_seq)
        action = self.policy.decide(predicted_state, confidence)
        self.difficulty = max(400.0, self.difficulty + action.difficulty_delta)
        latency = time.monotonic() - start_time

        self.store.log_attempt(
            session_id=self.session_id, puzzle_id=puzzle.puzzle_id,
            correct=behavior_event.correct, time_to_move=behavior_event.time_to_move,
            eval_loss=behavior_event.eval_loss, puzzle_rating=behavior_event.puzzle_rating,
            predicted_state=predicted_state, confidence=confidence,
            difficulty_delta=action.difficulty_delta, show_hint=action.show_hint,
            pacing_delay=action.pacing_delay, sense_to_adapt_latency=latency,
        )
        return SessionUpdate(
            puzzle=puzzle, correct=behavior_event.correct, predicted_state=predicted_state,
            confidence=confidence, probs=probs, action=action, sense_to_adapt_latency=latency,
        )

    @staticmethod
    def _padded_sequence(history: list[np.ndarray], item_shape) -> np.ndarray:
        pad_count = SEQ_LEN - len(history)
        if pad_count > 0:
            padding = [np.zeros(item_shape, dtype=np.float32) for _ in range(pad_count)]
            return np.stack(padding + history)
        return np.stack(history)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_integration_session.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add web/session.py tests/test_integration_session.py
git commit -m "feat: add TutorSession orchestrator tying EEG, chess, model, and adaptive engine together"
```

---

## Task 12: FastAPI Web Server + WebSocket

**Files:**
- Create: `web/server.py`
- Test: `tests/test_web_server.py`

**Interfaces:**
- Consumes: `web.session.TutorSession`; `eeg.simulated.SimulatedEEGSource`; `chess_task.puzzles.PuzzleTaskEngine`; `chess_task.evaluator.{MoveEvaluator, StockfishEvaluator}`; `model.inference.StatePredictor`; `adaptive.rule_based.RuleBasedPolicy`; `storage.db.SessionStore`.
- Produces: `web.server.app` (FastAPI instance), `web.server.DEFAULT_CHECKPOINT`, `web.server.NullEvaluator`, `web.server.build_evaluator() -> MoveEvaluator`. WebSocket protocol on `/ws/session`: server sends `{"type": "puzzle", "puzzle_id": str, "fen": str}`, client replies `{"move_uci": str, "time_to_move": float}`, server sends `{"type": "update", "correct": bool, "predicted_state": str, "confidence": float, "probs": dict, "action": {"difficulty_delta": float, "show_hint": bool, "pacing_delay": float}, "difficulty": float}`, then loops back to the next puzzle.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_web_server.py
from pathlib import Path
import torch
from fastapi.testclient import TestClient
from model.fusion import FusionLSTMClassifier
from common.config import NUM_CHANNELS, BAND_NAMES, NUM_BEHAVIOR_FEATS
from common.states import STATES
import web.server as server_module
from storage.db import SessionStore

def test_websocket_session_round_trip(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "checkpoint.pt"
    model = FusionLSTMClassifier(
        num_channels=NUM_CHANNELS, num_bands=len(BAND_NAMES),
        num_behavior_feats=NUM_BEHAVIOR_FEATS, num_classes=len(STATES),
    )
    torch.save(model.state_dict(), checkpoint_path)
    monkeypatch.setattr(server_module, "DEFAULT_CHECKPOINT", checkpoint_path)
    monkeypatch.setattr(server_module, "_store", SessionStore(":memory:"))

    client = TestClient(server_module.app)
    with client.websocket_connect("/ws/session") as ws:
        puzzle_msg = ws.receive_json()
        assert puzzle_msg["type"] == "puzzle"
        assert "fen" in puzzle_msg

        ws.send_json({"move_uci": "a1a8", "time_to_move": 5.0})
        update_msg = ws.receive_json()
        assert update_msg["type"] == "update"
        assert update_msg["predicted_state"] in STATES
        assert 0.0 <= update_msg["confidence"] <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_web_server.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'web.server'`

- [ ] **Step 3: Create `web/static/index.html` placeholder (needed for `StaticFiles` mount)**

```html
<!DOCTYPE html>
<html><head><title>NeuroTutor</title></head><body>Loading...</body></html>
```

- [ ] **Step 4: Write `web/server.py`**

```python
from __future__ import annotations
import shutil
from pathlib import Path
import chess
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from eeg.simulated import SimulatedEEGSource
from chess_task.puzzles import PuzzleTaskEngine
from chess_task.evaluator import MoveEvaluator, StockfishEvaluator
from model.inference import StatePredictor
from adaptive.rule_based import RuleBasedPolicy
from storage.db import SessionStore
from web.session import TutorSession

DEFAULT_CHECKPOINT = Path("model/checkpoints/best.pt")
STATIC_DIR = Path(__file__).parent / "static"

class NullEvaluator(MoveEvaluator):
    """Used when no Stockfish binary is available; treats any wrong move as a fixed loss."""
    def eval_loss(self, board: chess.Board, played_move: chess.Move, best_move: chess.Move) -> float:
        return 200.0

def build_evaluator() -> MoveEvaluator:
    if shutil.which("stockfish") is not None:
        return StockfishEvaluator()
    return NullEvaluator()

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
_store = SessionStore("neurotutor.db")

@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")

@app.websocket("/ws/session")
async def session_endpoint(websocket: WebSocket):
    await websocket.accept()
    eeg_source = SimulatedEEGSource(target_state="Focused")
    task_engine = PuzzleTaskEngine(evaluator=build_evaluator())
    predictor = StatePredictor(DEFAULT_CHECKPOINT)
    session = TutorSession(eeg_source, task_engine, predictor, RuleBasedPolicy(), _store)

    try:
        while True:
            puzzle = session.next_puzzle()
            session.record_eeg_chunk(eeg_source.generate_chunk())
            await websocket.send_json({"type": "puzzle", "puzzle_id": puzzle.puzzle_id, "fen": puzzle.fen})

            message = await websocket.receive_json()
            update = session.submit_move(puzzle, message["move_uci"], message["time_to_move"])

            await websocket.send_json({
                "type": "update",
                "correct": update.correct,
                "predicted_state": update.predicted_state,
                "confidence": update.confidence,
                "probs": update.probs,
                "action": {
                    "difficulty_delta": update.action.difficulty_delta,
                    "show_hint": update.action.show_hint,
                    "pacing_delay": update.action.pacing_delay,
                },
                "difficulty": session.difficulty,
            })
    except WebSocketDisconnect:
        pass
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_web_server.py -v`
Expected: 1 passed.

- [ ] **Step 6: Commit**

```bash
git add web/server.py web/static/index.html tests/test_web_server.py
git commit -m "feat: add FastAPI WebSocket server driving live tutor sessions"
```

---

## Task 13: Frontend (Chess Board + Live State Display)

**Files:**
- Modify: `web/static/index.html`
- Create: `web/static/style.css`
- Create: `web/static/app.js`

**Interfaces:**
- Consumes: the `/ws/session` WebSocket protocol defined in Task 12.
- No Python interfaces produced — this task is pure static frontend, verified manually per Testing Strategy.

- [ ] **Step 1: Replace `web/static/index.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>NeuroTutor — Chess Pilot</title>
  <link rel="stylesheet" href="/static/style.css">
</head>
<body>
  <h1>NeuroTutor</h1>
  <div id="board"></div>
  <div id="status">
    <div>State: <span id="state">-</span> (confidence <span id="confidence">-</span>)</div>
    <div>Difficulty: <span id="difficulty">-</span></div>
    <div id="hint" hidden>Hint: overloaded/confused detected — take your time and re-scan the board.</div>
    <div id="feedback"></div>
  </div>
  <p>Click a piece's square, then click the destination square to move.</p>
  <script src="/static/app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Write `web/static/style.css`**

```css
body { font-family: system-ui, sans-serif; margin: 2rem; }
#board { display: grid; grid-template-columns: repeat(8, 48px); grid-template-rows: repeat(8, 48px); border: 2px solid #333; width: max-content; }
.square { display: flex; align-items: center; justify-content: center; font-size: 32px; cursor: pointer; user-select: none; }
.square.light { background: #eeeed2; }
.square.dark { background: #769656; }
.square.selected { outline: 3px solid #ff5555; outline-offset: -3px; }
#status { margin-top: 1rem; font-size: 18px; }
#hint { color: #b45309; font-weight: bold; }
#feedback { margin-top: 0.5rem; }
```

- [ ] **Step 3: Write `web/static/app.js`**

```javascript
const PIECE_UNICODE = {
  K: "♔", Q: "♕", R: "♖", B: "♗", N: "♘", P: "♙",
  k: "♚", q: "♛", r: "♜", b: "♝", n: "♞", p: "♟",
};

function parseFen(fen) {
  const board = [];
  const rows = fen.split(" ")[0].split("/");
  for (const row of rows) {
    const cells = [];
    for (const ch of row) {
      if (/\d/.test(ch)) {
        for (let i = 0; i < parseInt(ch, 10); i++) cells.push(null);
      } else {
        cells.push(ch);
      }
    }
    board.push(cells);
  }
  return board; // board[0] = rank 8 ... board[7] = rank 1
}

function squareName(rankIdx, fileIdx) {
  const files = "abcdefgh";
  return `${files[fileIdx]}${8 - rankIdx}`;
}

let selectedSquare = null;
let attemptStartMs = null;
let ws = null;

function clearSelection() {
  selectedSquare = null;
  document.querySelectorAll(".square.selected").forEach((el) => el.classList.remove("selected"));
}

function renderBoard(fen) {
  const board = parseFen(fen);
  const boardEl = document.getElementById("board");
  boardEl.innerHTML = "";
  for (let r = 0; r < 8; r++) {
    for (let f = 0; f < 8; f++) {
      const square = document.createElement("div");
      const name = squareName(r, f);
      square.className = "square " + ((r + f) % 2 === 0 ? "light" : "dark");
      square.dataset.square = name;
      const piece = board[r][f];
      if (piece) square.textContent = PIECE_UNICODE[piece];
      square.addEventListener("click", () => onSquareClick(name, square));
      boardEl.appendChild(square);
    }
  }
  clearSelection();
  attemptStartMs = performance.now();
}

function onSquareClick(square, el) {
  if (!selectedSquare) {
    selectedSquare = square;
    el.classList.add("selected");
    return;
  }
  const moveUci = selectedSquare + square;
  const timeToMove = (performance.now() - attemptStartMs) / 1000.0;
  ws.send(JSON.stringify({ move_uci: moveUci, time_to_move: timeToMove }));
  clearSelection();
}

function connect() {
  ws = new WebSocket(`ws://${window.location.host}/ws/session`);
  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.type === "puzzle") {
      renderBoard(msg.fen);
      document.getElementById("feedback").textContent = "";
    } else if (msg.type === "update") {
      document.getElementById("state").textContent = msg.predicted_state;
      document.getElementById("confidence").textContent = msg.confidence.toFixed(2);
      document.getElementById("difficulty").textContent = msg.difficulty.toFixed(0);
      document.getElementById("hint").hidden = !msg.action.show_hint;
      document.getElementById("feedback").textContent = msg.correct ? "Correct!" : "Not quite — next puzzle incoming.";
    }
  };
}

connect();
```

- [ ] **Step 4: Manually verify in a browser**

Run: `uvicorn web.server:app --reload` (requires a trained checkpoint at
`model/checkpoints/best.pt` — produced in Task 14), then open
`http://localhost:8000` and confirm: the board renders from the puzzle's
FEN, clicking two squares sends a move and receives an update, and the
state/confidence/difficulty/hint fields update after each move.

- [ ] **Step 5: Commit**

```bash
git add web/static/
git commit -m "feat: add browser frontend for the chess tutor session"
```

---

## Task 14: End-to-End Wiring, Real Dataset/Checkpoint, README

**Files:**
- Create: `README.md`
- Modify: `.gitignore` (verify `data/` and `model/checkpoints/` are already covered from Task 1 — no change expected)

**Interfaces:**
- No new code interfaces — this task runs the pipeline built in Tasks 1-13 end-to-end and documents it.

- [ ] **Step 1: Write `README.md`**

```markdown
# NeuroTutor — Chess Pilot

A cognitive-state-aware chess tutor: a simulated EEG signal is fused with
live chess-puzzle performance through a GCN+MLP+LSTM classifier to predict
one of 5 cognitive states (Focused, Overloaded, Confused, Fatigued,
Engaged) and adapt puzzle difficulty, hints, and pacing in real time.

See `docs/superpowers/specs/2026-09-10-chess-eeg-tutor-design.md` for the
full design, and `docs/superpowers/plans/2026-09-10-chess-eeg-tutor.md`
for the implementation plan.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional, for real move-quality evaluation on incorrect puzzle attempts
(falls back to a fixed penalty if absent):

```bash
brew install stockfish   # macOS
```

## Generate the synthetic training dataset

```bash
python -m data_gen.generate_dataset --out-dir data/synthetic --examples-per-state 200
```

## Train the fusion classifier

```bash
python -m model.train --data-dir data/synthetic --checkpoint-path model/checkpoints/best.pt
```

Prints validation/test accuracy and F1 once training completes.

## Run the tutor

```bash
uvicorn web.server:app --reload
```

Open `http://localhost:8000`, click a piece's square then a destination
square to move. The live predicted cognitive state, confidence, and
adaptive difficulty update after each attempt.

## Run the tests

```bash
pytest
```

## Scope of this build

This is the first sub-project of the larger NeuroTutor capstone (see the
spec's "Non-goals" section): a real fusion classifier trained on synthetic
data, behind an `EEGSource` interface a real headset can implement later,
validated on chess tactics puzzles. ONNX/TensorRT edge deployment and real
EEG hardware integration are follow-on sub-projects.
```

- [ ] **Step 2: Generate the real synthetic dataset**

Run: `python -m data_gen.generate_dataset --out-dir data/synthetic --examples-per-state 200`
Expected: `Wrote synthetic dataset splits to data/synthetic` and three `.npz` files under `data/synthetic/`.

- [ ] **Step 3: Train the real checkpoint**

Run: `python -m model.train --data-dir data/synthetic --checkpoint-path model/checkpoints/best.pt`
Expected: printed dict with `test_accuracy` and `test_f1`; `model/checkpoints/best.pt` exists. (Both `data/` and `model/checkpoints/` are gitignored — this is a local build artifact, not committed.)

- [ ] **Step 4: Run the full test suite**

Run: `pytest -v`
Expected: all tests from Tasks 1-13 pass.

- [ ] **Step 5: Manually verify the live app end-to-end**

Run: `uvicorn web.server:app --reload`, open `http://localhost:8000` in a
browser, play through at least 5 puzzles, and confirm the predicted state
and difficulty visibly change across attempts (this exercises the full
loop: simulated EEG → preprocessing → trained model → adaptive policy →
SQLite → frontend).

- [ ] **Step 6: Commit**

```bash
git add README.md
git commit -m "docs: add setup/run instructions and finalize chess-pilot build"
```
