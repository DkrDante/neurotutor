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
