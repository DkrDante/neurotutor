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

## Known limitation: headline accuracy does not prove the fusion works

The synthetic data generator gives each of the 5 cognitive states both a
distinct behavior profile (`data_gen/virtual_player.py`) *and* a distinct
EEG band profile (`eeg/simulated.py`). The classes are therefore separable
from the behavior features alone, so the trained model's high test accuracy
does **not** by itself demonstrate that the EEG/GCN branch contributes
anything to the prediction — a behavior-only model would likely score
similarly on this dataset.

This is a property of the synthetic data, not of the architecture, and it
is not something the current metrics can distinguish. Anyone extending this
work should run **ablations** before drawing conclusions about the fusion:

- train and evaluate with the EEG branch zeroed/removed (behavior-only),
- train and evaluate with the behavior branch zeroed/removed (EEG-only),
- compare both against the full fusion model on the same splits.

Only a fusion model that clearly beats both single-modality baselines
demonstrates that combining the streams is doing real work. No ablation
harness ships with this build.

## Scope of this build

This is the first sub-project of the larger NeuroTutor capstone (see the
spec's "Non-goals" section): a real fusion classifier trained on synthetic
data, behind an `EEGSource` interface a real headset can implement later,
validated on chess tactics puzzles. ONNX/TensorRT edge deployment and real
EEG hardware integration are follow-on sub-projects.
