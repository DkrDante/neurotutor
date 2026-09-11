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

## Ablation study: does the EEG branch actually matter?

The synthetic data generator gives each of the 5 cognitive states both a
distinct behavior profile (`data_gen/virtual_player.py`) *and* a distinct
EEG band profile (`eeg/simulated.py`), so headline accuracy alone doesn't
prove the EEG/GCN branch is contributing anything — the classes could be
separable from behavior features alone. Run the ablation:

```bash
python -m model.ablation --data-dir data/synthetic --epochs 20
```

This trains three variants on the *same* dataset splits — the real fused
model, a behavior-only variant (EEG input zeroed), and an EEG-only variant
(behavior input zeroed) — and reports all three. A run on the default
200-examples-per-state dataset produced:

| variant       | val_acc | test_acc | test_f1 |
|---------------|---------|----------|---------|
| fused         | 1.0000  | 1.0000   | 1.0000  |
| behavior_only | 0.7933  | 0.7333   | 0.7230  |
| eeg_only      | 1.0000  | 1.0000   | 1.0000  |

Two takeaways, and both matter for how to read this pilot's results:

1. **The fused model isn't being carried by behavior alone** — behavior-only
   caps out around 73%, well below the fused model's 100%. In *this*
   synthetic dataset, the EEG branch is doing real, necessary work.
2. **But the EEG-only variant matches the fused model exactly**, so this
   run doesn't establish that *fusing* the two modalities beats a
   good-enough EEG signal by itself — it only establishes that behavior
   alone isn't sufficient. That's because `eeg/simulated.py`'s per-state
   band-power parameters are quite cleanly separated (a modeling choice
   for a first pilot), which makes the EEG signal easier to classify than
   the noisier, probability/Gaussian-sampled behavior features. Real EEG
   will not be this clean, so don't take the 100% EEG-only number as
   evidence about real hardware — take it as evidence that this dataset's
   EEG simulation needs to be noisier before its numbers mean much, and
   that fusion's actual value can only be judged once EEG signal quality
   is closer to real headset data.

No claims about the value of sensor fusion should be drawn from headline
accuracy alone on this dataset — always check the ablation table above it.

## Scope of this build

This is the first sub-project of the larger NeuroTutor capstone (see the
spec's "Non-goals" section): a real fusion classifier trained on synthetic
data, behind an `EEGSource` interface a real headset can implement later,
validated on chess tactics puzzles. ONNX/TensorRT edge deployment and real
EEG hardware integration are follow-on sub-projects.
