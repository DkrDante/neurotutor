# NeuroTutor — Chess Pilot: Design Spec

Date: 2026-09-10
Status: Approved for planning

## Background

NeuroTutor (BCSE497J, Team Yashvardhan Singh / Nishtha Tiwari) is a B.Tech
capstone project: a cognitive-state-aware adaptive tutoring framework that
fuses EEG signals with live task performance to classify a learner's
cognitive state (Focused, Overloaded, Confused, Fatigued, Engaged) and adapt
instruction in real time. The academic proposal (see
`NeuroTutor_Academic_Review1 (1).pptx`) targets a GCN+MLP+LSTM fusion model,
edge deployment via ONNX/TensorRT, and validation on chess as the pilot
domain, with the architecture intended to generalize to other structured
tasks later.

This spec scopes the **first buildable sub-project**: a working, end-to-end
chess tutoring application with a real (not stubbed) fusion classifier,
trained on synthetic data, behind an EEG interface designed so real hardware
can be plugged in later without changing the rest of the system. Edge
deployment (ONNX/TensorRT) is out of scope here — it depends on physical
edge hardware and is a natural follow-on sub-project once this pipeline
works end-to-end.

## Goals

- A playable chess tutoring app (tactics puzzles) that adapts difficulty,
  hints, and pacing in real time based on a predicted cognitive state.
- A real GCN+MLP+LSTM fusion classifier (per the proposal's architecture),
  trained on synthetic EEG+behavior data generated in-repo, producing the
  5-class softmax output the rest of the system consumes.
- An EEG ingestion boundary (`EEGSource`) that today is satisfied by a
  simulator, and tomorrow by a real headset adapter, with zero changes to
  preprocessing, the model, or anything downstream.
- Session logging sufficient to report accuracy/F1, sense→adapt latency,
  and engagement trend, matching the proposal's validation criteria.

## Non-goals (this sub-project)

- ONNX export / TensorRT compilation / edge (Jetson-class) deployment.
- Real EEG hardware integration (Muse, OpenBCI, LSL adapters) — the
  interface is built for this, but no concrete adapter ships yet.
- Full-game chess mode (interface stubbed only; puzzles are the only
  playable mode).
- Multi-session personalization, RL-based adaptive policy, explainability
  tooling — all mentioned as "longer-term extensions" in the proposal.

## Architecture

Single Python backend (FastAPI + WebSocket) serving a browser-based
frontend, in one repo. Model **training** is a separate, offline concern
(scripts producing a checkpoint file) from model **serving** (the live
backend loads a checkpoint for inference) — mirroring the proposal's own
Stage A (model development) / Stage B (real-time deployment) split. This
is the seam where ONNX/TensorRT slots in later without touching serving
code, and keeps a 2-person team from building unneeded service boundaries
up front.

```
                         ┌─────────────────────────────────────┐
                         │           Training (offline)         │
                         │  synthetic_dataset_gen → train.py    │
                         │        → model checkpoint (.pt)      │
                         └───────────────┬───────────────────────┘
                                         │ loads
                                         ▼
┌────────────┐   epochs   ┌───────────────────────┐   state+conf   ┌──────────────┐
│ EEGSource   │──────────▶│                       │───────────────▶│              │
│(Simulated)  │            │   Backend (FastAPI)   │                │   Frontend    │
└────────────┘            │  preprocessing→fusion │   actions      │ (board, state,│
┌────────────┐  behavior  │  model→adaptive engine│───────────────▶│  hints, log)  │
│Chess Task   │──────────▶│                       │                │              │
│Engine       │            └───────────┬───────────┘                └──────────────┘
└────────────┘                        │ writes
                                       ▼
                                ┌─────────────┐
                                │   SQLite     │
                                │(session log) │
                                └─────────────┘
```

## Components

### 1. EEG Source Interface (`eeg/`)

- `EEGSource` abstract base: async generator yielding fixed-length,
  multi-channel raw sample windows plus a monotonic timestamp.
- `SimulatedEEGSource`: the only implementation now. Synthesizes
  theta/alpha-band power that tracks a target cognitive state (with
  realistic noise/drift), driven either by a fixed state (for manual
  testing) or by the synthetic dataset generator's virtual player.
- Real hardware later (Muse, OpenBCI, or generic LSL) implements the same
  interface; nothing downstream changes.

### 2. Chess Task Engine (`chess_task/`)

- Built on `python-chess` for board/move logic and a local Stockfish
  binary for objective move evaluation.
- Puzzle mode: serves puzzles from the public Lichess puzzle dataset
  (rated by difficulty), one at a time, at a difficulty level the
  adaptive engine controls.
- Emits a behavior event per attempt: correctness, time-to-move,
  Stockfish eval-loss, puzzle rating.
- Full-game mode: interface defined (`TaskEngine` base), not implemented.

### 3. Preprocessing (`preprocessing/`)

- Band-pass filter (1–40 Hz).
- Epoching aligned to puzzle-attempt boundaries — one epoch per attempt.
- Lightweight artifact rejection (amplitude thresholding); full ICA is
  deferred until real hardware makes it necessary.

### 4. Synthetic Dataset Generator (`data_gen/`)

- Drives `SimulatedEEGSource` plus a scripted "virtual player" whose
  play pattern (speed, error rate) is parameterized per target cognitive
  state (e.g., Fatigued → slower and more error-prone).
- Produces (EEG epoch, behavior features, state label) tuples on disk in
  a schema real recorded data would also use — swapping in a real dataset
  later is a data-loader change only, not a schema or model change.

### 5. Fusion Model (`model/`)

- PyTorch + PyTorch Geometric. GCN encoder over EEG channels + MLP
  encoder over behavior features, merged via a Fusion+LSTM head into a
  5-class softmax (Focused / Overloaded / Confused / Fatigued /
  Engaged), matching the proposal's O1 objective.
- `train.py`: offline training script against the synthetic dataset,
  reporting accuracy/F1/ROC-AUC, saving the best checkpoint.
- Live backend loads the checkpoint for inference only; no training code
  runs in the serving path.

### 6. Adaptive Decision Engine (`adaptive/`)

- `Policy` interface: predicted state + confidence → action (difficulty
  delta, hint trigger, pacing delay).
- `RuleBasedPolicy`: the only implementation now, using deck-consistent
  rules (e.g., Overloaded/Confused → lower difficulty + offer hint;
  Focused/Engaged → raise difficulty; Fatigued → slow pacing).
- An RL-based policy can implement the same interface later.

### 7. Session Store (`storage/`)

- SQLite. Tables: sessions, puzzle attempts, predicted states (with
  confidence), adaptive actions taken.
- Sufficient to compute accuracy/precision/recall/F1/ROC-AUC (against
  synthetic ground truth), sense→adapt latency, and engagement trend
  across a session — the proposal's validation metrics (O4).

### 8. Tutor Interface (`web/`)

- FastAPI backend, WebSocket stream of live state/confidence/actions to
  a single-page frontend.
- Frontend: chess board (chessboard.js or equivalent), live cognitive
  state + confidence bar, adaptive hints as they fire, session summary.

## Data Flow (live session)

1. Player is served a puzzle at the current difficulty.
2. `SimulatedEEGSource` streams samples continuously; `Chess Task Engine`
   emits a behavior event when the player moves/submits.
3. On attempt completion, preprocessing cuts the aligned EEG epoch and
   extracts behavior features.
4. Fusion model produces a state + confidence.
5. `RuleBasedPolicy` maps that state to the next puzzle's difficulty,
   whether to show a hint, and pacing.
6. Backend persists the attempt/state/action to SQLite and pushes the
   update to the frontend over WebSocket.
7. Loop continues with the next puzzle.

## Testing Strategy

- Unit tests per module against its interface (`EEGSource`,
  `TaskEngine`, `Policy`) using fakes/synthetic fixtures — no module
  test depends on another module's internals.
- Model: train/validate/test split on the synthetic dataset; assert
  accuracy/F1 above a baseline threshold before a checkpoint is accepted.
- Integration test: one scripted session end-to-end (simulated EEG +
  scripted puzzle attempts) asserting a state is produced, an action is
  taken, and it's persisted — no browser required.
- Manual UI verification in-browser for the frontend once wired up.

## Tech Stack

- Backend: Python, FastAPI, WebSockets.
- ML: PyTorch, PyTorch Geometric, scikit-learn (metrics), MNE-Python
  (filtering/epoching) or a lightweight equivalent.
- Chess: python-chess, Stockfish (local binary), Lichess puzzle dataset.
- Storage: SQLite.
- Frontend: single-page app, chessboard.js (or equivalent) over vanilla
  JS/React — kept light since this isn't the project's core novelty.

## Open Questions / Follow-on Sub-projects

- Milestone 2: ONNX export + TensorRT compilation + edge (Jetson-class)
  deployment, once this pipeline is validated.
- Milestone 3: a real `EEGSource` adapter (Muse/OpenBCI/LSL) once
  hardware is available, validated against the synthetic-trained model
  and then retrained on real recordings.
- Full-game chess mode, RL-based adaptive policy, multi-session
  personalization — deferred per the proposal's own "longer-term
  extensions."
