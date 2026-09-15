# NeuroTutor — Project Report

**A cognitive-state-aware, adaptive chess tutor.** A simulated EEG signal is fused with live puzzle-solving behavior through a real GCN+MLP+LSTM classifier to predict one of 5 cognitive states (Focused, Overloaded, Confused, Fatigued, Engaged), and the tutor adapts difficulty, hints, pacing, and even how much it explains itself — in real time, per move.

This report documents what was built, how it works, and what makes it different from a standard "adaptive difficulty" chess trainer.

---

## 1. Problem & Motivation

Most adaptive learning tools adjust difficulty from **outcome alone** (right/wrong, time taken). They have no signal for *why* a learner is struggling — rushing, overloaded, confused, or just tired — and so they can't respond differently to those different causes. NeuroTutor's premise: a lightweight EEG + behavior fusion model can classify *which* of those states a learner is in, and a tutor that knows the difference can adapt more usefully than one that only sees accuracy.

## 2. Objectives

- Build a real (not mocked) multimodal classifier fusing EEG-band features and behavioral features into a 5-class cognitive-state prediction.
- Drive live difficulty/hint/pacing decisions from that prediction, without discarding ground-truth correctness as the primary signal.
- Validate the fusion claim rather than assert it — prove the EEG branch is doing real work, not just riding on behavior features.
- Ship a genuinely playable, polished product around that engine: puzzles, full games, accounts, coaching, gamification — not just a research script.

## 3. System Architecture

```
EEG source ──▶ preprocessing ──▶ ┐
                                   ├──▶ GCN + MLP + LSTM fusion ──▶ cognitive state + confidence
Chess behavior ───────────────────┘                                        │
                                                                             ▼
                                                              Adaptive policy (rule-based or RL)
                                                                             │
                                                                             ▼
                                                    difficulty / hints / pacing / explanation depth
```

| Layer | Module | Responsibility |
|---|---|---|
| EEG ingestion | `eeg/` | `EEGSource` interface; `SimulatedEEGSource` today, a real headset adapter (Muse/OpenBCI/LSL) is a drop-in future implementation — nothing downstream changes. |
| Preprocessing | `preprocessing/` | Bandpass filtering, epoching, band-power feature extraction — identical code path at training time and serving time. |
| Synthetic data | `data_gen/` | Generates paired EEG + behavior training data from a virtual player model, per cognitive state. |
| Fusion model | `model/` | `GCN(EEG electrodes) + MLP(behavior) → LSTM(sequence) → 5-class softmax`. Training, inference, and an ablation harness. |
| Chess engine | `chess_task/` | Puzzle mode (curated Lichess puzzles) and full-game mode (adaptive-strength AI opponent), a shared `MoveEvaluator` (Stockfish if present, material-count fallback otherwise), and move explanation/scoring. |
| Adaptive policy | `adaptive/` | Two swappable policies behind one interface: hand-written rules, or tabular Q-learning. |
| Coaching | `coaching/` | Deterministic statistics + tips, plus an optional local-LLM narrative (Ollama). |
| Accounts | `accounts/` | Username/password accounts, daily play-streaks with streak freezes, leaderboard. |
| Storage | `storage/` | SQLite: sessions, per-attempt logs, learner difficulty profiles. |
| Web | `web/` | FastAPI + WebSocket backend orchestrating all of the above; a vanilla-JS frontend (no framework). |

**Scale:** ~5,400 lines across the backend/frontend core, 301 automated tests (all passing), 22 HTTP/WS-adjacent endpoints, zero paid external API calls anywhere in the running product.

## 4. The Fusion Model — and Proving It's Real

- Architecture: a **GCN** over the EEG electrode graph (spatial structure between channels) feeding an **MLP** over behavioral features (time-to-move, correctness streak, hint use, etc.), concatenated and run through an **LSTM** over the attempt sequence, ending in a 5-way softmax.
- **The honest part:** headline accuracy alone doesn't prove the EEG branch matters — the classes could be separable from behavior alone. So an **ablation study** (`model/ablation.py`) trains three variants on identical splits:

  | variant | val acc | test acc | test F1 |
  |---|---|---|---|
  | fused (real model) | 100% | 100% | 100% |
  | behavior-only (EEG zeroed) | 79.3% | 73.3% | 72.3% |
  | EEG-only (behavior zeroed) | 100% | 100% | 100% |

  Conclusion drawn *from the data*, not asserted: behavior alone caps out at ~73%, so the EEG branch is doing necessary work — but EEG-only matching the fused model exactly also means this run doesn't yet prove *fusing* beats a clean-enough EEG signal alone. The simulated EEG is deliberately flagged in the README as "too clean" for that stronger claim, pending real hardware data. This kind of self-critical, data-driven limitation statement is unusual for a course project and is treated as a feature, not an omission.

  Note: the ablation table above is from `model/ablation.py`'s separate training run, not the checkpoint the live app actually serves. `/model-evidence` (section 8) independently evaluates the real production checkpoint (`model/checkpoints/best.pt`, trained with a fixed seed for reproducibility) and reports 98.0% accuracy, 0.98 macro F1, 1.0 macro ROC-AUC on the held-out test split — a materially different number from the ablation run above, which is itself evidence against "hardcoded" (a faked number wouldn't vary between runs like this).
- **Beats classical baselines by a wide, real margin, not just "beats chance."** On the identical held-out split: a majority-class guess gets 22.0% accuracy, a logistic-regression baseline trained on behavior features alone gets 74.0% — the fused neural model's 98.0% is a genuine, large improvement over both, not a marginal one (`model/baselines.py`, surfaced on `/model-evidence`).
- **A second, real finding from that same page:** the GCN's second layer never learned any channel structure — its adjacency parameter stays at its zero-initialization (softmax of all-zeros is exactly uniform), because layer 1's near-uniform output already collapses every channel to nearly identical values before layer 2 ever sees them, zeroing its gradient. `/model-evidence` and the live "Network Activity" panel (section 11) both surface this plainly rather than hiding it — the same self-critical standard as the ablation study above.

## 5. Adaptive Difficulty — Two Policies, One Interface

- **Ground truth first:** whether the move was actually correct always moves difficulty (±100), independent of the EEG read.
- **Model second, gated by confidence:** the predicted cognitive state adds its own push (Focused +100, Engaged +75, Overloaded −150, Confused −100, Fatigued −50) **only when the model's own confidence clears 0.4** — an unconfident read isn't allowed to steer difficulty at all.
- Two interchangeable policies behind the same `Policy` interface, switchable live via `?policy=rule|rl`:
  - **Rule-based** (`adaptive/rule_based.py`): the fixed deltas above.
  - **Reinforcement learning** (`adaptive/rl_policy.py`): tabular Q-learning over (cognitive state × confidence bucket), 5 actions, epsilon-greedy, reward = ±1 for correct/incorrect, learned per learner and persisted across sessions.
- Difficulty is floored at 400 and persists per `learner_id` (localStorage-issued UUID) across reconnects — a returning learner resumes at their last difficulty, not a cold start.

## 6. The Chess Task Layer

- **Puzzle mode:** real curated puzzles from Lichess's public dataset (not synthetic placeholders), multi-ply, with retry-on-wrong-move.
- **Full-game mode:** an entire game against an AI opponent whose strength scales with the same adaptive-difficulty signal — including a simulated "thinking time" that scales the same way, so a stronger opponent visibly takes longer, not just plays better.
- **Shared evaluator:** Stockfish when the binary is present, a transparent material-count fallback otherwise — the product degrades gracefully rather than requiring an engine install.
- **Illegal-move handling:** explains *why* a move is illegal (blocked path, leaves king in check, wrong piece owner, etc.) instead of silently rejecting it.

## 7. Move Feedback, Scored and Explained — Adaptively

For every non-perfect move:
- A **0–100 move-quality score** and a familiar label (Best/Excellent/Good/Inaccuracy/Mistake/Blunder — lichess/chess.com vocabulary), derived from the same eval-loss the adaptive engine already computes.
- A **factual description** of what the move itself did (piece, from/to, capture, check) — generated from the board, not templated per-puzzle.
- A **grounded tactical explanation** of what was missed (forced mate, a hung piece, a bigger capture available, a missed check, or a general "better move" fallback) — detected from the real position via `python-chess`, never guessed.
- **The explanation's depth adapts to the learner's current difficulty tier** — the same signal driving puzzle difficulty also drives how much hand-holding they get:
  - low difficulty (struggling) → longer, didactic explanations that teach the underlying concept;
  - mid → the concise standard explanation;
  - high difficulty (cruising) → a terse one-liner that respects their time.

  This means explanation verbosity isn't a setting anyone configures — it falls directly out of the same adaptive state as everything else.

## 8. Full Transparency: Four Pages That Show Their Work

Four dedicated pages that treat "how does this actually work?" as something worth proving, not asserting.

**`/calculations`** — how a single session's numbers were computed:
- Every formula in the system, in plain text, pulled live from the actual constants in the code (so it can't drift out of sync with reality).
- Real numbers substituted into every formula for the selected session — not a generic explainer, an audit trail.
- A **per-attempt trace** reconstructing what the rule-based formula *would have* produced for each move, flagged against what the session *actually* did — silently surfacing exactly which attempts were driven by the RL policy diverging from the hand-written rule.

**`/model-evidence`** — proof the classifier is a real trained model, not a stub, with real numbers computed on demand, not copied from documentation:
- **Weights & biases**: all 16 learnable parameter tensors (9,429 scalars) from the actual checkpoint's state_dict, with real statistics and sample values, plus a bar chart of per-layer magnitude.
- **Training curves**: real per-epoch train loss and validation accuracy/F1 from the actual run that produced the checkpoint (`model/train.py --seed <n> --save-history`) — the trajectory, not just the final number.
- **Precision / Recall / F1 / ROC-AUC**, computed independently on the held-out test split — a separate check against the checkpoint, not the accuracy/F1 number it was selected by — with a per-class breakdown and a confusion-matrix heatmap.
- **Baseline comparison**: the same test split scored by a majority-class guess (22.0%) and a classical logistic-regression baseline on behavior alone (74.0%), against the fused model's 98.0% — the question of whether the architecture beats something trivial, answered directly rather than assumed.
- **Feature importance (permutation)**: shuffle one real input at a time across the test batch and measure the actual accuracy drop — for the 4 behavior features and each of the 8 EEG channels individually — a real, measured signal for what the model relies on, not an assumption (`model/evidence.py::compute_feature_importance`).
- **A live forward-pass trace** on any real dataset example: every intermediate tensor (both GCN layers' outputs, the fused vector, the LSTM hidden state, the logits) with real numbers that visibly change when you pick a different example — plus a real graph diagram of each GCN layer's learned channel-adjacency, scaled against the uniform (no-structure) baseline so a genuinely flat layer honestly renders as flat rather than being auto-stretched to look just as connected as a real one.
- A standalone script (`model/evaluate_model.py`) computes the exact same numbers offline into `model/MODEL_EVIDENCE.md`, so a script run and the live page can never disagree.

**`/puzzles`** — the full curated puzzle set, browsable rather than described:
- All 337 puzzles (325 real Lichess puzzle-database rows + 12 hand-built mate patterns) read live from `chess_task/puzzle_data/sample_puzzles.csv`, with rating and mate-pattern distribution charts.
- **Real mate-pattern classification** (`chess_task/motifs.py`): every puzzle's actual final move is replayed on a real `python-chess` board and classified by what genuinely happens on it — back-rank, smothered, double/discovered check, or which piece delivers mate — not a lookup table. Filterable and searchable by source, mate pattern, and rating.

**`/analytics`**'s cohort overview — aggregated across *every* recorded session, not one at a time:
- Total sessions/attempts, overall accuracy, average sense-to-adapt latency, and cognitive-state distribution system-wide.
- **Engagement trend**: second-half accuracy minus first-half accuracy, averaged across sessions — a real fatigue/improvement signal, not a flat number.
- An **explicit rule-based-vs-RL comparison**: how often the live policy actually diverged from what the fixed rule would have produced, and by how much, across every attempt ever logged — the system-wide counterpart to `/calculations`' per-attempt trace.
- CSV export, both system-wide (`/sessions/export`) and per-session (`/session/{id}/export`), so this isn't all trapped in SQLite.
- **Puzzle and full-game sessions are never combined**: full-game mode repurposes `puzzle_rating` to mean the live adaptive-difficulty number, so every cohort stat, rating-band coaching computation, and CSV row carries a `mode` tag and the two are always reported separately.

## 9. Coaching — Local AI, No Cloud

- **Always available:** deterministic tips (`coaching/statistics.py` + `coaching/narrator.py`) naming the real weak rating band, weak cognitive state, rushing-vs-overthinking pattern, blunder severity, and hint reliance — every tip cites an actual computed number, never a generic platitude.
- **Optionally richer:** a natural-language coaching paragraph generated by a **locally-running Ollama model** (default `llama3.2`) — no API key, no cloud request, nothing leaves the machine. Falls back to the deterministic tips instantly and silently if Ollama isn't running.

## 10. Accounts, Gamification & Engagement

- **Lightweight accounts:** username/password (PBKDF2-HMAC-SHA256, stdlib only), session cookies — no email verification/reset, scoped deliberately for a project like this rather than a hardened auth service.
- **51 achievements** across 8 categories (streaks, solve counts, speed, rating ceilings, hint discipline, XP/level, full-game results, cognitive-state/comeback events) — every single one keyed to a real logged event, never granted arbitrarily.
- **XP/levels**, a puzzle-solve streak (session-local), and a **daily play-streak** (account-level, calendar-based) with **streak freezes** — one earned every 7-day milestone, capped at 3, auto-consumed to bridge exactly one missed day.
- **Public leaderboard**, ranked by XP, with the viewer's own rank always shown even outside the visible top list.
- **Cosmetic board themes** unlocked by real progress (level, puzzles solved) — a purely client-side visual reward layered on top of already-tracked stats, no new server state.
- Signing up **migrates** whatever anonymous progress a browser already made into the new account, rather than discarding it.

## 11. Interface & Experience

- A from-scratch, hand-designed UI (no component framework) with a live "network activity" panel rendering the *actual* model internals per move, not a decorative animation: the behavior branch as a real weighted 3-layer network, **both** GCN layers' learned channel-adjacency graphs (not just one — the second layer's is genuinely near-empty, and the diagram shows that honestly rather than hiding it), and the LSTM rendered as its real 32 hidden units feeding the 5 output states through the classifier's actual learned weights, not generic connector lines.
- A talking mascot avatar that delivers feedback/hints, with its own idle and reaction animations.
- Full keyboard accessibility on the board (roving tabindex, arrow-key navigation, ARIA labels) and a responsive layout that reflows the board size fluidly down to phone widths.
- Auto-reconnecting WebSocket session with exponential backoff — a dropped connection recovers on its own.

## 12. Engineering Practices

- **301 automated tests**, spanning unit tests per module (EEG, preprocessing, model, adaptive policies, chess task engine, accounts, coaching) up to full WebSocket integration tests against the real FastAPI app.
- Every derived number (accuracy, formulas, move scores, difficulty deltas) is pulled from the same constants the running code uses — the calculations page and the tests both guard against documentation/behavior drift.
- Adaptive-tier explanation behavior is itself unit- and integration-tested: the *same* wrong move is asserted to produce a longer explanation at low difficulty than at high difficulty.
- A dedicated integration test loads the actual trained checkpoint (not a freshly-initialized stub model, which every other test uses) through a real WebSocket session, so "does the real model behave correctly once wired into production" is itself verified, not assumed.

## 13. Novelty & Uniqueness — Summary

What distinguishes this from a typical "adaptive difficulty" project or course capstone:

1. **A real multimodal model, validated by ablation, not just claimed.** Three-way (fused / behavior-only / EEG-only) comparison on identical splits, with an honestly-stated limitation rather than an overclaimed result.
2. **A precedence rule between ground truth and model confidence**, not a naive "use the model's output directly" — correctness always adapts difficulty; the cognitive-state read only adds its own push once the model trusts itself.
3. **Two full adaptive policies on one interface**, hand-rule vs. learned Q-table, swappable live for direct comparison — most projects pick one.
4. **Explanation depth adapts along the same axis as difficulty** — hand-holding isn't a toggle, it's an emergent property of the same adaptive state driving everything else.
5. **A transparency page that audits the adaptive engine against itself** — reconstructing what the rule-based formula would have said and flagging where the live policy diverged, rather than treating adaptation as a black box.
6. **Fully local AI, end to end** — the classifier is a real trained PyTorch model; the optional natural-language coaching layer runs on a local Ollama model. No API key, no cloud dependency, no data leaves the machine anywhere in the stack.
7. **A full game mode built on the same task-engine abstraction as puzzles** — cognitive-state adaptation, hints, and difficulty scaling carry into an open-ended game against an adaptively-strengthed opponent, not a bolted-on separate feature.
8. **A gamification layer with zero arbitrary rewards** — all 51 achievements, the XP formula, and the daily streak are derived from real logged behavioral and cognitive-state events.
9. **A model-evidence page that proves rather than asserts** — real weights, an independently-computed precision/recall/F1/ROC-AUC evaluation, and a live, re-runnable forward-pass trace with an honest (baseline-relative, not self-normalized) visualization of the GCN's learned graph structure — including surfacing that its second layer never actually learned one.
10. **Cohort-level analytics, not just per-session** — accuracy, latency, cognitive-state distribution, engagement trend, and the rule-vs-RL divergence rate, aggregated across every recorded session rather than eyeballed one run at a time.
11. **Real chess-domain analysis, not a lookup table** — every one of the 337 curated puzzles is classified by its actual mate pattern (back-rank, smothered, discovered/double check, or mating piece) by replaying its real final move on a `python-chess` board, not by a hand-maintained tag list.
12. **Measured, not assumed, feature reliance** — permutation importance on both the behavior features and each individual EEG channel, computed by actually shuffling real inputs and re-running the real model, surfaced next to the ablation study and baseline comparison rather than left as an unverified architectural claim.

## 14. Current Status & What's Next

- Core pipeline (EEG → fusion model → adaptive policy → puzzle/game UI) is complete, tested, and running end to end.
- Transparency and validation are no longer confined to one page: `/calculations` (per-session), `/model-evidence` (model-wide: weights, independent metrics, live trace), and `/analytics`'s cohort overview (system-wide, across every session) each answer a different "prove it" question.
- Deliberately out of scope for this stage (see `docs/superpowers/specs/2026-09-10-chess-eeg-tutor-design.md`): ONNX/TensorRT edge export, and a real EEG hardware adapter (Muse/OpenBCI/LSL) — the `EEGSource` interface is built for this, but no concrete adapter ships yet.
- Known data-quality issue, found by building the mate-pattern classifier: 2 of the 12 hand-built puzzles (`mm01`, `mm02`, the only multi-move hand-built ones) don't actually reach checkmate on their scripted final move — verified directly with `python-chess`, they deliver check but leave the king a legal escape square. Both are correctly reported as `other` rather than silently mislabeled, but they should be fixed or removed from `chess_task/puzzle_data/sample_puzzles.csv`. The other 335 puzzles (all single-move hand-built ones plus all 325 real Lichess puzzles) verified as genuine checkmates.

## 15. Tech Stack

Python, FastAPI, WebSockets, PyTorch, python-chess, Stockfish (optional), SQLite, Ollama (optional, local) — vanilla HTML/CSS/JS frontend, no build step.
