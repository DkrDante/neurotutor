"""Train and honestly evaluate the fusion model on REAL headset recordings.

Why this is not just model/train.py pointed at new data:

* Examples cut from one block are near-duplicates (overlapping windows from the
  same minutes of the same brain), so a random train/test split leaks and gives
  inflated accuracy. Everything here is split by BLOCK (or by participant),
  and every example is scored exactly once, by a model that never saw its block.
* With a handful of blocks per class the effective sample size is the number
  of BLOCKS, not examples — so the report states block counts, block-level
  accuracy ("did the model call the whole block right?"), and compares against
  baselines that are hard to beat on small data (majority class, logistic
  regression on behaviour / on band power / on both), plus a fused / EEG-only /
  behaviour-only neural ablation on the very same folds.
* The label set is the model's 5 states, but the protocol only induces four
  (Engaged has no condition), so Engaged simply has no data; "Struggling"
  (Overloaded + Confused merged) is also reported, since those two are the
  hardest to separate and the tutor treats them alike.

    python -m model.train_real data/real/P01.npz --checkpoint model/checkpoints/real_fused.pt
    python -m model.train_real data/real/P01_time.npz --eeg-only --checkpoint model/checkpoints/live_eeg.pt
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from common.config import BAND_NAMES, NUM_BEHAVIOR_FEATS
from common.states import STATES
from data_gen.from_recordings import Dataset, load, merge
from model.fusion import FusionLSTMClassifier

# puzzle_rating (feature 3) is excluded by default: the collection protocol sets
# it per condition (harder band for Confused/Overloaded), so it would let the model
# "read" the experimental manipulation instead of the participant.
DEFAULT_BEHAVIOR_MASK = [1.0, 1.0, 1.0, 0.0]
EEG_ONLY_MASK = [0.0, 0.0, 0.0, 0.0]
STRUGGLING = {"Overloaded", "Confused"}


# ------------------------------------------------------------------ training
def _new_model(num_channels: int) -> FusionLSTMClassifier:
    return FusionLSTMClassifier(
        num_channels=num_channels, num_bands=len(BAND_NAMES),
        num_behavior_feats=NUM_BEHAVIOR_FEATS, num_classes=len(STATES),
    )


def fit(
    X_eeg: np.ndarray, X_beh: np.ndarray, y: np.ndarray, *, behavior_mask: list[float], use_eeg: bool = True,
    epochs: int = 40, lr: float = 2e-3, weight_decay: float = 1e-2, batch_size: int = 16, seed: int = 0,
) -> FusionLSTMClassifier:
    """Class-weighted AdamW training for a fixed number of epochs (no early
    stopping on test data; there isn't a spare validation set at this scale)."""
    torch.manual_seed(seed)
    mask = torch.tensor(behavior_mask, dtype=torch.float32)
    eeg = torch.tensor(X_eeg, dtype=torch.float32) * (1.0 if use_eeg else 0.0)
    beh = torch.tensor(X_beh, dtype=torch.float32) * mask
    labels = torch.tensor(y, dtype=torch.long)

    counts = np.bincount(y, minlength=len(STATES)).astype(np.float64)
    present = counts > 0
    weights = np.zeros(len(STATES))
    weights[present] = counts.sum() / (present.sum() * counts[present])
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))

    model = _new_model(X_eeg.shape[2])   # the headset's channel count, e.g. 7 for a DSI-7
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    n = len(labels)
    for _ in range(epochs):
        model.train()
        order = torch.randperm(n, generator=generator)
        for i in range(0, n, batch_size):
            idx = order[i:i + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(eeg[idx], beh[idx]), labels[idx])
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def predict(model: FusionLSTMClassifier, X_eeg: np.ndarray, X_beh: np.ndarray, *, behavior_mask: list[float], use_eeg: bool = True) -> np.ndarray:
    with torch.no_grad():
        eeg = torch.tensor(X_eeg, dtype=torch.float32) * (1.0 if use_eeg else 0.0)
        beh = torch.tensor(X_beh, dtype=torch.float32) * torch.tensor(behavior_mask, dtype=torch.float32)
        return torch.softmax(model(eeg, beh), dim=-1).numpy()


# ------------------------------------------------------------------- splits
def block_folds(groups: np.ndarray, y: np.ndarray, max_folds: int = 5, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """Block-wise folds in which every class that has data is represented in
    every test fold. Raises if some class has a single block (it could never
    be tested on unseen data)."""
    block_label: dict[str, int] = {}
    for g, label in zip(groups, y):
        block_label[str(g)] = int(label)
    by_label: dict[int, list[str]] = {}
    for g, label in block_label.items():
        by_label.setdefault(label, []).append(g)
    thin = {STATES[l]: len(b) for l, b in by_label.items() if len(b) < 2}
    if thin:
        raise ValueError(
            f"Block-wise cross-validation needs >= 2 usable blocks per state; too few for: {thin}. "
            "Collect more blocks for those states (or use --cv participants with >= 2 participants)."
        )
    k = min(max_folds, min(len(b) for b in by_label.values()))
    rng = np.random.default_rng(seed)
    fold_blocks: list[set[str]] = [set() for _ in range(k)]
    for blocks in by_label.values():
        blocks = list(blocks)
        rng.shuffle(blocks)
        for i, b in enumerate(blocks):
            fold_blocks[i % k].add(b)
    idx = np.arange(len(y))
    return [
        (idx[~np.isin(groups, list(test))], idx[np.isin(groups, list(test))]) for test in fold_blocks
    ]


def participant_folds(participants: np.ndarray, y: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    people = sorted(set(participants))
    if len(people) < 2:
        raise ValueError("--cv participants needs recordings from at least 2 participants.")
    idx = np.arange(len(y))
    folds = []
    for person in people:
        test = idx[participants == person]
        train = idx[participants != person]
        if len(set(y[train])) < 2:
            raise ValueError(f"Training on everyone except {person} leaves fewer than 2 states.")
        folds.append((train, test))
    return folds


# --------------------------------------------------------------- evaluation
def _metrics(y_true: np.ndarray, probs: np.ndarray, groups: np.ndarray) -> dict:
    pred = probs.argmax(axis=1)
    labels_present = sorted(set(y_true) | set(pred))
    names = [STATES[i] for i in range(len(STATES))]
    merged_true = np.array([1 if names[t] in STRUGGLING else 0 if names[t] == "Focused" else 2 for t in y_true])
    merged_pred = np.array([1 if names[p] in STRUGGLING else 0 if names[p] == "Focused" else 2 for p in pred])

    block_hits, block_total = 0, 0
    per_block = {}
    for g in sorted(set(groups)):
        sel = groups == g
        vote = np.bincount(pred[sel], minlength=len(STATES)).argmax()
        truth = int(y_true[sel][0])
        per_block[str(g)] = {"true": names[truth], "predicted": names[int(vote)], "correct": bool(vote == truth), "n": int(sel.sum())}
        block_hits += int(vote == truth)
        block_total += 1

    # Balanced accuracy: the mean per-class recall, i.e. chance is 1/(#states present)
    # no matter how lopsided the classes are.
    recalls = [float((pred[y_true == c] == c).mean()) for c in sorted(set(y_true))]
    cm = confusion_matrix(y_true, pred, labels=list(range(len(STATES))))
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(f1_score(y_true, pred, labels=sorted(set(y_true)), average="macro", zero_division=0)),
        "struggling_merged_accuracy": float(accuracy_score(merged_true, merged_pred)),
        "block_accuracy": block_hits / block_total if block_total else None,
        "blocks_correct": f"{block_hits}/{block_total}",
        "per_block": per_block,
        "confusion_matrix": cm.tolist(),
        "chance_balanced": 1.0 / len(set(y_true)),
        "labels_present": [names[i] for i in labels_present],
    }


def _flat_behavior(X_beh: np.ndarray, mask: list[float]) -> np.ndarray:
    return (X_beh * np.asarray(mask, dtype=np.float32)).reshape(len(X_beh), -1)


def _flat_eeg(X_eeg: np.ndarray) -> np.ndarray:
    # Mean band power over the sequence, per channel and band: (N, channels*bands).
    return X_eeg.mean(axis=1).reshape(len(X_eeg), -1)


def _logreg(train_X, train_y, test_X) -> np.ndarray:
    classes = sorted(set(train_y))
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
    clf.fit(train_X, train_y)
    probs = np.zeros((len(test_X), len(STATES)))
    probs[:, clf.classes_] = clf.predict_proba(test_X)
    return probs


def cross_validate(
    ds: Dataset, folds: list[tuple[np.ndarray, np.ndarray]], *, behavior_mask: list[float], epochs: int, seed: int,
    ablation: bool, log=print,
) -> dict:
    variants: dict[str, dict] = {"fused": dict(use_eeg=True, mask=behavior_mask)}
    if ablation:
        variants["eeg_only"] = dict(use_eeg=True, mask=EEG_ONLY_MASK)
        variants["behavior_only"] = dict(use_eeg=False, mask=behavior_mask)

    oof = {name: np.zeros((len(ds), len(STATES))) for name in variants}
    oof.update({name: np.zeros((len(ds), len(STATES))) for name in ("majority", "logreg_behavior", "logreg_eeg", "logreg_both")})
    covered = np.zeros(len(ds), dtype=bool)

    for f, (train, test) in enumerate(folds):
        log(f"  fold {f + 1}/{len(folds)}: train {len(train)} / test {len(test)} examples")
        covered[test] = True
        for name, cfg in variants.items():
            model = fit(ds.X_eeg[train], ds.X_behavior[train], ds.y[train], behavior_mask=cfg["mask"],
                        use_eeg=cfg["use_eeg"], epochs=epochs, seed=seed + f)
            oof[name][test] = predict(model, ds.X_eeg[test], ds.X_behavior[test], behavior_mask=cfg["mask"], use_eeg=cfg["use_eeg"])
        majority = np.bincount(ds.y[train], minlength=len(STATES)).argmax()
        oof["majority"][test, majority] = 1.0
        beh, eeg = _flat_behavior(ds.X_behavior, behavior_mask), _flat_eeg(ds.X_eeg)
        oof["logreg_behavior"][test] = _logreg(beh[train], ds.y[train], beh[test])
        oof["logreg_eeg"][test] = _logreg(eeg[train], ds.y[train], eeg[test])
        both = np.hstack([beh, eeg])
        oof["logreg_both"][test] = _logreg(both[train], ds.y[train], both[test])

    return {
        name: _metrics(ds.y[covered], probs[covered], ds.group[covered]) for name, probs in oof.items()
    }


# ------------------------------------------------------------------- report
def format_report(ds: Dataset, results: dict, cv_kind: str, folds: int, epochs: int) -> str:
    names = list(results)
    lines = [
        "# Real-data model report",
        "",
        f"* Data: **{len(ds)} examples** ({ds.alignment}-aligned) from **{len(set(ds.participant))} participant(s)**, "
        f"**{len(set(ds.group))} blocks**.",
        f"* Evaluation: **{cv_kind}** cross-validation, {folds} folds; every example is predicted once by a model that never saw its block.",
        f"* Training: {epochs} epochs, class-weighted AdamW; behaviour features used by the fused model: "
        f"{[n for n, m in zip(['correct', 'time_to_move', 'eval_loss', 'puzzle_rating'], DEFAULT_BEHAVIOR_MASK) if m]}.",
        "",
        "**How to read this.** Examples inside a block are near-duplicates, so the number of independent observations is "
        "the number of *blocks*, not examples. Block accuracy (majority vote per block) is the most honest headline; "
        "balanced accuracy corrects for uneven class sizes (chance = 1 / number of states present).",
        "",
        "| Model | Examples | Accuracy | Balanced acc. | Macro F1 | Focused / Struggling / Fatigued (merged) acc. | Blocks right |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in names:
        r = results[name]
        lines.append(
            f"| {name} | {r['n']} | {r['accuracy']:.3f} | {r['balanced_accuracy']:.3f} | {r['macro_f1']:.3f} | "
            f"{r['struggling_merged_accuracy']:.3f} | {r['blocks_correct']} |"
        )
    best = results["fused"]
    lines += ["", f"Chance level (balanced accuracy): **{best['chance_balanced']:.3f}**.", "",
              "## Fused model — confusion matrix (rows = true, columns = predicted)", "",
              "| | " + " | ".join(STATES) + " |", "|" + "---|" * (len(STATES) + 1)]
    for state, row in zip(STATES, best["confusion_matrix"]):
        lines.append(f"| **{state}** | " + " | ".join(str(v) for v in row) + " |")
    lines += ["", "## Fused model — per block", "", "| Block | True | Predicted (vote) | Examples | Correct |", "|---|---|---|---|---|"]
    for block, info in best["per_block"].items():
        lines.append(f"| {block} | {info['true']} | {info['predicted']} | {info['n']} | {'yes' if info['correct'] else 'no'} |")
    lines += ["", "## Caveats", "",
              "* Few blocks per state means wide uncertainty; treat differences of a block or two as noise.",
              "* Labels come from the experimental condition, confirmed by the participant's own self-report — not an objective ground truth.",
              "* With a small number of participants this measures within-person decodability; it does not establish generalisation to new people unless `participants` cross-validation was used.",
              "* Engaged has no induction condition and therefore no data."]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate on real recordings (see module docstring).")
    parser.add_argument("datasets", nargs="+", type=Path, help=".npz file(s) from data_gen.from_recordings")
    parser.add_argument("--checkpoint", type=Path, default=Path("model/checkpoints/real_fused.pt"))
    parser.add_argument("--eeg-only", action="store_true", help="train an EEG-only model (for the live monitor): behaviour input masked")
    parser.add_argument("--cv", choices=["blocks", "participants"], default="blocks")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ablation", action="store_true", help="also cross-validate EEG-only and behaviour-only networks")
    parser.add_argument("--report", type=Path, default=None, help="default: <checkpoint>.report.md")
    parser.add_argument("--skip-cv", action="store_true", help="just fit on everything (not recommended; no performance estimate)")
    args = parser.parse_args()

    ds = merge([load(p) for p in args.datasets])
    if len(ds) == 0:
        raise SystemExit("The dataset is empty — check the block report (python -m data_gen.from_recordings ...).")
    mask = EEG_ONLY_MASK if args.eeg_only else DEFAULT_BEHAVIOR_MASK
    print(f"{len(ds)} examples, {len(set(ds.group))} blocks, {len(set(ds.participant))} participant(s); "
          f"states present: {[STATES[i] for i in sorted(set(ds.y))]}")

    summary: dict = {}
    if not args.skip_cv:
        folds = block_folds(ds.group, ds.y, seed=args.seed) if args.cv == "blocks" else participant_folds(ds.participant, ds.y)
        print(f"Cross-validating ({args.cv}, {len(folds)} folds)…")
        results = cross_validate(ds, folds, behavior_mask=mask, epochs=args.epochs, seed=args.seed, ablation=args.ablation or args.eeg_only)
        report = format_report(ds, results, f"{args.cv}-wise", len(folds), args.epochs)
        report_path = args.report or args.checkpoint.with_name(args.checkpoint.name + ".report.md")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report)
        print(report)
        print(f"Report written to {report_path}")
        summary = {k: {m: v for m, v in r.items() if m not in ("per_block", "confusion_matrix")} for k, r in results.items()}

    print("Fitting the final model on all data…")
    model = fit(ds.X_eeg, ds.X_behavior, ds.y, behavior_mask=mask, epochs=args.epochs, seed=args.seed)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.checkpoint)
    Path(str(args.checkpoint) + ".config.json").write_text(json.dumps({
        "states": STATES, "behavior_mask": mask, "num_channels": int(ds.X_eeg.shape[2]), "alignment": ds.alignment, "epochs": args.epochs, "seed": args.seed,
        "trained_on": {
            "examples": len(ds), "blocks": len(set(ds.group)), "participants": sorted(set(ds.participant)),
            "states_present": [STATES[i] for i in sorted(set(ds.y))],
        },
        "cross_validation": summary, "trained_at": time.time(),
    }, indent=2))
    print(f"Saved {args.checkpoint} (+ .config.json)")


if __name__ == "__main__":
    main()
