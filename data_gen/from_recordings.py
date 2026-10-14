"""Turn headset recordings (eeg/recorder.py, produced by /collect) into a
labelled training dataset in the SAME schema the synthetic generator writes
(X_eeg, X_behavior, y), so model training needs no changes — plus the
group/participant bookkeeping needed to split by BLOCK instead of at random.

Nothing is annotated by hand. The recording already contains, on one clock:
  * block_start / block_end markers carrying each block's condition,
  * the participant's self-report ratings after each block,
  * "disturbance" notes the operator tapped,
  * every chess attempt (copied from the tutor's log).

Pipeline per recording:
  1. parse markers into blocks;
  2. learn the participant's resting baseline (mean/std per channel) from the
     eyes-open block and z-score the whole recording with it — the same
     normalisation the live source applies (preprocessing.realtime);
  3. decide each block's training label from its CONDITION, but only keep the
     block if the participant's own ratings confirm the condition did what it
     was meant to ("manipulation check") — a block that was supposed to be
     Overloaded but was rated effortless is dropped, not trained on;
  4. cut examples: 5-step sequences of (4 s EEG window -> band-power features,
     behaviour vector), either aligned to chess ATTEMPTS (matches how the live
     tutor builds its input; for the fused model) or to TIME (for the
     EEG-only live monitor model);
  5. drop any window that overlaps an artifact, a data gap or a disturbance note.

    python -m data_gen.from_recordings recordings/P01 --out data/real/P01.npz
"""
from __future__ import annotations
import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
import numpy as np
from chess_task.base import BehaviorEvent
from common.config import EPOCH_SECONDS, SEQ_LEN
from common.states import STATE_TO_INDEX
from data_gen.generate_dataset import behavior_to_vector
from eeg.lsl import ARTIFACT_SD
from eeg.recorder import load_recording
from preprocessing.features import extract_node_features
from preprocessing.filters import bandpass_filter
from preprocessing.realtime import RunningNormalizer

TARGET_STD = 0.5

# Condition (as run by /collect) -> the model's state label. Engaged has no
# induction condition in the protocol, so it simply has no examples; the 5-class
# output layout is kept so a model trained here is a drop-in for best.pt.
CONDITION_TO_STATE = {
    "Focused": "Focused",
    "Confused": "Confused",
    "Overloaded": "Overloaded",
    "Late_Focused": "Fatigued",
}
IGNORED_CONDITIONS = {"Practice", "Practice_Dual", "Rest", "Baseline_EO", "Baseline_EC", "Baseline_EO_end"}


@dataclass
class Thresholds:
    """Manipulation checks on the self-reports. Deliberately modest: they only have
    to reject blocks where the participant says the opposite of what the condition
    was meant to induce.

    Two report formats exist. The current /collect asks three plain questions
    (difficulty 1-4, "how did your mind feel?" one of five phrases, sleepiness 1-5);
    recordings made with the earlier six 1-7 sliders (mental_demand, frustration, ...)
    are still understood, with the `overloaded_demand ... fatigue_margin` fields below."""
    # --- current format (difficulty / mind / sleepiness)
    focused_minds: tuple = ("sharp", "pushing")          # mind feel that fits a Focused block ...
    focused_max_difficulty: int = 3                      # ... and the puzzles were not "too hard"
    confused_minds: tuple = ("lost", "overwhelmed")      # Confused: lost or overwhelmed, or ...
    confused_difficulty: int = 4                         # ... "too hard"
    overloaded_minds: tuple = ("overwhelmed", "lost", "pushing")   # Overloaded: strained, or ...
    overloaded_difficulty: int = 3                       # ... at least "hard"
    fatigue_sleepiness: int = 4                          # "tired" or worse in a late block ...
    drifting_sleepiness: int = 3                         # ... or "tired/drifting" and at least "getting tired"
    sleepiness_margin: float = 1.0                       # ... and above their early-block average by this much
    # --- earlier six-slider format
    overloaded_demand: int = 5        # mental_demand >= this, or ...
    overloaded_frustration: int = 5   # ... frustration >= this
    confused_confusion: int = 4
    focused_concentration: int = 4
    focused_max_demand: int = 5
    fatigue_min: int = 5              # absolute self-reported fatigue in a late block ...
    fatigue_margin: float = 1.0       # ... and at least this much above their early-block average
    require_ratings: bool = True


@dataclass
class Block:
    block: str
    condition: str
    kind: str
    start: float
    end: float
    aborted: bool = False
    incomplete: bool = False
    ratings: dict | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class BlockDecision:
    block: str
    condition: str
    state: str | None
    used: bool
    reason: str
    duration: float
    ratings: dict | None
    examples: int = 0


@dataclass
class Dataset:
    X_eeg: np.ndarray                 # (N, SEQ_LEN, channels, bands)
    X_behavior: np.ndarray            # (N, SEQ_LEN, 4)
    y: np.ndarray                     # (N,) state index
    group: np.ndarray                 # (N,) block uid "<participant>/<session>/<block>"
    participant: np.ndarray           # (N,)
    condition: np.ndarray             # (N,)
    t_end: np.ndarray                 # (N,) wall time of the last window's end
    report: list = field(default_factory=list)
    alignment: str = "attempts"

    def __len__(self) -> int:
        return len(self.y)


# ------------------------------------------------------------------ parsing
def parse_blocks(markers: list[dict], recording_end: float) -> list[Block]:
    """Pairs block_start/block_end markers by block id, attaches ratings."""
    open_blocks: dict[str, Block] = {}
    done: list[Block] = []
    ratings: dict[str, dict] = {}
    for m in markers:
        kind = m.get("kind")
        if kind == "block_start":
            block = Block(
                block=str(m.get("block", f"auto-{len(done)}")), condition=str(m.get("condition", "?")),
                kind=str(m.get("block_kind", "")), start=m["t"], end=recording_end,
            )
            open_blocks[block.block] = block
        elif kind == "block_end":
            block = open_blocks.pop(str(m.get("block")), None)
            if block is not None:
                block.end = m["t"]
                block.aborted = bool(m.get("aborted", False))
                done.append(block)
        elif kind == "rating":
            ratings[str(m.get("block"))] = m.get("ratings") or {}
    for block in open_blocks.values():   # recording ended / crashed mid-block
        block.incomplete = True
        done.append(block)
    for block in done:
        block.ratings = ratings.get(block.block)
    return sorted(done, key=lambda b: b.start)


def _check_report(condition: str, r: dict, th: Thresholds, early_sleepiness: float | None) -> tuple[bool, str]:
    """Manipulation check on the three-question report: does what the participant says
    about the block not contradict the condition it was meant to induce?"""
    mind = str(r.get("mind", ""))
    diff = float(r.get("difficulty", 0))
    sleep = float(r.get("sleepiness", 0))
    if condition == "Overloaded":
        ok = mind in th.overloaded_minds or diff >= th.overloaded_difficulty
        why = f"mind={mind!r}, difficulty={diff:.0f} (need mind in {list(th.overloaded_minds)} or difficulty>={th.overloaded_difficulty})"
    elif condition == "Confused":
        ok = mind in th.confused_minds or diff >= th.confused_difficulty
        why = f"mind={mind!r}, difficulty={diff:.0f} (need mind in {list(th.confused_minds)} or difficulty>={th.confused_difficulty})"
    elif condition == "Focused":
        ok = mind in th.focused_minds and diff <= th.focused_max_difficulty
        why = f"mind={mind!r}, difficulty={diff:.0f} (need mind in {list(th.focused_minds)} and difficulty<={th.focused_max_difficulty})"
    else:  # Late_Focused -> Fatigued
        tired = sleep >= th.fatigue_sleepiness or (mind == "drifting" and sleep >= th.drifting_sleepiness)
        margin_ok = early_sleepiness is None or sleep >= early_sleepiness + th.sleepiness_margin
        ok = tired and margin_ok
        why = (f"sleepiness={sleep:.0f}, mind={mind!r} (need sleepiness>={th.fatigue_sleepiness}, or 'drifting' with sleepiness>={th.drifting_sleepiness}"
               + (f", and >= early average {early_sleepiness:.1f}+{th.sleepiness_margin:g}" if early_sleepiness is not None else "") + ")")
    return ok, why


def decide_blocks(blocks: list[Block], th: Thresholds) -> list[BlockDecision]:
    """Which blocks become training data, with which label, and why not."""
    def _early(key: str):
        vals = [b.ratings[key] for b in blocks
                if b.condition in ("Focused", "Confused", "Overloaded") and b.ratings and key in b.ratings]
        return float(np.mean(vals)) if vals else None
    early_fatigue = _early("fatigue")
    early_sleepiness = _early("sleepiness")

    decisions = []
    for b in blocks:
        state = CONDITION_TO_STATE.get(b.condition)
        d = BlockDecision(b.block, b.condition, state, False, "", b.duration, b.ratings)
        decisions.append(d)
        if b.condition in IGNORED_CONDITIONS or state is None:
            d.reason = "not a training condition"
            continue
        if b.incomplete:
            d.reason = "recording ended mid-block"
            continue
        if b.aborted:
            d.reason = "block ended early by the operator"
            continue
        r = b.ratings
        if r is None:
            if th.require_ratings:
                d.reason = "no self-report ratings"
                continue
            d.used, d.reason = True, "accepted without ratings (--no-ratings-filter)"
            continue

        if "mind" in r:      # the current three-question format
            d.used, why = _check_report(b.condition, r, th, early_sleepiness)
            d.reason = ("manipulation check passed: " if d.used else "manipulation check FAILED: ") + why
            continue

        def g(key: str) -> float:
            return float(r.get(key, 0))

        if b.condition == "Overloaded":
            ok = g("mental_demand") >= th.overloaded_demand or g("frustration") >= th.overloaded_frustration
            why = f"mental_demand={g('mental_demand'):.0f}, frustration={g('frustration'):.0f} (need demand>={th.overloaded_demand} or frustration>={th.overloaded_frustration})"
        elif b.condition == "Confused":
            ok = g("confusion") >= th.confused_confusion
            why = f"confusion={g('confusion'):.0f} (need >={th.confused_confusion})"
        elif b.condition == "Focused":
            ok = g("concentration") >= th.focused_concentration and g("mental_demand") <= th.focused_max_demand
            why = f"concentration={g('concentration'):.0f}, mental_demand={g('mental_demand'):.0f} (need conc>={th.focused_concentration}, demand<={th.focused_max_demand})"
        else:  # Late_Focused -> Fatigued
            margin_ok = early_fatigue is None or g("fatigue") >= early_fatigue + th.fatigue_margin
            ok = g("fatigue") >= th.fatigue_min and margin_ok
            why = (f"fatigue={g('fatigue'):.0f} (need >={th.fatigue_min}"
                   + (f" and >= early average {early_fatigue:.1f}+{th.fatigue_margin:g}" if early_fatigue is not None else "") + ")")
        d.used = ok
        d.reason = ("manipulation check passed: " if ok else "manipulation check FAILED: ") + why
    return decisions


# ---------------------------------------------------------- signal handling
def fit_baseline(rec: dict, blocks: list[Block], settle: float = 3.0) -> tuple[dict, str]:
    """Per-channel mean/std from the eyes-open resting block (first one that is
    long enough); falls back to the whole recording if there isn't one."""
    eeg, times = rec["eeg"].astype(np.float64), rec["times"]
    for b in blocks:
        if b.condition == "Baseline_EO" and b.duration > settle + 5 and not b.incomplete:
            lo, hi = np.searchsorted(times, [b.start + settle, b.end])
            if hi - lo > 4 * rec["meta"]["sample_rate"]:
                norm = RunningNormalizer(eeg.shape[1], target_std=TARGET_STD)
                norm.update(eeg[lo:hi])
                norm.freeze()
                return norm.to_dict(), "eyes-open baseline block"
    norm = RunningNormalizer(eeg.shape[1], target_std=TARGET_STD)
    norm.update(eeg)
    norm.freeze()
    return norm.to_dict(), "whole recording (no usable baseline block)"


def _window_features(z: np.ndarray, end_idx: int, epoch_samples: int, rate: int) -> np.ndarray:
    window = z[end_idx - epoch_samples:end_idx]
    return extract_node_features(bandpass_filter(window, rate), rate).astype(np.float32)


def _window_ok(z: np.ndarray, times: np.ndarray, end_idx: int, epoch_samples: int, rate: int,
               artifact_threshold: float, note_spans: list[tuple[float, float]]) -> bool:
    start_idx = end_idx - epoch_samples
    if start_idx < 0 or end_idx > len(times):
        return False
    span = times[end_idx - 1] - times[start_idx]
    if abs(span - (epoch_samples - 1) / rate) > 0.35:       # a gap / dropout inside the window
        return False
    if np.abs(z[start_idx:end_idx]).max() > artifact_threshold:
        return False
    w0, w1 = times[start_idx], times[end_idx - 1]
    return not any(w0 <= n1 and w1 >= n0 for n0, n1 in note_spans)


# ------------------------------------------------------------------ building
def build_from_recording(
    directory: Path, alignment: str = "attempts", thresholds: Thresholds | None = None,
    time_stride: float = 4.0, settle_seconds: float = 20.0, note_before: float = 6.0, note_after: float = 4.0,
) -> Dataset:
    if alignment not in ("attempts", "time"):
        raise ValueError("alignment must be 'attempts' or 'time'")
    th = thresholds or Thresholds()
    rec = load_recording(Path(directory))
    meta, eeg, times = rec["meta"], rec["eeg"].astype(np.float64), rec["times"]
    rate = int(meta["sample_rate"])
    epoch_samples = int(EPOCH_SECONDS * rate)
    participant = str(meta["participant"])
    session = Path(directory).name
    recording_end = float(times[-1]) if len(times) else float(meta.get("ended_at") or 0.0)

    blocks = parse_blocks(rec["markers"], recording_end)
    decisions = decide_blocks(blocks, th)
    baseline, baseline_source = fit_baseline(rec, blocks)
    z = RunningNormalizer.from_dict(baseline).transform(eeg)
    artifact_threshold = ARTIFACT_SD * TARGET_STD
    notes = [(m["t"] - note_before, m["t"] + note_after) for m in rec["markers"] if m.get("kind") == "note"]

    X_eeg, X_beh, ys, groups, conds, t_ends = [], [], [], [], [], []
    attempts = sorted(rec["attempts"], key=lambda a: a["created_at"])
    by_block = {b.block: b for b in blocks}

    for d in decisions:
        if not d.used:
            continue
        b = by_block[d.block]
        settle = min(settle_seconds, 0.25 * b.duration)
        lo_t, hi_t = b.start + settle, b.end
        uid = f"{participant}/{session}/{b.block}"
        count = 0

        if alignment == "attempts":
            steps = []
            for a in attempts:
                t = a["created_at"]
                if not (lo_t + EPOCH_SECONDS <= t <= hi_t):
                    continue
                end_idx = int(np.searchsorted(times, t))
                ok = _window_ok(z, times, end_idx, epoch_samples, rate, artifact_threshold, notes)
                steps.append((t, end_idx, a, ok))
            for i in range(len(steps) - SEQ_LEN + 1):
                run = steps[i:i + SEQ_LEN]
                if not all(s[3] for s in run):
                    continue
                X_eeg.append(np.stack([_window_features(z, s[1], epoch_samples, rate) for s in run]))
                X_beh.append(np.stack([behavior_to_vector(BehaviorEvent(
                    puzzle_id=s[2]["puzzle_id"], correct=s[2]["correct"], time_to_move=s[2]["time_to_move"],
                    eval_loss=s[2]["eval_loss"], puzzle_rating=s[2]["puzzle_rating"])) for s in run]))
                ys.append(STATE_TO_INDEX[d.state]); groups.append(uid); conds.append(d.condition); t_ends.append(run[-1][0])
                count += 1
        else:  # time-aligned, EEG only
            t = lo_t + SEQ_LEN * EPOCH_SECONDS
            while t <= hi_t:
                ends = [t - (SEQ_LEN - 1 - j) * EPOCH_SECONDS for j in range(SEQ_LEN)]
                idxs = [int(np.searchsorted(times, e)) for e in ends]
                if all(_window_ok(z, times, i, epoch_samples, rate, artifact_threshold, notes) for i in idxs):
                    X_eeg.append(np.stack([_window_features(z, i, epoch_samples, rate) for i in idxs]))
                    X_beh.append(np.zeros((SEQ_LEN, 4), dtype=np.float32))
                    ys.append(STATE_TO_INDEX[d.state]); groups.append(uid); conds.append(d.condition); t_ends.append(t)
                    count += 1
                t += time_stride
        d.examples = count
        if count == 0 and d.used:
            d.used = False
            d.reason += " — but no clean example windows could be cut (artifacts / too few attempts)"

    channels = eeg.shape[1]
    n = len(ys)
    ds = Dataset(
        X_eeg=np.array(X_eeg, dtype=np.float32).reshape(n, SEQ_LEN, channels, 3) if n else np.zeros((0, SEQ_LEN, channels, 3), np.float32),
        X_behavior=np.array(X_beh, dtype=np.float32).reshape(n, SEQ_LEN, 4) if n else np.zeros((0, SEQ_LEN, 4), np.float32),
        y=np.array(ys, dtype=np.int64), group=np.array(groups, dtype=str),
        participant=np.array([participant] * n, dtype=str), condition=np.array(conds, dtype=str),
        t_end=np.array(t_ends, dtype=np.float64),
        report=[{**asdict(d), "participant": participant, "session": session, "baseline": baseline_source} for d in decisions],
        alignment=alignment,
    )
    return ds


def merge(datasets: list[Dataset]) -> Dataset:
    datasets = [d for d in datasets if len(d)] or datasets[:1]
    cat = lambda name: np.concatenate([getattr(d, name) for d in datasets], axis=0)
    return Dataset(
        X_eeg=cat("X_eeg"), X_behavior=cat("X_behavior"), y=cat("y"), group=cat("group"),
        participant=cat("participant"), condition=cat("condition"), t_end=cat("t_end"),
        report=[r for d in datasets for r in d.report], alignment=datasets[0].alignment,
    )


def find_recordings(paths: list[Path]) -> list[Path]:
    """Each path may be one recording directory, or a participant/root folder
    containing recording directories."""
    found = []
    for p in map(Path, paths):
        if (p / "meta.json").exists():
            found.append(p)
        else:
            found.extend(sorted(m.parent for m in p.rglob("meta.json")))
    return found


def save(ds: Dataset, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path, X_eeg=ds.X_eeg, X_behavior=ds.X_behavior, y=ds.y, group=ds.group,
        participant=ds.participant, condition=ds.condition, t_end=ds.t_end, alignment=np.array(ds.alignment),
    )
    path.with_suffix(".report.json").write_text(json.dumps(ds.report, indent=2))


def load(path: Path) -> Dataset:
    data = np.load(path, allow_pickle=False)
    report_path = Path(path).with_suffix(".report.json")
    return Dataset(
        X_eeg=data["X_eeg"], X_behavior=data["X_behavior"], y=data["y"], group=data["group"],
        participant=data["participant"], condition=data["condition"], t_end=data["t_end"],
        report=json.loads(report_path.read_text()) if report_path.exists() else [], alignment=str(data["alignment"]),
    )


def summarize(ds: Dataset) -> str:
    from common.states import STATES
    lines = [f"{len(ds)} examples ({ds.alignment}-aligned) from {len(set(ds.participant))} participant(s)", "", "Blocks:"]
    for r in ds.report:
        if r["condition"] in IGNORED_CONDITIONS:
            continue
        flag = "USED   " if r["used"] else "dropped"
        lines.append(f"  [{flag}] {r['participant']}/{r['block']:<8} {r['condition']:<13} -> {str(r['state']):<10} "
                     f"{r['duration']:5.0f}s  {r['examples']:>3} ex  {r['reason']}")
    lines += ["", "Examples per state:"]
    for i, s in enumerate(STATES):
        n = int((ds.y == i).sum())
        blocks = len({g for g, yy in zip(ds.group, ds.y) if yy == i})
        lines.append(f"  {s:<11} {n:>5} examples from {blocks} block(s)")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a labelled dataset from /collect recordings.")
    parser.add_argument("paths", nargs="+", type=Path, help="recording dir(s) or a participant folder")
    parser.add_argument("--out", type=Path, default=Path("data/real/dataset.npz"))
    parser.add_argument("--alignment", choices=["attempts", "time"], default="attempts",
                        help="attempts: one step per chess attempt (fused model); time: fixed 4 s steps, EEG only (live monitor)")
    parser.add_argument("--time-stride", type=float, default=4.0)
    parser.add_argument("--no-ratings-filter", action="store_true", help="keep blocks even without/against self-reports")
    args = parser.parse_args()

    th = Thresholds(require_ratings=not args.no_ratings_filter)
    recordings = find_recordings(args.paths)
    if not recordings:
        raise SystemExit("No recordings (meta.json) found under the given paths.")
    parts = [build_from_recording(r, alignment=args.alignment, thresholds=th, time_stride=args.time_stride) for r in recordings]
    ds = merge(parts)
    save(ds, args.out)
    print(summarize(ds))
    print(f"\nWrote {args.out} (+ {args.out.with_suffix('.report.json')})")


if __name__ == "__main__":
    main()
