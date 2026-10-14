"""Synthesise a complete /collect-style recording on disk — through the REAL
signal path (fake 250 Hz microvolt headset -> eeg.lsl conditioning/resampling)
and the REAL recorder — so the whole pipeline downstream of the headset
(from_recordings -> train_real -> live) can be built, tested and rehearsed
before a single participant exists.

    python -m data_gen.fake_recording --participant SIM01 --block-seconds 60

It is clearly synthetic (the recording's meta.json source kind is "fake"), and
exists for development and demo-rehearsal only: models trained on it say
nothing about real brains. `--separation` scales how different the states look
(1.0 = the simulator's clean states; lower = harder, more like real EEG).
"""
from __future__ import annotations
import argparse
import random
from pathlib import Path
import numpy as np
from common.states import STATES
from data_gen.virtual_player import VirtualPlayer
from eeg.fake_lsl import make_generator
from eeg.lsl import LSLEEGSource
from eeg.recorder import DEFAULT_ROOT, SessionRecorder
from eeg.simulated import STATE_BAND_PARAMS

NATIVE_RATE = 250.0
CONDITION_STATE = {
    "Focused": "Focused", "Confused": "Confused", "Overloaded": "Overloaded", "Late_Focused": "Fatigued",
    "Practice": "Focused", "Practice_Dual": "Overloaded", "Baseline_EO": "Engaged", "Baseline_EC": "Engaged",
    "Baseline_EO_end": "Engaged", "Rest": "Engaged",
}
# Typical self-reports for the state each condition induces (1-7), +/- noise.
RATING_MEANS = {
    "Focused":    dict(mental_demand=3.5, effort=4, frustration=2, confusion=2, fatigue=2.5, concentration=6),
    "Confused":   dict(mental_demand=5, effort=5, frustration=4, confusion=6, fatigue=3, concentration=3.5),
    "Overloaded": dict(mental_demand=6.3, effort=6, frustration=5.5, confusion=4, fatigue=3.5, concentration=3),
    "Late_Focused": dict(mental_demand=4, effort=4.5, frustration=3, confusion=3, fatigue=6, concentration=3.5),
}


def _blend_params(state: str, separation: float) -> dict:
    mid = {k: float(np.mean([p[k] for p in STATE_BAND_PARAMS.values()])) for k in ("theta", "alpha")}
    return {k: mid[k] + separation * (STATE_BAND_PARAMS[state][k] - mid[k]) for k in ("theta", "alpha")}


def build_fake_recording(
    participant: str = "SIM01", root: Path = DEFAULT_ROOT, block_seconds: float = 60.0, seed: int = 0,
    separation: float = 1.0, inconsistent_condition_index: int | None = None, start_time: float = 1_800_000_000.0,
    shuffle_conditions: bool = True, channels: int = 8, native_rate: float = NATIVE_RATE,
) -> Path:
    rng = random.Random(seed)
    order = ["Focused", "Focused", "Confused", "Confused", "Overloaded", "Overloaded"]
    if shuffle_conditions:
        rng.shuffle(order)
    steps = [("Baseline_EO", "baseline", 30.0), ("Baseline_EC", "baseline", 15.0), ("Practice", "task", 20.0)]
    steps += [(c, "task", block_seconds) for c in order] + [("Late_Focused", "task", block_seconds)] * 2
    steps += [("Baseline_EO_end", "baseline", 15.0)]

    source = LSLEEGSource(calibration_seconds=10_000)   # never freezes: we only want the conditioned microvolts
    source._setup(channels, native_rate)
    generate, gen_state = make_generator(channels, native_rate, 50.0, seed=seed)
    np_rng = np.random.default_rng(seed)

    # make_generator reads STATE_BAND_PARAMS; swap in blended params for `separation`.
    original = {k: dict(v) for k, v in STATE_BAND_PARAMS.items()}
    for state in STATES:
        STATE_BAND_PARAMS[state].update(_blend_params(state, separation))

    recorder = SessionRecorder(participant, 128, channels, root=root, label="fake", source_info={"kind": "fake"})
    recorder.directory.mkdir(parents=True, exist_ok=True)
    clock = start_time
    attempts = []
    try:
        recorder.add_marker("recording_start", _t=clock, participant=participant)
        task_index = 0
        for index, (condition, kind, seconds) in enumerate(steps):
            block_id = f"b{index}" if kind == "task" else f"{condition}"
            if condition == "Practice":
                block_id = "practice-1"
            gen_state["name"] = CONDITION_STATE[condition]
            recorder.add_marker("block_start", _t=clock, block=block_id, condition=condition, block_kind=kind, seconds=seconds)
            player = VirtualPlayer(CONDITION_STATE[condition], seed=seed * 1000 + index)
            next_attempt = clock + 6.0
            end = clock + seconds
            t = clock
            while t < end - 1e-9:
                n_raw = int(native_rate * 0.25)
                raw = generate(n_raw)
                stamps = t + np.arange(n_raw) / native_rate
                chunk = source.process_raw(raw, stamps)
                if chunk is not None:
                    recorder.write_chunk(t + 0.25, chunk)
                t += 0.25
                if kind == "task" and t >= next_attempt and condition != "Practice":
                    event = player.attempt(f"fake-{index}-{len(attempts)}", int(np_rng.integers(900, 1500)))
                    attempts.append({
                        "session_id": f"fake-{block_id}", "mode": "puzzle", "puzzle_id": event.puzzle_id, "correct": event.correct,
                        "time_to_move": event.time_to_move, "eval_loss": event.eval_loss, "puzzle_rating": event.puzzle_rating,
                        "predicted_state": CONDITION_STATE[condition], "confidence": 0.8, "difficulty_delta": 0.0,
                        "show_hint": False, "pacing_delay": 0.0, "sense_to_adapt_latency": 0.01, "created_at": t,
                    })
                    next_attempt = t + max(3.0, event.time_to_move)
            clock = end
            recorder.add_marker("block_end", _t=clock, block=block_id, condition=condition, aborted=False)
            if kind == "task" and condition not in ("Practice", "Practice_Dual"):
                means = dict(RATING_MEANS[condition])
                if inconsistent_condition_index is not None and task_index == inconsistent_condition_index:
                    means = dict(RATING_MEANS["Focused"]) if condition != "Focused" else dict(RATING_MEANS["Overloaded"])
                ratings = {k: int(np.clip(round(v + np_rng.normal(0, 0.6)), 1, 7)) for k, v in means.items()}
                recorder.add_marker("rating", _t=clock + 5, block=block_id, condition=condition, ratings=ratings)
                clock += 10
                task_index += 1
        recorder.add_marker("recording_stop", _t=clock)
    finally:
        for state, params in original.items():
            STATE_BAND_PARAMS[state].update(params)
    recorder.started_at = start_time
    recorder.stop(attempts=attempts)
    return recorder.directory


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a synthetic /collect-style recording (development only).")
    parser.add_argument("--participant", default="SIM01")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--block-seconds", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--separation", type=float, default=1.0)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--native-rate", type=float, default=NATIVE_RATE)
    args = parser.parse_args()
    print(build_fake_recording(
        args.participant, args.root, args.block_seconds, args.seed, args.separation,
        channels=args.channels, native_rate=args.native_rate,
    ))


if __name__ == "__main__":
    main()
