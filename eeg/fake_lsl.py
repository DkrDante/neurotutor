"""A fake EEG headset that publishes a realistic-ish LSL stream, so the whole
real-headset path (eeg.lsl -> hub -> /live -> recorder) can be built and
tested before any hardware arrives.

    python -m eeg.fake_lsl                       # 8 ch, 250 Hz, Focused
    python -m eeg.fake_lsl --cycle 20            # rotate through the 5 states every 20 s
    python -m eeg.fake_lsl --rate 256 --channels 4 --mains 60
    python -m eeg.fake_lsl --dsi7                # 7 labelled EEG ch @ 300 Hz + trigger/reference/aux channels

Unlike eeg.simulated (clean, unit-scale, 128 Hz), this deliberately looks like
a real device: microvolts, a large per-channel DC offset with slow drift, 50/60 Hz
mains hum, 1/f-ish background, occasional blink artifacts, and a native rate that
isn't the model's 128 Hz — exactly what eeg.lsl / preprocessing.realtime exist
to handle. State differences reuse the simulator's theta/alpha profile (scaled
to microvolts) so band power visibly moves when the state changes.
"""
from __future__ import annotations
import argparse
import time
import numpy as np
from eeg.simulated import STATE_BAND_PARAMS
from common.states import STATES

UV_PER_UNIT = 40.0  # simulator amplitude 1.0 -> 40 uV


def make_generator(num_channels: int, rate: float, mains_hz: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    dc = rng.uniform(-3000, 3000, size=num_channels)           # electrode offsets
    gains = rng.uniform(0.8, 1.2, size=num_channels)
    phase = rng.uniform(0, 2 * np.pi, size=(num_channels, 2))
    state = {"t": 0.0, "name": "Focused", "pink": np.zeros(num_channels), "blink_until": -1.0}

    def generate(n: int) -> np.ndarray:
        t = state["t"] + np.arange(n) / rate
        params = STATE_BAND_PARAMS[state["name"]]
        out = np.empty((n, num_channels))
        for ch in range(num_channels):
            theta = params["theta"] * np.sin(2 * np.pi * 6.0 * t + phase[ch, 0])
            alpha = params["alpha"] * np.sin(2 * np.pi * 10.0 * t + phase[ch, 1])
            brain = gains[ch] * UV_PER_UNIT * (theta + alpha)
            # Leaky random walk ~ 1/f background plus white noise.
            walk = np.cumsum(rng.normal(0, 0.8, size=n)) * 0.02
            state["pink"][ch] = 0.995 * state["pink"][ch] + walk[-1]
            noise = rng.normal(0, 4.0, size=n) + state["pink"][ch]
            hum = 12.0 * np.sin(2 * np.pi * mains_hz * t)
            drift = 60.0 * np.sin(2 * np.pi * 0.05 * t + ch)
            out[:, ch] = dc[ch] + drift + hum + brain + noise
        # Occasional eye blink: a slow ~150 uV bump, strongest on the first channels.
        if state["blink_until"] < state["t"] and rng.random() < 0.01 * n / rate * 4:
            state["blink_until"] = state["t"] + 0.3
        if state["blink_until"] >= state["t"]:
            bump = 150.0 * np.hanning(n)
            weights = np.linspace(1.0, 0.2, num_channels)
            out += bump[:, None] * weights[None, :]
        state["t"] += n / rate
        return out

    return generate, state


DSI7_LABELS = ["Fz", "F3", "F4", "C3", "C4", "Pz", "P4"]
DSI7_EXTRAS = ["X1", "X2", "X3", "TRG", "CM"]


def extra_channel(label: str, n: int, rng: np.random.Generator) -> np.ndarray:
    """What a non-EEG channel of a real device looks like: a constant trigger line,
    a big drifting common-mode channel, small aux/accelerometer wiggles."""
    if label.upper() == "TRG":
        return np.zeros(n)
    if label.upper() == "CM":
        return 5000 + rng.normal(0, 400, n)
    return rng.normal(0, 0.05, n)


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish a fake EEG headset over LSL.")
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--rate", type=float, default=250.0)
    parser.add_argument("--mains", type=float, default=50.0)
    parser.add_argument("--state", choices=STATES, default="Focused")
    parser.add_argument("--cycle", type=float, default=0.0, help="rotate states every N seconds (0 = fixed)")
    parser.add_argument("--name", default="FakeEEG")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--labels", default="", help="comma-separated EEG channel labels (default Ch1..ChN)")
    parser.add_argument("--extra", default="", help="comma-separated NON-EEG channel labels appended after the EEG (e.g. X1,TRG,CM)")
    parser.add_argument("--dsi7", action="store_true", help="preset: 7 labelled EEG channels @ 300 Hz plus DSI-style extra channels")
    args = parser.parse_args()

    if args.dsi7:
        args.channels, args.rate, args.name = 7, 300.0, "DSI-7-fake"
        args.labels, args.extra = ",".join(DSI7_LABELS), ",".join(DSI7_EXTRAS)
    labels = [l.strip() for l in args.labels.split(",") if l.strip()] or [f"Ch{i + 1}" for i in range(args.channels)]
    if len(labels) != args.channels:
        raise SystemExit(f"--labels has {len(labels)} entries but --channels is {args.channels}")
    extras = [l.strip() for l in args.extra.split(",") if l.strip()]
    total = args.channels + len(extras)

    import pylsl
    info = pylsl.StreamInfo(args.name, "EEG", total, args.rate, pylsl.cf_float32, f"fake-{args.name}")
    channels = info.desc().append_child("channels")
    for label in labels:
        ch = channels.append_child("channel")
        ch.append_child_value("label", label)
        ch.append_child_value("unit", "microvolts")
        ch.append_child_value("type", "EEG")
    for label in extras:
        ch = channels.append_child("channel")
        ch.append_child_value("label", label)
        ch.append_child_value("unit", "microvolts")
        ch.append_child_value("type", "Trigger" if label.upper() == "TRG" else "Reference" if label.upper() == "CM" else "Aux")
    outlet = pylsl.StreamOutlet(info)

    generate, state = make_generator(args.channels, args.rate, args.mains, seed=args.seed)
    extra_rng = np.random.default_rng(args.seed + 1)
    state["name"] = args.state
    extra_note = f" + extra {extras}" if extras else ""
    print(f"Publishing '{args.name}' (type EEG): {args.channels} ch {labels}{extra_note} @ {args.rate:g} Hz, state={args.state}. Ctrl+C to stop.")

    chunk_samples = max(1, int(args.rate * 0.05))
    started = time.time()
    next_push = started
    last_state_idx = STATES.index(args.state)
    try:
        while True:
            if args.cycle > 0:
                idx = (STATES.index(args.state) + int((time.time() - started) // args.cycle)) % len(STATES)
                if idx != last_state_idx:
                    last_state_idx = idx
                    state["name"] = STATES[idx]
                    print(f"[{time.time() - started:6.1f}s] state -> {STATES[idx]}")
            block = generate(chunk_samples)
            if extras:
                block = np.hstack([block] + [extra_channel(l, chunk_samples, extra_rng)[:, None] for l in extras])
            outlet.push_chunk(block.astype(np.float32).tolist())
            next_push += chunk_samples / args.rate
            delay = next_push - time.time()
            if delay > 0:
                time.sleep(delay)
    except KeyboardInterrupt:
        print("Stopped.")


if __name__ == "__main__":
    main()
