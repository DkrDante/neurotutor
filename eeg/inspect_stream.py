"""Headset-day check, no server needed:

    python -m eeg.inspect_stream                 # find the first LSL stream of type EEG
    python -m eeg.inspect_stream --name DSI-7    # or by name
    python -m eeg.inspect_stream --seconds 8 --mains 60

Prints what the stream really is — name, native rate, every channel's label/type,
which channels this app will use (trigger/reference/aux channels are dropped), the
raw value scale per channel (so you can see at a glance whether it is in microvolts)
and a per-channel signal-quality verdict after filtering.
"""
from __future__ import annotations
import argparse
import time
import numpy as np
from eeg.lsl import LSLEEGSource, LSLStreamNotFound


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a live LSL EEG stream.")
    parser.add_argument("--name", default=None, help="match by stream name instead of type EEG")
    parser.add_argument("--type", default="EEG")
    parser.add_argument("--seconds", type=float, default=6.0, help="how long to listen")
    parser.add_argument("--mains", type=float, default=50.0, help="mains frequency to notch (50 or 60)")
    parser.add_argument("--unit-scale", type=float, default=1.0, help="multiply raw values by this to get microvolts")
    args = parser.parse_args()

    source = LSLEEGSource(stream_type=args.type, stream_name=args.name, notch_hz=args.mains, unit_scale=args.unit_scale, calibration_seconds=2)
    try:
        info = source.connect()
    except LSLStreamNotFound as err:
        raise SystemExit(f"\n{err}\n")

    print(f"\nStream '{info['name']}' (type {info['type']}): {info['total_channels']} channels @ {info['native_rate']:g} Hz "
          f"-> resampled to {info['model_rate']} Hz")
    print(f"Listening for {args.seconds:g} s …\n")

    raw_chunks = []
    deadline = time.time() + args.seconds
    while time.time() < deadline:
        pulled, _ = source._inlet.pull_chunk(timeout=0.0, max_samples=int(source.native_rate * 2))
        if pulled:
            raw_chunks.append(np.asarray(pulled, dtype=np.float64))
            # Also push through the real signal path so the quality verdict is the app's own.
            stamps = np.full(len(pulled), time.time()) - np.arange(len(pulled))[::-1] / source.native_rate
            source.process_raw(np.asarray(pulled, dtype=np.float64), stamps)
        time.sleep(0.05)
    if not raw_chunks:
        raise SystemExit("The stream resolved but delivered no samples. Is the headset actually streaming?")

    raw = np.vstack(raw_chunks)
    quality = {idx: q for idx, q in zip(source._indices, source.latest_quality)}
    total_samples = raw.shape[0]
    print(f"Received {total_samples} samples ({total_samples / args.seconds:.0f}/s).\n")
    print(f"{'#':>3}  {'label':<10} {'type':<10} {'used':<5} {'raw mean':>12} {'raw std':>10}  quality")
    for i in range(info["total_channels"]):
        used = i in source._indices
        q = quality.get(i)
        verdict = f"{q['status']} ({q['rms']:.1f} µV RMS)" if q else "—"
        print(f"{i:>3}  {info['all_channel_labels'][i]:<10} {info['channel_types'][i] or '-':<10} {'YES' if used else 'no':<5} "
              f"{raw[:, i].mean():>12.4g} {raw[:, i].std():>10.4g}  {verdict}")
    print(f"\nUsed by the app: {source.num_channels} channels {source.channel_names}")
    if source.unit_warning:
        print(f"\nWARNING: {source.unit_warning}")
    bad = [source.channel_names[k] for k, q in enumerate(source.latest_quality) if q["status"] in ("flat", "noisy")]
    if bad:
        print(f"Check contact / noise on: {bad}")
    print("\nIf this looks right, start the app (./run.sh) and connect on /live.\n")
    source.close()


if __name__ == "__main__":
    main()
