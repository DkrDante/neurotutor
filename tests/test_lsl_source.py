"""LSLEEGSource: the signal path (no network) and a real in-process LSL loop."""
from __future__ import annotations
import threading
import time
import numpy as np
import pytest
from common.config import NUM_CHANNELS, SAMPLE_RATE
from eeg.fake_lsl import make_generator
from eeg.lsl import LSLEEGSource, LSLStreamNotFound
from preprocessing.features import extract_node_features
from preprocessing.filters import bandpass_filter


def _fed_source(seconds: float, native: float = 250.0, state: str = "Focused", **kwargs) -> tuple[LSLEEGSource, list]:
    source = LSLEEGSource(calibration_seconds=5, **kwargs)
    source._setup(NUM_CHANNELS, native)
    generate, gen_state = make_generator(NUM_CHANNELS, native, 50.0)
    gen_state["name"] = state
    chunks, t = [], 0.0
    per_tick = int(native * 0.05)
    for _ in range(int(round(seconds / (per_tick / native)))):
        raw = generate(per_tick)
        stamps = t + np.arange(len(raw)) / native
        t += len(raw) / native
        chunk = source.process_raw(raw, stamps)
        if chunk is not None:
            chunks.append(chunk)
    return source, chunks


def test_signal_path_outputs_model_rate_and_training_scale():
    source, chunks = _fed_source(12)
    samples = np.vstack([c.samples for c in chunks])
    assert abs(len(samples) - 12 * SAMPLE_RATE) < 10                   # 250 Hz -> ~128 Hz
    assert not source.calibrating                                      # 5 s baseline finished
    tail = samples[-4 * SAMPLE_RATE:]
    assert np.all(np.abs(tail.mean(axis=0)) < 0.2)                     # DC removed
    assert np.all((tail.std(axis=0) > 0.2) & (tail.std(axis=0) < 0.9))  # ~ the simulator's scale
    assert chunks[0].uv is not None and chunks[0].uv.shape == chunks[0].samples.shape
    assert np.abs(chunks[-1].uv).max() > 5                             # uv is real microvolts, not normalised


def test_states_differ_in_band_power_after_the_pipeline():
    """Different simulated brain states must still look different after
    conditioning — otherwise the real-headset path would erase the signal."""
    def mean_features(state):
        _, chunks = _fed_source(14, state=state)
        window = np.vstack([c.samples for c in chunks])[-4 * SAMPLE_RATE:]
        return extract_node_features(bandpass_filter(window, SAMPLE_RATE), SAMPLE_RATE).mean(axis=0)
    focused, overloaded = mean_features("Focused"), mean_features("Overloaded")
    assert overloaded[0] > focused[0] * 1.5   # theta rises when Overloaded
    assert focused[1] > overloaded[1] * 1.5   # alpha is higher when Focused


def test_quality_is_reported_and_flat_channel_is_caught():
    source = LSLEEGSource(calibration_seconds=2)
    source._setup(4, 250.0)
    rng = np.random.default_rng(0)
    for k in range(40):
        raw = rng.normal(0, 15, size=(25, 4)) + 3000
        raw[:, 2] = 3000.0  # a lifted electrode reads a constant
        source.process_raw(raw, k * 0.1 + np.arange(25) / 250.0)
    assert [q["status"] for q in source.latest_quality] == ["ok", "ok", "flat", "ok"]


def test_channel_selection_and_validation():
    source = LSLEEGSource(channel_indices=[1, 3, 5])
    source._setup(8, 250.0)
    assert source.num_channels == 3
    with pytest.raises(LSLStreamNotFound):
        LSLEEGSource(channel_indices=[0, 9])._setup(8, 250.0)
    wide = LSLEEGSource(max_channels=8)
    wide._setup(16, 250.0)
    assert wide.num_channels == 8


def test_artifact_threshold_scales_with_target_std():
    assert LSLEEGSource(target_std=0.5, artifact_sd=10).artifact_threshold == 5.0


def test_no_stream_found_raises_a_helpful_error():
    pytest.importorskip("pylsl")
    source = LSLEEGSource(stream_name="definitely-not-a-real-stream-xyz", resolve_timeout=0.5)
    with pytest.raises(LSLStreamNotFound, match="No LSL stream"):
        source.connect()


def test_end_to_end_over_a_real_lsl_outlet():
    """A genuine pylsl outlet -> LSLEEGSource.connect()/poll(), in-process."""
    pylsl = pytest.importorskip("pylsl")
    name = f"pytest-eeg-{time.time_ns()}"
    info = pylsl.StreamInfo(name, "EEG", 8, 250.0, pylsl.cf_float32, name)
    outlet = pylsl.StreamOutlet(info)
    stop = threading.Event()

    def feed():
        generate, _ = make_generator(8, 250.0, 50.0)
        nxt = time.time()
        while not stop.is_set():
            outlet.push_chunk(generate(12).astype(np.float32).tolist())
            nxt += 12 / 250.0
            time.sleep(max(0.0, nxt - time.time()))

    thread = threading.Thread(target=feed, daemon=True)
    thread.start()
    try:
        source = LSLEEGSource(stream_name=name, resolve_timeout=5.0, calibration_seconds=2)
        info_dict = source.connect()
        assert info_dict["native_rate"] == 250.0 and source.num_channels == 8
        received = 0
        deadline = time.time() + 6
        while time.time() < deadline and received < 400:
            chunk = source.poll()
            if chunk is not None:
                received += chunk.samples.shape[0]
            time.sleep(0.05)
        source.close()
        assert received >= 400
    finally:
        stop.set()
        thread.join(timeout=2)


class _Info:
    def __init__(self, name, kind="StreamData", channels=8, rate=250.0):
        self._n, self._t, self._c, self._r = name, kind, channels, rate

    def name(self): return self._n
    def type(self): return self._t
    def channel_count(self): return self._c
    def nominal_srate(self): return self._r


def test_rank_streams_picks_eeg_and_skips_ppg_imu_markers():
    from eeg.lsl import rank_eeg_streams
    infos = [
        _Info("neurot_PPG", channels=2, rate=100.0), _Info("neurot_IMU", channels=7, rate=100.0),
        _Info("neurot_Marker", channels=1, rate=0.0), _Info("neurot_EEG"), _Info("EEG"),
    ]
    ranked = [i.name() for i in rank_eeg_streams(infos)]
    assert ranked == ["neurot_EEG", "EEG"]          # device-prefixed stream first; PPG/IMU/marker dropped


def test_rank_streams_prefers_type_eeg_and_handles_none():
    from eeg.lsl import rank_eeg_streams
    assert rank_eeg_streams([]) == []
    assert rank_eeg_streams([_Info("only_markers", rate=0.0)]) == []
    ranked = rank_eeg_streams([_Info("Signal"), _Info("Amp", kind="EEG")])
    assert ranked[0].name() == "Amp"


def test_resolve_labels_names_the_h100_and_respects_overrides():
    from eeg.lsl import H100_LABELS, resolve_labels
    generic = [f"Ch{i}" for i in range(1, 9)]
    assert resolve_labels(generic, 250.0) == (H100_LABELS, "profile:H100")
    assert resolve_labels(generic, 300.0) == (generic, "stream")                      # other rate: not this cap
    assert resolve_labels(generic[:7], 250.0) == (generic[:7], "stream")              # other channel count
    real = ["Fz", "Cz", "Pz", "P3", "P4", "O1", "O2", "C3"]
    assert resolve_labels(real, 250.0) == (real, "stream")                            # real labels are never replaced
    assert resolve_labels(generic, 250.0, override=list("ABCDEFGH")) == (list("ABCDEFGH"), "manual")
    with pytest.raises(LSLStreamNotFound):
        resolve_labels(generic, 250.0, override=["A", "B"])
