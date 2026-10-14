"""Channel-count-agnostic pipeline: a DSI-7-style stream (7 EEG channels @ 300 Hz,
plus trigger/reference/aux channels) must flow from the LSL adapter through
recording, training and serving without anything assuming 8 channels."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
import web.server as server_module
from accounts.storage import AccountStore
from common.config import BAND_NAMES, NUM_BEHAVIOR_FEATS
from common.states import STATES
from data_gen.fake_recording import build_fake_recording
from data_gen.from_recordings import build_from_recording, merge, save
from eeg.fake_lsl import DSI7_EXTRAS, DSI7_LABELS, extra_channel, make_generator
from eeg.hub import EEGHub
from eeg.live_state import LiveStateEstimator
from eeg.lsl import LSLEEGSource, LSLStreamNotFound, select_channels
from eeg.simulated import SimulatedEEGSource
from model import train_real
from model.fusion import FusionLSTMClassifier
from model.inference import NullStatePredictor, StatePredictor, checkpoint_num_channels
from storage.db import SessionStore

DSI_LABELS = DSI7_LABELS + DSI7_EXTRAS
DSI_TYPES = ["EEG"] * 7 + ["Aux", "Aux", "Aux", "Trigger", "Reference"]


# ------------------------------------------------------------ channel selection
def test_auto_selection_keeps_only_the_electrodes():
    assert select_channels(DSI_LABELS, DSI_TYPES) == list(range(7))
    assert select_channels(DSI_LABELS) == list(range(7))           # labels alone are enough
    assert select_channels(["Fz", "Pz", "A1", "A2", "TRG"]) == [0, 1]   # ear references / trigger excluded


def test_explicit_names_beat_indices_and_are_case_insensitive_and_ordered():
    assert select_channels(DSI_LABELS, names=["pz", "FZ"], indices=[0, 1, 2]) == [5, 0]
    assert select_channels(DSI_LABELS, indices=[3, 1]) == [3, 1]


def test_selection_errors_are_specific():
    with pytest.raises(LSLStreamNotFound, match="not in the stream"):
        select_channels(DSI_LABELS, names=["Oz"])
    with pytest.raises(LSLStreamNotFound, match="out of range"):
        select_channels(DSI_LABELS, indices=[40])


def test_unlabelled_or_oddly_labelled_streams_fall_back_to_everything():
    assert select_channels(["Ch1", "Ch2", "Ch3"]) == [0, 1, 2]
    assert select_channels(["TRG", "CM"]) == [0, 1]      # nothing looks like EEG: don't return an empty set
    assert select_channels([f"Ch{i}" for i in range(12)], max_channels=8) == list(range(8))


# ------------------------------------------------------- the DSI-style signal path
def _dsi_source(seconds=14, **kwargs):
    source = LSLEEGSource(calibration_seconds=5, **kwargs)
    source._setup(len(DSI_LABELS), 300.0, DSI_LABELS, DSI_TYPES)
    generate, _ = make_generator(7, 300.0, 50.0)
    rng = np.random.default_rng(0)
    chunks, t = [], 0.0
    for _ in range(int(seconds / 0.05)):
        eeg = generate(15)
        raw = np.hstack([eeg] + [extra_channel(l, 15, rng)[:, None] for l in DSI7_EXTRAS])
        chunk = source.process_raw(raw, t + np.arange(15) / 300.0)
        t += 15 / 300.0
        if chunk is not None:
            chunks.append(chunk)
    return source, chunks


def test_dsi_style_stream_yields_seven_clean_labelled_channels():
    source, chunks = _dsi_source()
    assert source.num_channels == 7 and source.channel_names == DSI7_LABELS
    samples = np.vstack([c.samples for c in chunks])
    assert samples.shape[1] == 7
    assert abs(len(samples) - 14 * 128) < 12                                  # 300 Hz -> ~128 Hz
    tail = samples[-3 * 128:]
    assert np.all((tail.std(axis=0) > 0.2) & (tail.std(axis=0) < 0.9))       # training scale: the huge CM channel never leaks in
    assert all(q["status"] == "ok" for q in source.latest_quality) and source.unit_warning is None


def test_unit_warning_when_the_stream_is_in_volts_and_fixed_by_unit_scale():
    def run(unit_scale):
        source = LSLEEGSource(calibration_seconds=2, unit_scale=unit_scale)
        source._setup(3, 300.0, ["Fz", "Cz", "Pz"], ["EEG"] * 3)
        rng = np.random.default_rng(1)
        for k in range(60):
            raw = rng.normal(0, 20e-6, size=(30, 3)) + 1e-3     # 20 uV of EEG expressed in VOLTS
            source.process_raw(raw, k * 0.1 + np.arange(30) / 300.0)
        return source
    assert "microvolts" in (run(1.0).unit_warning or "")
    fixed = run(1e6)
    assert fixed.unit_warning is None and all(q["status"] == "ok" for q in fixed.latest_quality)


# -------------------------------------------- 7-channel data -> trained model -> serving
@pytest.fixture(scope="module")
def dsi_dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("dsi")
    parts = [build_from_recording(build_fake_recording(p, root, block_seconds=240, seed=40 + i, separation=0.6,
                                                       channels=7, native_rate=300.0), alignment="attempts")
             for i, p in enumerate(["D1", "D2"])]
    return merge(parts)


def test_dataset_and_trainer_follow_the_headsets_channel_count(dsi_dataset, tmp_path, monkeypatch):
    assert dsi_dataset.X_eeg.shape[2] == 7
    npz, ckpt = tmp_path / "dsi.npz", tmp_path / "dsi.pt"
    save(dsi_dataset, npz)
    monkeypatch.setattr(sys, "argv", ["train_real", str(npz), "--checkpoint", str(ckpt), "--epochs", "4", "--skip-cv"])
    train_real.main()
    assert json.loads(Path(str(ckpt) + ".config.json").read_text())["num_channels"] == 7
    assert checkpoint_num_channels(ckpt) == 7
    predictor = StatePredictor(ckpt)
    assert predictor.num_channels == 7
    state, confidence, probs = predictor.predict(dsi_dataset.X_eeg[0], dsi_dataset.X_behavior[0])
    assert state in STATES and abs(sum(probs.values()) - 1) < 1e-5
    with pytest.raises(Exception):                      # an 8-channel input is refused, not silently mangled
        predictor.predict(np.zeros((5, 8, 3), dtype=np.float32), dsi_dataset.X_behavior[0])


def test_original_checkpoints_still_report_eight_channels(tmp_path):
    ckpt = tmp_path / "plain.pt"
    torch.save(FusionLSTMClassifier(8, len(BAND_NAMES), NUM_BEHAVIOR_FEATS, len(STATES)).state_dict(), ckpt)
    assert checkpoint_num_channels(ckpt) == 8 and StatePredictor(ckpt).num_channels == 8


def test_null_predictor_claims_nothing():
    state, confidence, probs = NullStatePredictor().predict(np.zeros((5, 7, 3)), np.zeros((5, 4)))
    assert confidence == 0.0 and len(set(probs.values())) == 1
    assert NullStatePredictor().predict_with_internals(None, None)["model_available"] is False


def test_live_estimator_declines_to_guess_for_the_wrong_channel_count(tmp_path):
    ckpt = tmp_path / "eeg8.pt"
    torch.save(FusionLSTMClassifier(8, len(BAND_NAMES), NUM_BEHAVIOR_FEATS, len(STATES)).state_dict(), ckpt)
    estimator = LiveStateEstimator(real_checkpoint=tmp_path / "none.pt", synthetic_checkpoint=ckpt)
    assert estimator.num_channels == 8
    rng = np.random.default_rng(0)
    assert estimator.estimate(rng.normal(0, 0.5, size=(128 * 24, 8)), 128) is not None
    assert estimator.estimate(rng.normal(0, 0.5, size=(128 * 24, 7)), 128) is None


# ------------------------------------------------------------- server behaviour
def _ckpt(path: Path, channels: int, with_sidecar: bool) -> Path:
    torch.save(FusionLSTMClassifier(channels, len(BAND_NAMES), NUM_BEHAVIOR_FEATS, len(STATES)).state_dict(), path)
    if with_sidecar:
        Path(str(path) + ".config.json").write_text(json.dumps({"num_channels": channels, "behavior_mask": [1, 1, 1, 0]}))
    return path


@pytest.fixture
def api7(tmp_path, monkeypatch):
    """Server whose hub is fed a 7-channel simulator."""
    monkeypatch.setattr(server_module, "_store", SessionStore(":memory:"))
    monkeypatch.setattr(server_module, "_account_store", AccountStore(":memory:"))
    monkeypatch.setattr(server_module, "_hub", EEGHub(recordings_root=tmp_path / "rec"))
    monkeypatch.setattr(server_module, "EEG_ONLY_CHECKPOINT", tmp_path / "no-eeg-only.pt")
    monkeypatch.setattr(server_module, "_live_estimator", None)

    def connect(client):
        async def go():
            await server_module._hub.connect(SimulatedEEGSource(num_channels=7, chunk_seconds=0.05), "sim")
        client.portal.call(go)

    with TestClient(server_module.app) as client:
        yield client, connect


def _first_update(ws, chunks=None):
    """Solves the first puzzle via the Solution hint and returns the update; any
    eeg_chunk messages seen along the way are appended to `chunks`."""
    def next_of(kind):
        while True:
            msg = ws.receive_json()
            if msg["type"] == "eeg_chunk" and chunks is not None:
                chunks.append(msg)
            if msg["type"] == kind:
                return msg
    puzzle = next_of("puzzle")
    ws.send_json({"type": "puzzle_hint"})
    hint = next_of("puzzle_hint")
    ws.send_json({"move_uci": hint["square"] + hint["target_square"], "time_to_move": 2.0, "attempt_token": puzzle["attempt_token"]})
    return next_of("update")


def test_tutor_refuses_a_mismatched_stream_with_a_clear_next_step(api7, tmp_path, monkeypatch):
    client, connect = api7
    monkeypatch.setattr(server_module, "DEFAULT_CHECKPOINT", _ckpt(tmp_path / "eight.pt", 8, with_sidecar=False))
    connect(client)
    with client.websocket_connect("/ws/session?source=hub") as ws:
        msg = ws.receive_json()
    assert msg["type"] == "error" and "7 channels" in msg["message"] and "expects 8" in msg["message"]
    assert "train_real" in msg["message"]


def test_data_collection_mode_runs_without_a_model_for_this_headset(api7, tmp_path, monkeypatch):
    client, connect = api7
    monkeypatch.setattr(server_module, "DEFAULT_CHECKPOINT", _ckpt(tmp_path / "eight.pt", 8, with_sidecar=False))
    connect(client)
    with client.websocket_connect("/ws/session?source=hub&policy=fixed&learner_id=collect-x&target_min=900&target_max=1150") as ws:
        update = _first_update(ws)
    assert update["correct"] is True
    assert update["confidence"] == 0.0 and update["network_activity"]["model_available"] is False
    assert update["action"] == {"difficulty_delta": 0.0, "show_hint": False, "pacing_delay": 0.0}


def test_a_seven_channel_model_serves_a_seven_channel_stream(api7, tmp_path, monkeypatch):
    client, connect = api7
    monkeypatch.setattr(server_module, "DEFAULT_CHECKPOINT", _ckpt(tmp_path / "seven.pt", 7, with_sidecar=True))
    connect(client)
    with client.websocket_connect("/ws/session?source=hub") as ws:
        update = _first_update(ws)
    assert update["predicted_state"] in STATES
    assert len(update["network_activity"]["gcn1_adjacency"]) == 7              # a 7-node graph for the diagram
    assert len(update["network_activity"]["gcn1_node_activity"]) == 7


def test_the_private_simulator_is_sized_to_the_loaded_model(api7, tmp_path, monkeypatch):
    client, _ = api7
    monkeypatch.setattr(server_module, "DEFAULT_CHECKPOINT", _ckpt(tmp_path / "seven.pt", 7, with_sidecar=True))
    chunks: list = []
    with client.websocket_connect("/ws/session") as ws:       # default: source=sim
        update = _first_update(ws, chunks)
    assert chunks and len(chunks[0]["samples"][0]) == 7
    assert update["predicted_state"] in STATES


def test_connect_warns_when_the_stream_and_tutor_model_disagree(api7, tmp_path, monkeypatch):
    client, _ = api7
    monkeypatch.setattr(server_module, "DEFAULT_CHECKPOINT", _ckpt(tmp_path / "eight.pt", 8, with_sidecar=False))
    monkeypatch.setattr(server_module, "SimulatedEEGSource", lambda **kw: SimulatedEEGSource(num_channels=7, **{k: v for k, v in kw.items() if k != "num_channels"}))
    res = client.post("/api/eeg/connect", json={"source": "sim"}).json()
    assert res["warnings"] and "7 usable channels" in res["warnings"][0] and "expects 8" in res["warnings"][0]
    status = client.get("/api/eeg/status").json()
    assert status["tutor_model_channels"] == 8 and status["num_channels"] == 7
