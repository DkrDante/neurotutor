"""EEGHub, SessionRecorder and the /api/eeg, /api/record, /ws/live endpoints."""
from __future__ import annotations
import asyncio
import json
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
import web.server as server_module
from accounts.storage import AccountStore
from common.config import BAND_NAMES, NUM_BEHAVIOR_FEATS, NUM_CHANNELS
from common.states import STATES
from eeg.hub import EEGHub, HubSubscriberSource, collect_attempts
from eeg.recorder import SessionRecorder, load_recording
from eeg.simulated import SimulatedEEGSource
from eeg.source import EEGChunk
from model.fusion import FusionLSTMClassifier
from storage.db import SessionStore


def _chunk(n=32, uv=True):
    samples = np.random.default_rng(0).normal(size=(n, NUM_CHANNELS))
    return EEGChunk(timestamp=0.0, samples=samples, uv=samples * 20 if uv else None)


def test_recorder_round_trip_and_monotonic_times(tmp_path):
    rec = SessionRecorder("P 01/x", 128, NUM_CHANNELS, root=tmp_path, label="pilot")
    for k in range(5):
        rec.write_chunk(1000.0 + k * 0.25, _chunk())
    rec.add_marker("block_start", condition="Focused", block="b1")
    directory = rec.stop(attempts=[{
        "session_id": "s", "mode": "puzzle", "puzzle_id": "p", "correct": True, "time_to_move": 2.0,
        "eval_loss": 0.0, "puzzle_rating": 900, "predicted_state": "Focused", "confidence": 0.9,
        "difficulty_delta": 100.0, "show_hint": False, "pacing_delay": 0.0,
        "sense_to_adapt_latency": 0.01, "created_at": 1001.0,
    }])
    assert "P_01_x" in str(directory)  # participant id is made path-safe
    loaded = load_recording(directory)
    assert loaded["eeg"].shape == (160, NUM_CHANNELS)
    assert np.all(np.diff(loaded["times"]) > 0)
    assert loaded["meta"]["uv_available"] is True and loaded["meta"]["ended_at"] is not None
    assert [m["kind"] for m in loaded["markers"]] == ["block_start"]
    assert loaded["attempts"][0]["correct"] is True and loaded["attempts"][0]["puzzle_rating"] == 900


def test_recorder_flags_simulator_data_without_microvolts(tmp_path):
    rec = SessionRecorder("sim", 128, NUM_CHANNELS, root=tmp_path)
    rec.write_chunk(5.0, _chunk(uv=False))
    loaded = load_recording(rec.stop() and rec.directory)
    assert loaded["meta"]["uv_available"] is False


def test_collect_attempts_filters_by_wall_clock():
    store = SessionStore(":memory:")
    sid = store.create_session()
    for _ in range(3):
        store.log_attempt(sid, "p", True, 1.0, 0.0, 900, "Focused", 0.9, 100.0, False, 0.0, 0.01)
    attempts = store.get_session_attempts(sid)
    times = [a["created_at"] for a in attempts]
    got = collect_attempts(store, min(times) - 1, max(times) + 1)
    assert len(got) == 3 and got[0]["session_id"] == sid
    assert collect_attempts(store, max(times) + 10, max(times) + 20) == []


def test_hub_fans_out_chunks_records_and_tracks_blocks(tmp_path):
    async def scenario():
        hub = EEGHub(recordings_root=tmp_path)
        source = SimulatedEEGSource(chunk_seconds=0.05, seed=1)
        await hub.connect(source, "sim")
        sub_a, sub_b = hub.subscribe(), hub.subscribe()
        hub.start_recording("P1")
        hub.add_marker("block_start", condition="Focused", block="b1")
        assert hub.current_block["condition"] == "Focused"
        await asyncio.sleep(0.4)
        got_a, got_b = sub_a.qsize(), sub_b.qsize()
        window = hub.window(1.0)
        status = hub.status()
        directory = hub.stop_recording(attempts=[])
        hub.add_marker("block_end", block="b1")
        assert hub.current_block is None
        await hub.disconnect()
        return got_a, got_b, window, status, directory

    got_a, got_b, window, status, directory = asyncio.run(scenario())
    assert got_a >= 3 and got_a == got_b          # both subscribers see every chunk
    assert 0 < window.shape[0] <= 128 and window.shape[1] == NUM_CHANNELS
    assert status["connected"] and status["recording"]["participant"] == "P1"
    assert load_recording(directory)["eeg"].shape[0] > 0


def test_hub_subscriber_source_requires_a_connected_hub():
    with pytest.raises(RuntimeError):
        HubSubscriberSource(EEGHub())


def test_slow_subscriber_does_not_stall_the_pump(tmp_path):
    async def scenario():
        hub = EEGHub(recordings_root=tmp_path)
        await hub.connect(SimulatedEEGSource(chunk_seconds=0.01, seed=2), "sim")
        slow = hub.subscribe()  # never drained
        await asyncio.sleep(0.5)
        count = hub.chunks_received
        await hub.disconnect()
        return count, slow.qsize()
    count, qsize = asyncio.run(scenario())
    assert count > 20 and qsize <= 256


# ---------------------------------------------------------------- HTTP / WS API
@pytest.fixture
def api(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(FusionLSTMClassifier(
        num_channels=NUM_CHANNELS, num_bands=len(BAND_NAMES),
        num_behavior_feats=NUM_BEHAVIOR_FEATS, num_classes=len(STATES),
    ).state_dict(), checkpoint)
    monkeypatch.setattr(server_module, "DEFAULT_CHECKPOINT", checkpoint)
    monkeypatch.setattr(server_module, "_store", SessionStore(":memory:"))
    monkeypatch.setattr(server_module, "_account_store", AccountStore(":memory:"))
    monkeypatch.setattr(server_module, "_hub", EEGHub(recordings_root=tmp_path / "rec"))
    monkeypatch.setattr(server_module, "RECORDINGS_ROOT", tmp_path / "rec")
    monkeypatch.setattr(server_module, "_live_estimator", None)
    with TestClient(server_module.app) as client:
        yield client


def test_pages_are_served(api):
    for path in ("/live", "/collect"):
        res = api.get(path)
        assert res.status_code == 200 and "text/html" in res.headers["content-type"]


def test_status_when_nothing_is_connected(api):
    status = api.get("/api/eeg/status").json()
    assert status["connected"] is False and status["recording"] is None


def test_connect_validation_and_missing_lsl_stream(api):
    assert api.post("/api/eeg/connect", json={"source": "bogus"}).status_code == 400
    assert api.post("/api/eeg/connect", json={"source": "sim", "sim_state": "Nope"}).status_code == 400
    res = api.post("/api/eeg/connect", json={"source": "lsl", "stream_name": "no-such-stream-xyz"})
    assert res.status_code == 404 and "No LSL stream" in res.json()["detail"]


def test_sim_connect_status_state_change_and_disconnect(api):
    assert api.post("/api/eeg/connect", json={"source": "sim"}).json()["connected"] is True
    status = api.get("/api/eeg/status").json()
    assert status["connected"] and status["kind"] == "sim" and status["num_channels"] == NUM_CHANNELS
    assert api.post("/api/eeg/sim-state", json={"state": "Fatigued"}).json()["state"] == "Fatigued"
    assert api.post("/api/eeg/sim-state", json={"state": "x"}).status_code == 400
    assert api.post("/api/eeg/recalibrate", json={"seconds": 5}).status_code == 409  # sim has no baseline
    api.post("/api/eeg/disconnect")
    assert api.get("/api/eeg/status").json()["connected"] is False
    assert api.post("/api/eeg/sim-state", json={"state": "Focused"}).status_code == 409


def test_recording_lifecycle_over_http(api, tmp_path):
    assert api.post("/api/record/start", json={"participant": "P1"}).status_code == 409  # no source yet
    api.post("/api/eeg/connect", json={"source": "sim"})
    assert api.post("/api/record/start", json={"participant": "  "}).status_code == 400
    started = api.post("/api/record/start", json={"participant": "P1", "label": "t"}).json()
    assert started["recording"]
    assert api.post("/api/record/start", json={"participant": "P1"}).status_code == 409  # already recording
    marker = api.post("/api/record/marker", json={"kind": "block_start", "data": {"condition": "Focused", "block": "b1"}}).json()
    assert marker["condition"] == "Focused"
    assert api.get("/api/eeg/status").json()["current_block"]["condition"] == "Focused"
    import time as _t; _t.sleep(0.6)
    stopped = api.post("/api/record/stop").json()
    assert stopped["recording"] is False and stopped["samples"] > 0
    assert api.post("/api/record/stop").status_code == 409
    listed = api.get("/api/record/list").json()["recordings"]
    assert listed and listed[0]["participant"] == "P1" and listed[0]["samples"] == stopped["samples"]


def test_live_websocket_streams_chunks_and_status(api):
    api.post("/api/eeg/connect", json={"source": "sim"})
    seen = set()
    with api.websocket_connect("/ws/live") as ws:
        for _ in range(60):
            msg = ws.receive_json()
            seen.add(msg["type"])
            if msg["type"] == "status":
                assert msg["connected"] and msg["num_channels"] == NUM_CHANNELS
                assert len(msg["quality"]) == NUM_CHANNELS
            if {"eeg_chunk", "status"} <= seen:
                break
    assert {"eeg_chunk", "status"} <= seen


def test_tutor_session_requires_a_connected_hub_when_asked_for_it(api):
    with api.websocket_connect("/ws/session?source=hub") as ws:
        msg = ws.receive_json()
    assert msg["type"] == "error" and "No EEG source" in msg["message"]


def test_tutor_session_on_the_hub_and_updates_reach_the_live_monitor(api):
    api.post("/api/eeg/connect", json={"source": "sim"})
    with api.websocket_connect("/ws/live") as live:
        with api.websocket_connect("/ws/session?source=hub&learner_id=hub-test") as tutor:
            while True:
                puzzle = tutor.receive_json()
                if puzzle["type"] == "puzzle":
                    break
            tutor.send_json({"type": "puzzle_hint"})
            while True:
                hint = tutor.receive_json()
                if hint["type"] == "puzzle_hint":
                    break
            tutor.send_json({
                "move_uci": hint["square"] + hint["target_square"], "time_to_move": 2.0,
                "attempt_token": puzzle["attempt_token"],
            })
            while tutor.receive_json()["type"] != "update":
                pass
        update = None
        for _ in range(200):
            msg = live.receive_json()
            if msg["type"] == "tutor_update":
                update = msg
                break
    assert update is not None and update["correct"] is True and update["predicted_state"] in STATES


def test_tutor_blocked_when_stream_shape_does_not_match_the_model(api, monkeypatch):
    api.post("/api/eeg/connect", json={"source": "sim"})
    server_module._hub.source.num_channels = 4  # as if a 4-channel headset were connected
    with api.websocket_connect("/ws/session?source=hub") as ws:
        msg = ws.receive_json()
    assert msg["type"] == "error" and "model expects" in msg["message"]


def test_a_crashing_source_is_reported_not_silently_dropped(tmp_path):
    class Exploding(SimulatedEEGSource):
        async def stream(self):
            yield self.generate_chunk()
            raise RuntimeError("bluetooth dropped")

    async def scenario():
        hub = EEGHub(recordings_root=tmp_path)
        await hub.connect(Exploding(chunk_seconds=0.01), "sim")
        await asyncio.sleep(0.2)
        status = hub.status()
        await hub.disconnect()
        return status

    status = asyncio.run(scenario())
    assert status["connected"] is False and "bluetooth dropped" in status["error"]
