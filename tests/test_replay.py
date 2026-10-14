"""ReplayEEGSource and the replay path through the hub / API."""
from __future__ import annotations
import asyncio
import numpy as np
import pytest
from fastapi.testclient import TestClient
import web.server as server_module
from accounts.storage import AccountStore
from data_gen.fake_recording import build_fake_recording
from eeg.hub import EEGHub
from eeg.replay import ReplayEEGSource
from storage.db import SessionStore


@pytest.fixture(scope="module")
def recording(tmp_path_factory):
    root = tmp_path_factory.mktemp("rec")
    return root, build_fake_recording("SIM01", root, block_seconds=60, seed=3)


def test_replay_streams_normalised_signal_and_tracks_the_recorded_condition(recording):
    _, directory = recording
    source = ReplayEEGSource(directory, speed=100.0, loop=False)
    assert source.num_channels == 8 and source.sample_rate == 128

    async def run():
        chunks, conditions = [], []
        async for chunk in source.stream():
            chunks.append(chunk)
            conditions.append(source.now_condition)
        return chunks, conditions
    chunks, conditions = asyncio.run(run())
    samples = np.vstack([c.samples for c in chunks])
    assert samples.shape[1] == 8 and 0.1 < samples.std() < 2.0                  # baseline-normalised, training scale
    assert chunks[0].uv is not None
    assert conditions[0] == "Baseline_EO" and "Overloaded" in set(conditions) and "Late_Focused" in set(conditions)


def test_replay_loops(recording):
    _, directory = recording
    source = ReplayEEGSource(directory, speed=1000.0, loop=True)

    async def run():
        count = 0
        async for _ in source.stream():
            count += 1
            if source.laps >= 1:
                return count
    assert asyncio.run(asyncio.wait_for(run(), 20)) > 10


@pytest.fixture
def api(recording, tmp_path, monkeypatch):
    root, _ = recording
    monkeypatch.setattr(server_module, "_store", SessionStore(":memory:"))
    monkeypatch.setattr(server_module, "_account_store", AccountStore(":memory:"))
    monkeypatch.setattr(server_module, "_hub", EEGHub(recordings_root=tmp_path / "out"))
    monkeypatch.setattr(server_module, "RECORDINGS_ROOT", root)
    with TestClient(server_module.app) as client:
        yield client


def test_replay_connect_through_the_api_and_status_shows_the_condition(recording, api):
    _, directory = recording
    res = api.post("/api/eeg/connect", json={"source": "replay", "recording": str(directory), "speed": 8})
    assert res.status_code == 200 and res.json()["connected"]
    import time as _t; _t.sleep(0.8)
    status = api.get("/api/eeg/status").json()
    assert status["kind"] == "replay" and status["replay"]["condition"] and status["replay"]["speed"] == 8


def test_replay_refuses_paths_outside_the_recordings_folder(api, tmp_path):
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "meta.json").write_text("{}")
    assert api.post("/api/eeg/connect", json={"source": "replay", "recording": str(outside)}).status_code == 404
    assert api.post("/api/eeg/connect", json={"source": "replay", "recording": "../../etc"}).status_code == 404
    assert api.post("/api/eeg/connect", json={"source": "replay"}).status_code == 400
