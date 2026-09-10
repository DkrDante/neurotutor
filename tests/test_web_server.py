from pathlib import Path
import torch
from fastapi.testclient import TestClient
from model.fusion import FusionLSTMClassifier
from common.config import NUM_CHANNELS, BAND_NAMES, NUM_BEHAVIOR_FEATS
from common.states import STATES
import web.server as server_module
from storage.db import SessionStore

def test_websocket_session_round_trip(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "checkpoint.pt"
    model = FusionLSTMClassifier(
        num_channels=NUM_CHANNELS, num_bands=len(BAND_NAMES),
        num_behavior_feats=NUM_BEHAVIOR_FEATS, num_classes=len(STATES),
    )
    torch.save(model.state_dict(), checkpoint_path)
    monkeypatch.setattr(server_module, "DEFAULT_CHECKPOINT", checkpoint_path)
    monkeypatch.setattr(server_module, "_store", SessionStore(":memory:"))

    client = TestClient(server_module.app)
    with client.websocket_connect("/ws/session") as ws:
        puzzle_msg = ws.receive_json()
        assert puzzle_msg["type"] == "puzzle"
        assert "fen" in puzzle_msg

        ws.send_json({"move_uci": "a1a8", "time_to_move": 5.0})
        update_msg = ws.receive_json()
        assert update_msg["type"] == "update"
        assert update_msg["predicted_state"] in STATES
        assert 0.0 <= update_msg["confidence"] <= 1.0
