from __future__ import annotations
import time
from dataclasses import dataclass
import numpy as np
from chess_task.base import TaskEngine, Puzzle
from eeg.source import EEGSource
from preprocessing.epoching import EpochBuffer
from preprocessing.features import extract_node_features
from model.inference import StatePredictor
from adaptive.policy import Policy, Action
from storage.db import SessionStore
from common.config import SEQ_LEN
from data_gen.generate_dataset import behavior_to_vector

@dataclass
class SessionUpdate:
    puzzle: Puzzle
    correct: bool
    predicted_state: str
    confidence: float
    probs: dict
    action: Action
    sense_to_adapt_latency: float

class TutorSession:
    def __init__(
        self, eeg_source: EEGSource, task_engine: TaskEngine, predictor: StatePredictor,
        policy: Policy, store: SessionStore, initial_difficulty: float = 1000.0,
    ):
        self.eeg_source = eeg_source
        self.task_engine = task_engine
        self.predictor = predictor
        self.policy = policy
        self.store = store
        self.difficulty = initial_difficulty
        self.session_id = store.create_session()
        self.epoch_buffer = EpochBuffer(num_channels=eeg_source.num_channels, sample_rate=eeg_source.sample_rate)
        self._eeg_history: list[np.ndarray] = []
        self._behavior_history: list[np.ndarray] = []

    def next_puzzle(self) -> Puzzle:
        return self.task_engine.get_puzzle(self.difficulty)

    def record_eeg_chunk(self, chunk) -> None:
        self.epoch_buffer.add_chunk(chunk)

    def submit_move(self, puzzle: Puzzle, move_uci: str, time_to_move: float) -> SessionUpdate:
        behavior_event = self.task_engine.submit_move(puzzle, move_uci, time_to_move)
        # sense_to_adapt_latency covers only EEG processing + inference + policy decision.
        # The task-engine/evaluator call above (seconds under Stockfish) and the storage
        # write below are deliberately excluded — neither is part of "sense -> adapt".
        start_time = time.monotonic()
        epoch = self.epoch_buffer.extract_epoch()
        node_features = extract_node_features(epoch.samples, self.eeg_source.sample_rate)
        behavior_vector = behavior_to_vector(behavior_event)

        self._eeg_history = (self._eeg_history + [node_features])[-SEQ_LEN:]
        self._behavior_history = (self._behavior_history + [behavior_vector])[-SEQ_LEN:]

        eeg_seq = self._padded_sequence(self._eeg_history, node_features.shape)
        behavior_seq = self._padded_sequence(self._behavior_history, behavior_vector.shape)

        predicted_state, confidence, probs = self.predictor.predict(eeg_seq, behavior_seq)
        action = self.policy.decide(predicted_state, confidence)
        self.difficulty = max(400.0, self.difficulty + action.difficulty_delta)
        latency = time.monotonic() - start_time

        self.store.log_attempt(
            session_id=self.session_id, puzzle_id=puzzle.puzzle_id,
            correct=behavior_event.correct, time_to_move=behavior_event.time_to_move,
            eval_loss=behavior_event.eval_loss, puzzle_rating=behavior_event.puzzle_rating,
            predicted_state=predicted_state, confidence=confidence,
            difficulty_delta=action.difficulty_delta, show_hint=action.show_hint,
            pacing_delay=action.pacing_delay, sense_to_adapt_latency=latency,
        )
        return SessionUpdate(
            puzzle=puzzle, correct=behavior_event.correct, predicted_state=predicted_state,
            confidence=confidence, probs=probs, action=action, sense_to_adapt_latency=latency,
        )

    @staticmethod
    def _padded_sequence(history: list[np.ndarray], item_shape) -> np.ndarray:
        pad_count = SEQ_LEN - len(history)
        if pad_count > 0:
            padding = [np.zeros(item_shape, dtype=np.float32) for _ in range(pad_count)]
            return np.stack(padding + history)
        return np.stack(history)
