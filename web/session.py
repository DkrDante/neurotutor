from __future__ import annotations
import time
from dataclasses import dataclass
import numpy as np
from chess_task.base import TaskEngine, Puzzle
from eeg.source import EEGSource
from preprocessing.epoching import EpochBuffer
from preprocessing.filters import bandpass_filter
from preprocessing.features import extract_node_features
from model.inference import StatePredictor
from adaptive.policy import Policy, Action
from adaptive.proactive import PROACTIVE_DIFFICULTY_NUDGE, detect_struggle_trend
from storage.db import SessionStore
from common.config import SEQ_LEN
from data_gen.generate_dataset import behavior_to_vector

@dataclass
class SessionUpdate:
    puzzle: Puzzle
    correct: bool
    eval_loss: float
    predicted_state: str
    confidence: float
    probs: dict
    action: Action
    sense_to_adapt_latency: float
    network_activity: dict
    proactive_intervention: bool

class TutorSession:
    def __init__(
        self, eeg_source: EEGSource, task_engine: TaskEngine, predictor: StatePredictor,
        policy: Policy, store: SessionStore, initial_difficulty: float = 1000.0,
        eeg_only_predictor: StatePredictor | None = None,
        target_rating_min: float | None = None, target_rating_max: float | None = None,
        mode: str = "puzzle",
    ):
        # Recorded on every logged attempt (see log_attempt below) so puzzle
        # and full-game attempts — which repurpose puzzle_rating/puzzle_id
        # for different meanings, see chess_task/full_game.py — never get
        # silently combined in cross-session analytics or rating-band
        # coaching stats (both are keyed off this same string).
        self.mode = mode
        # When set, next_puzzle() draws only from this rating band — targeted
        # practice on a weak spot (see coaching.statistics' weakest_rating_band)
        # instead of the usual nearest-to-difficulty selection across the whole set.
        self.target_rating_min = target_rating_min
        self.target_rating_max = target_rating_max
        self.eeg_source = eeg_source
        self.task_engine = task_engine
        self.predictor = predictor
        # A second, real model — trained with behavior features zeroed out (see
        # model/ablation.py's eeg_only variant) — run alongside the fused model so
        # the live UI can show a genuine "chess state, from EEG alone" prediction,
        # not a derived/decorative readout of the fused model's internals.
        self.eeg_only_predictor = eeg_only_predictor
        self.policy = policy
        self.store = store
        self.difficulty = initial_difficulty
        self.session_id = store.create_session()
        self.epoch_buffer = EpochBuffer(num_channels=eeg_source.num_channels, sample_rate=eeg_source.sample_rate)
        self._eeg_history: list[np.ndarray] = []
        self._behavior_history: list[np.ndarray] = []
        # Rolling (predicted_state, confidence) history feeding the PROACTIVE
        # loop (adaptive.proactive) — separate from _eeg_history/_behavior_history
        # above, which feed the model's own input window instead.
        self._state_history: list[tuple[str, float]] = []
        self._proactive_active = False

    def next_puzzle(self) -> Puzzle:
        return self.task_engine.get_puzzle(
            self.difficulty, rating_min=self.target_rating_min, rating_max=self.target_rating_max,
        )

    def record_eeg_chunk(self, chunk) -> None:
        self.epoch_buffer.add_chunk(chunk)

    def submit_move(self, puzzle: Puzzle, move_uci: str, time_to_move: float) -> SessionUpdate:
        behavior_event = self.task_engine.submit_move(puzzle, move_uci, time_to_move)
        # sense_to_adapt_latency covers only EEG processing + inference + policy decision.
        # The task-engine/evaluator call above (seconds under Stockfish) and the storage
        # write below are deliberately excluded — neither is part of "sense -> adapt".
        start_time = time.monotonic()
        epoch = self.epoch_buffer.extract_epoch()
        # Same bandpass applied in data_gen.generate_dataset, so serving-time features
        # match the distribution the model was trained on.
        filtered = bandpass_filter(epoch.samples, self.eeg_source.sample_rate)
        node_features = extract_node_features(filtered, self.eeg_source.sample_rate)
        behavior_vector = behavior_to_vector(behavior_event)

        if not epoch.is_artifact:
            self._eeg_history = (self._eeg_history + [node_features])[-SEQ_LEN:]
            self._behavior_history = (self._behavior_history + [behavior_vector])[-SEQ_LEN:]
        # On an artifact-flagged epoch we leave the rolling window untouched: a corrupted
        # epoch would otherwise pollute the model's input for the next SEQ_LEN attempts.
        # The attempt itself still runs (predicted on the previous history) and is logged.

        eeg_seq = self._padded_sequence(self._eeg_history, node_features.shape)
        behavior_seq = self._padded_sequence(self._behavior_history, behavior_vector.shape)

        # predict_with_internals() is a strict superset of predict(): same state/
        # confidence/probs, plus the real activations/weights the live "network
        # activity" visualization renders (nothing here is synthetic/decorative).
        internals = self.predictor.predict_with_internals(eeg_seq, behavior_seq)
        predicted_state, confidence, probs = internals["state"], internals["confidence"], internals["probs"]

        if self.eeg_only_predictor is not None:
            eeg_only = self.eeg_only_predictor.predict_with_internals(eeg_seq, np.zeros_like(behavior_seq))
            internals["eeg_only_state"] = eeg_only["state"]
            internals["eeg_only_confidence"] = eeg_only["confidence"]
            internals["eeg_only_probs"] = eeg_only["probs"]
        action = self.policy.decide(predicted_state, confidence, behavior_event.correct)

        # Proactive loop: acts on a TREND across recent predicted states, on
        # top of whatever the reactive policy above already decided — so it
        # can kick in even on an attempt that was scored correct, before a
        # run of concrete errors ever shows up. Only applies the extra nudge
        # once per trend "episode" (not every attempt the trend persists),
        # so difficulty doesn't crash down repeatedly for one sustained dip.
        self._state_history = (self._state_history + [(predicted_state, confidence)])[-10:]
        trend = detect_struggle_trend(self._state_history)
        proactive_intervention = trend and not self._proactive_active
        self._proactive_active = trend

        difficulty_delta = action.difficulty_delta + (PROACTIVE_DIFFICULTY_NUDGE if proactive_intervention else 0.0)
        self.difficulty = max(400.0, self.difficulty + difficulty_delta)
        latency = time.monotonic() - start_time

        self.store.log_attempt(
            session_id=self.session_id, puzzle_id=puzzle.puzzle_id,
            correct=behavior_event.correct, time_to_move=behavior_event.time_to_move,
            eval_loss=behavior_event.eval_loss, puzzle_rating=behavior_event.puzzle_rating,
            predicted_state=predicted_state, confidence=confidence,
            difficulty_delta=action.difficulty_delta, show_hint=action.show_hint,
            pacing_delay=action.pacing_delay, sense_to_adapt_latency=latency,
            mode=self.mode,
        )
        return SessionUpdate(
            puzzle=puzzle, correct=behavior_event.correct, eval_loss=behavior_event.eval_loss,
            predicted_state=predicted_state, confidence=confidence, probs=probs, action=action,
            sense_to_adapt_latency=latency, network_activity=internals,
            proactive_intervention=proactive_intervention,
        )

    @staticmethod
    def _padded_sequence(history: list[np.ndarray], item_shape) -> np.ndarray:
        pad_count = SEQ_LEN - len(history)
        if pad_count > 0:
            padding = [np.zeros(item_shape, dtype=np.float32) for _ in range(pad_count)]
            return np.stack(padding + history)
        return np.stack(history)
