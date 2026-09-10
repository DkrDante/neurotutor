from __future__ import annotations
import sqlite3
import time
import uuid

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    started_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS attempts (
    attempt_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    puzzle_id TEXT NOT NULL,
    correct INTEGER NOT NULL,
    time_to_move REAL NOT NULL,
    eval_loss REAL NOT NULL,
    puzzle_rating INTEGER NOT NULL,
    predicted_state TEXT NOT NULL,
    confidence REAL NOT NULL,
    difficulty_delta REAL NOT NULL,
    show_hint INTEGER NOT NULL,
    pacing_delay REAL NOT NULL,
    sense_to_adapt_latency REAL NOT NULL,
    created_at REAL NOT NULL,
    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
);
"""

class SessionStore:
    def __init__(self, db_path: str = ":memory:"):
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        self._conn.execute("INSERT INTO sessions (session_id, started_at) VALUES (?, ?)", (session_id, time.time()))
        self._conn.commit()
        return session_id

    def log_attempt(
        self, session_id: str, puzzle_id: str, correct: bool, time_to_move: float,
        eval_loss: float, puzzle_rating: int, predicted_state: str, confidence: float,
        difficulty_delta: float, show_hint: bool, pacing_delay: float, sense_to_adapt_latency: float,
    ) -> str:
        attempt_id = str(uuid.uuid4())
        self._conn.execute(
            """INSERT INTO attempts (
                attempt_id, session_id, puzzle_id, correct, time_to_move, eval_loss,
                puzzle_rating, predicted_state, confidence, difficulty_delta, show_hint,
                pacing_delay, sense_to_adapt_latency, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                attempt_id, session_id, puzzle_id, int(correct), time_to_move, eval_loss,
                puzzle_rating, predicted_state, confidence, difficulty_delta, int(show_hint),
                pacing_delay, sense_to_adapt_latency, time.time(),
            ),
        )
        self._conn.commit()
        return attempt_id

    def get_session_summary(self, session_id: str) -> dict:
        rows = self._conn.execute(
            "SELECT correct, confidence, sense_to_adapt_latency, predicted_state FROM attempts "
            "WHERE session_id = ? ORDER BY created_at",
            (session_id,),
        ).fetchall()
        if not rows:
            return {"num_attempts": 0, "accuracy_rate": 0.0, "avg_latency": 0.0, "state_trend": []}
        num_attempts = len(rows)
        accuracy_rate = sum(r[0] for r in rows) / num_attempts
        avg_latency = sum(r[2] for r in rows) / num_attempts
        state_trend = [r[3] for r in rows]
        return {"num_attempts": num_attempts, "accuracy_rate": accuracy_rate, "avg_latency": avg_latency, "state_trend": state_trend}

    def close(self) -> None:
        self._conn.close()
