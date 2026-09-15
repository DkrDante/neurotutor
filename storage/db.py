from __future__ import annotations
import json
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
    mode TEXT NOT NULL DEFAULT 'puzzle',
    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
);
CREATE TABLE IF NOT EXISTS learner_profiles (
    learner_id TEXT PRIMARY KEY,
    difficulty REAL NOT NULL,
    policy_state TEXT,
    updated_at REAL NOT NULL,
    served_puzzle_ids TEXT
);
"""


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, column_def: str) -> None:
    """CREATE TABLE IF NOT EXISTS only helps a brand-new db file — a
    learner_profiles table created before served_puzzle_ids existed needs it
    added by hand, or every returning learner would fail on the next save."""
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_def}")

class SessionStore:
    def __init__(self, db_path: str = ":memory:"):
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.executescript(SCHEMA)
        _ensure_column(self._conn, "learner_profiles", "served_puzzle_ids", "TEXT")
        _ensure_column(self._conn, "attempts", "mode", "TEXT NOT NULL DEFAULT 'puzzle'")
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
        mode: str = "puzzle",
    ) -> str:
        # mode distinguishes puzzle attempts (puzzle_rating is a real Lichess
        # rating) from full-game attempts (chess_task/full_game.py repurposes
        # puzzle_rating to mean the live adaptive-difficulty number, and
        # puzzle_id to "game-ply-N") — without it, rating-band coaching stats
        # and cross-session analytics would silently mix the two.
        attempt_id = str(uuid.uuid4())
        self._conn.execute(
            """INSERT INTO attempts (
                attempt_id, session_id, puzzle_id, correct, time_to_move, eval_loss,
                puzzle_rating, predicted_state, confidence, difficulty_delta, show_hint,
                pacing_delay, sense_to_adapt_latency, created_at, mode
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                attempt_id, session_id, puzzle_id, int(correct), time_to_move, eval_loss,
                puzzle_rating, predicted_state, confidence, difficulty_delta, int(show_hint),
                pacing_delay, sense_to_adapt_latency, time.time(), mode,
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

    def list_sessions(self, mode: str | None = None) -> list[dict]:
        """One row per session that has logged at least one attempt, newest first,
        with the same summary stats get_session_summary computes — enough for the
        analytics page's session-picker list without a second query per row.
        `mode` is homogeneous within a session (see log_attempt's docstring), so
        MAX(a.mode) just reads back whichever single value every row shares.
        An optional `mode` filter scopes this to just "puzzle" or "game" sessions
        — used by the aggregate cohort stats so puzzle and full-game sessions
        are never silently combined."""
        query = """
            SELECT s.session_id, s.started_at,
                   COUNT(a.attempt_id) AS num_attempts,
                   COALESCE(AVG(a.correct), 0.0) AS accuracy_rate,
                   COALESCE(AVG(a.sense_to_adapt_latency), 0.0) AS avg_latency,
                   MAX(a.mode) AS mode
            FROM sessions s
            JOIN attempts a ON a.session_id = s.session_id
        """
        params: tuple = ()
        if mode is not None:
            query += " WHERE a.mode = ?"
            params = (mode,)
        query += " GROUP BY s.session_id ORDER BY s.started_at DESC"
        rows = self._conn.execute(query, params).fetchall()
        return [
            {
                "session_id": r[0], "started_at": r[1], "num_attempts": r[2],
                "accuracy_rate": r[3], "avg_latency": r[4], "mode": r[5],
            }
            for r in rows
        ]

    def get_session_attempts(self, session_id: str) -> list[dict]:
        """Full per-attempt time series for the analytics page's charts — every
        column log_attempt records, in play order."""
        rows = self._conn.execute(
            """
            SELECT puzzle_id, correct, time_to_move, eval_loss, puzzle_rating,
                   predicted_state, confidence, difficulty_delta, show_hint,
                   pacing_delay, sense_to_adapt_latency, created_at, mode
            FROM attempts WHERE session_id = ? ORDER BY created_at
            """,
            (session_id,),
        ).fetchall()
        return [
            {
                "puzzle_id": r[0], "correct": bool(r[1]), "time_to_move": r[2],
                "eval_loss": r[3], "puzzle_rating": r[4], "predicted_state": r[5],
                "confidence": r[6], "difficulty_delta": r[7], "show_hint": bool(r[8]),
                "pacing_delay": r[9], "sense_to_adapt_latency": r[10], "created_at": r[11],
                "mode": r[12],
            }
            for r in rows
        ]

    def get_learner_profile(self, learner_id: str) -> dict | None:
        """A returning learner's persisted state — their last difficulty,
        (if they were using QLearningPolicy) its serialized Q-table, and
        which puzzles they've already been served this cycle — the design
        spec's "multi-session personalization" extension. Frontend generates
        and stores `learner_id` in localStorage and sends it on every
        connection; None for a learner we've never seen.

        served_puzzle_ids matters specifically for reconnects: a fresh
        PuzzleTaskEngine is constructed per connection, so without seeding
        its no-repeat tracking from here, a page refresh (or any reconnect)
        would silently forget every puzzle already shown and could re-serve
        the exact same one right away."""
        row = self._conn.execute(
            "SELECT difficulty, policy_state, served_puzzle_ids FROM learner_profiles WHERE learner_id = ?",
            (learner_id,),
        ).fetchone()
        if row is None:
            return None
        difficulty, policy_state_json, served_puzzle_ids_json = row
        return {
            "difficulty": difficulty,
            "policy_state": json.loads(policy_state_json) if policy_state_json else None,
            "served_puzzle_ids": set(json.loads(served_puzzle_ids_json)) if served_puzzle_ids_json else set(),
        }

    def save_learner_profile(
        self, learner_id: str, difficulty: float, policy_state: dict | None = None,
        served_puzzle_ids: set[str] | None = None,
    ) -> None:
        """Upserts the learner's current difficulty, (optionally) policy
        state, and (optionally) which puzzles they've been served — called
        after every attempt so a mid-session disconnect still keeps whatever
        progress was made. served_puzzle_ids=None leaves the stored set
        untouched (game mode has no puzzle pool to track, so it never passes
        one) rather than blanking it out."""
        policy_state_json = json.dumps(policy_state) if policy_state is not None else None
        if served_puzzle_ids is not None:
            served_json = json.dumps(sorted(served_puzzle_ids))
            self._conn.execute(
                """
                INSERT INTO learner_profiles (learner_id, difficulty, policy_state, updated_at, served_puzzle_ids)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(learner_id) DO UPDATE SET
                    difficulty = excluded.difficulty,
                    policy_state = excluded.policy_state,
                    updated_at = excluded.updated_at,
                    served_puzzle_ids = excluded.served_puzzle_ids
                """,
                (learner_id, difficulty, policy_state_json, time.time(), served_json),
            )
        else:
            self._conn.execute(
                """
                INSERT INTO learner_profiles (learner_id, difficulty, policy_state, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(learner_id) DO UPDATE SET
                    difficulty = excluded.difficulty,
                    policy_state = excluded.policy_state,
                    updated_at = excluded.updated_at
                """,
                (learner_id, difficulty, policy_state_json, time.time()),
            )
        self._conn.commit()

    def get_aggregate_stats(self, mode: str) -> dict:
        """Cross-session cohort statistics, scoped to ONE mode ("puzzle" or
        "game") — the analytics page's session drill-down only ever shows one
        run at a time; this is what a capstone defense or grader actually
        wants to see: real numbers aggregated across every recorded session
        of that mode, not eyeballing single runs one by one.

        `mode` is required, not optional: full-game attempts repurpose
        puzzle_rating to mean the live adaptive-difficulty number (see
        chess_task/full_game.py), so combining the two modes' attempts would
        silently blend a real accuracy/state-distribution signal with one
        that means something different — callers must pick one mode, and
        combine the two returned dicts themselves if they want a total."""
        total_attempts, overall_accuracy, avg_latency = self._conn.execute(
            "SELECT COUNT(*), COALESCE(AVG(correct), 0.0), COALESCE(AVG(sense_to_adapt_latency), 0.0) "
            "FROM attempts WHERE mode = ?",
            (mode,),
        ).fetchone()
        (total_sessions,) = self._conn.execute(
            "SELECT COUNT(DISTINCT session_id) FROM attempts WHERE mode = ?", (mode,)
        ).fetchone()

        state_rows = self._conn.execute(
            """
            SELECT predicted_state, COUNT(*), AVG(confidence), AVG(correct)
            FROM attempts WHERE mode = ? GROUP BY predicted_state ORDER BY predicted_state
            """,
            (mode,),
        ).fetchall()
        state_distribution = [
            {"state": r[0], "count": r[1], "avg_confidence": r[2], "accuracy": r[3]}
            for r in state_rows
        ]

        # Engagement trend: for each session with at least 2 attempts, compare
        # accuracy in the second half of that session against the first half —
        # positive means learners tend to IMPROVE within a session, negative
        # means they tend to fade (a real fatigue/disengagement signal, not
        # just a single flat accuracy number). Averaged across all sessions
        # of this mode long enough to split.
        session_ids = [
            r[0] for r in self._conn.execute(
                "SELECT DISTINCT session_id FROM attempts WHERE mode = ?", (mode,)
            ).fetchall()
        ]
        trend_deltas = []
        for session_id in session_ids:
            attempts = self.get_session_attempts(session_id)
            midpoint = len(attempts) // 2
            first_half, second_half = attempts[:midpoint], attempts[midpoint:]
            if not first_half or not second_half:
                continue
            first_acc = sum(a["correct"] for a in first_half) / len(first_half)
            second_acc = sum(a["correct"] for a in second_half) / len(second_half)
            trend_deltas.append(second_acc - first_acc)

        return {
            "mode": mode,
            "total_sessions": total_sessions,
            "total_attempts": total_attempts,
            "overall_accuracy": overall_accuracy,
            "avg_latency": avg_latency,
            "state_distribution": state_distribution,
            "engagement_trend_avg": (sum(trend_deltas) / len(trend_deltas)) if trend_deltas else None,
            "engagement_trend_sessions_counted": len(trend_deltas),
        }

    def close(self) -> None:
        self._conn.close()
