"""Account data: usernames/password hashes, session tokens, each account's
persisted gamification progress (XP, achievements, solve streak — the
server-side counterpart to what app.js used to keep only in localStorage),
and its daily play-streak. The account-system analog of storage.db's
SessionStore, which owns puzzle-attempt history instead.
"""
from __future__ import annotations
import json
import sqlite3
import time
import uuid

from accounts.auth import generate_session_token, hash_password, verify_password
from accounts.streak import compute_streak_update

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    username TEXT NOT NULL UNIQUE,
    password_salt TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    created_at REAL NOT NULL,
    total_xp INTEGER NOT NULL DEFAULT 0,
    solved_count INTEGER NOT NULL DEFAULT 0,
    current_streak INTEGER NOT NULL DEFAULT 0,
    best_streak INTEGER NOT NULL DEFAULT 0,
    no_hint_streak INTEGER NOT NULL DEFAULT 0,
    no_hint_total_count INTEGER NOT NULL DEFAULT 0,
    max_rating_solved INTEGER NOT NULL DEFAULT 0,
    games_played INTEGER NOT NULL DEFAULT 0,
    games_won INTEGER NOT NULL DEFAULT 0,
    puzzles_served INTEGER NOT NULL DEFAULT 0,
    unlocked_achievements TEXT NOT NULL DEFAULT '[]',
    daily_streak_current INTEGER NOT NULL DEFAULT 0,
    daily_streak_longest INTEGER NOT NULL DEFAULT 0,
    daily_streak_last_date TEXT,
    daily_streak_freezes INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS account_sessions (
    session_token TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    created_at REAL NOT NULL,
    FOREIGN KEY(user_id) REFERENCES users(user_id)
);
"""

PROGRESS_FIELDS = [
    "total_xp", "solved_count", "current_streak", "best_streak",
    "no_hint_streak", "no_hint_total_count", "max_rating_solved",
    "games_played", "games_won", "puzzles_served",
]


class UsernameTakenError(Exception):
    pass


class InvalidCredentialsError(Exception):
    pass


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, column_def: str) -> None:
    """CREATE TABLE IF NOT EXISTS only helps a brand-new db file — a users
    table created before daily_streak_freezes existed needs it added by hand,
    or every account created before this feature shipped would 500 on login."""
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_def}")


class AccountStore:
    def __init__(self, db_path: str = ":memory:"):
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.executescript(SCHEMA)
        _ensure_column(self._conn, "users", "daily_streak_freezes", "INTEGER NOT NULL DEFAULT 0")
        _ensure_column(self._conn, "users", "puzzles_served", "INTEGER NOT NULL DEFAULT 0")
        self._conn.commit()

    def create_user(self, username: str, password: str, initial_progress: dict | None = None) -> str:
        existing = self._conn.execute("SELECT 1 FROM users WHERE username = ?", (username,)).fetchone()
        if existing is not None:
            raise UsernameTakenError(username)
        user_id = str(uuid.uuid4())
        salt_hex, hash_hex = hash_password(password)
        progress = initial_progress or {}
        unlocked = json.dumps(progress.get("unlocked_achievements", []))
        self._conn.execute(
            f"""
            INSERT INTO users (
                user_id, username, password_salt, password_hash, created_at,
                {", ".join(PROGRESS_FIELDS)}, unlocked_achievements
            ) VALUES (?, ?, ?, ?, ?, {", ".join(["?"] * len(PROGRESS_FIELDS))}, ?)
            """,
            (
                user_id, username, salt_hex, hash_hex, time.time(),
                *(progress.get(field, 0) for field in PROGRESS_FIELDS),
                unlocked,
            ),
        )
        self._conn.commit()
        return user_id

    def verify_login(self, username: str, password: str) -> str:
        row = self._conn.execute(
            "SELECT user_id, password_salt, password_hash FROM users WHERE username = ?", (username,)
        ).fetchone()
        if row is None or not verify_password(password, row[1], row[2]):
            raise InvalidCredentialsError()
        return row[0]

    def create_account_session(self, user_id: str) -> str:
        token = generate_session_token()
        self._conn.execute(
            "INSERT INTO account_sessions (session_token, user_id, created_at) VALUES (?, ?, ?)",
            (token, user_id, time.time()),
        )
        self._conn.commit()
        return token

    def get_user_id_for_session(self, session_token: str) -> str | None:
        row = self._conn.execute(
            "SELECT user_id FROM account_sessions WHERE session_token = ?", (session_token,)
        ).fetchone()
        return row[0] if row else None

    def delete_account_session(self, session_token: str) -> None:
        self._conn.execute("DELETE FROM account_sessions WHERE session_token = ?", (session_token,))
        self._conn.commit()

    def get_profile(self, user_id: str) -> dict | None:
        row = self._conn.execute(
            f"""
            SELECT username, {", ".join(PROGRESS_FIELDS)}, unlocked_achievements,
                   daily_streak_current, daily_streak_longest, daily_streak_last_date, daily_streak_freezes
            FROM users WHERE user_id = ?
            """,
            (user_id,),
        ).fetchone()
        if row is None:
            return None
        username = row[0]
        progress_values = row[1:1 + len(PROGRESS_FIELDS)]
        unlocked_achievements, daily_current, daily_longest, daily_last_date, daily_freezes = row[1 + len(PROGRESS_FIELDS):]
        profile = {"username": username, **dict(zip(PROGRESS_FIELDS, progress_values))}
        profile["unlocked_achievements"] = json.loads(unlocked_achievements)
        profile["daily_streak"] = {
            "current": daily_current, "longest": daily_longest,
            "last_play_date": daily_last_date, "freezes_available": daily_freezes,
        }
        return profile

    def save_progress(self, user_id: str, progress: dict) -> None:
        """Upserts the client's gamification counters — NOT the daily streak,
        which only ever changes through record_daily_checkin's own date logic."""
        unlocked = json.dumps(progress.get("unlocked_achievements", []))
        self._conn.execute(
            f"""
            UPDATE users SET {", ".join(f"{field} = ?" for field in PROGRESS_FIELDS)}, unlocked_achievements = ?
            WHERE user_id = ?
            """,
            (*(progress.get(field, 0) for field in PROGRESS_FIELDS), unlocked, user_id),
        )
        self._conn.commit()

    def record_daily_checkin(self, user_id: str, today: str | None = None) -> dict:
        today = today or time.strftime("%Y-%m-%d", time.localtime())
        row = self._conn.execute(
            """
            SELECT daily_streak_current, daily_streak_longest, daily_streak_last_date, daily_streak_freezes
            FROM users WHERE user_id = ?
            """,
            (user_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Unknown user_id: {user_id}")
        current, longest, last_date, freezes = row
        result = compute_streak_update(last_date, current, longest, freezes, today)
        self._conn.execute(
            """
            UPDATE users SET daily_streak_current = ?, daily_streak_longest = ?,
                daily_streak_last_date = ?, daily_streak_freezes = ?
            WHERE user_id = ?
            """,
            (
                result["current_streak"], result["longest_streak"], result["last_play_date"],
                result["freezes_available"], user_id,
            ),
        )
        self._conn.commit()
        return result

    def get_leaderboard(self, limit: int = 50) -> list[dict]:
        """Ranked by total XP (ties broken by puzzles solved) — public to any
        visitor, logged in or not; usernames are the only identifying info an
        account has, and choosing one already implies it's shown in the app."""
        rows = self._conn.execute(
            """
            SELECT username, total_xp, solved_count, best_streak, daily_streak_longest
            FROM users ORDER BY total_xp DESC, solved_count DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [
            {
                "username": r[0], "total_xp": r[1], "solved_count": r[2],
                "best_streak": r[3], "daily_streak_longest": r[4],
            }
            for r in rows
        ]

    def get_rank(self, user_id: str) -> int | None:
        """1-based rank by total XP among all accounts, or None if user_id
        doesn't exist. Ties share the same rank (dense: no gaps for ties)."""
        row = self._conn.execute("SELECT total_xp FROM users WHERE user_id = ?", (user_id,)).fetchone()
        if row is None:
            return None
        (xp,) = row
        (ahead,) = self._conn.execute("SELECT COUNT(*) FROM users WHERE total_xp > ?", (xp,)).fetchone()
        return ahead + 1

    def close(self) -> None:
        self._conn.close()
