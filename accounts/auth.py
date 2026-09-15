"""Password hashing and session-token helpers for the account system.

Deliberately stdlib-only (PBKDF2-HMAC-SHA256 via `hashlib`, no bcrypt/argon2
dependency) — a lightweight profile for a solo capstone project, not a
hardened production auth service. No email verification, password reset, or
login-attempt rate limiting; see the account endpoints in web/server.py for
the full scope this was built to.
"""
from __future__ import annotations
import hashlib
import hmac
import secrets

PBKDF2_ITERATIONS = 200_000


def hash_password(password: str) -> tuple[str, str]:
    """Returns (salt_hex, hash_hex) for a NEW password — caller stores both."""
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return salt.hex(), digest.hex()


def verify_password(password: str, salt_hex: str, expected_hash_hex: str) -> bool:
    salt = bytes.fromhex(salt_hex)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return hmac.compare_digest(digest.hex(), expected_hash_hex)


def generate_session_token() -> str:
    return secrets.token_urlsafe(32)
