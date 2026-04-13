"""Simple session auth using signed cookies."""

from typing import Optional
from itsdangerous import URLSafeTimedSerializer
from src.config import Config

_serializer = URLSafeTimedSerializer(Config.SECRET_KEY)


def create_session_cookie(user_id: int) -> str:
    """Create a signed session cookie containing the user ID."""
    return _serializer.dumps(user_id)


def decode_session_cookie(cookie: str) -> Optional[int]:
    """Decode a session cookie. Returns user_id or None if invalid/expired."""
    try:
        return _serializer.loads(cookie, max_age=Config.SESSION_MAX_AGE)
    except Exception:
        return None
