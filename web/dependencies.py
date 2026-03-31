"""FastAPI dependencies for DB sessions and auth."""

from fastapi import Request, HTTPException
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from db.database import SessionLocal
from db.crud import get_user_by_id
from db.models import User
from .auth import decode_session_cookie


def get_db():
    """Yield a DB session, auto-close after request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_current_user(request: Request, db: Session) -> User:
    """Read session cookie and return the authenticated User, or redirect to login."""
    cookie = request.cookies.get("session")
    if not cookie:
        raise HTTPException(status_code=303, headers={"Location": "/login"})

    user_id = decode_session_cookie(cookie)
    if user_id is None:
        raise HTTPException(status_code=303, headers={"Location": "/login"})

    user = get_user_by_id(db, user_id)
    if user is None:
        raise HTTPException(status_code=303, headers={"Location": "/login"})

    return user
