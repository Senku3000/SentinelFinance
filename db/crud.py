"""CRUD operations for all models."""

import json
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from pathlib import Path
from sqlalchemy.orm import Session

from .models import User, UserProfile, ChatMessage, UploadedDocument
from src.config import Config


def _empty_profile(user_id_str: str) -> Dict[str, Any]:
    """Empty profile for new users — no default values."""
    now = datetime.now(timezone.utc).isoformat()
    return {
        "user_id": user_id_str,
        "income": {"monthly": None, "annual": None, "source": None},
        "expenses": {"monthly": None, "breakdown": {}},
        "goals": [],
        "risk_tolerance": None,
        "existing_investments": {},
        "tax_details": {},
        "created_at": now,
        "updated_at": now,
    }


def user_id_str(db_id: int) -> str:
    """Convert DB integer ID to filesystem-compatible string."""
    return f"user_{db_id}"


def sync_profile_to_disk(uid_str: str, profile_dict: Dict[str, Any]):
    """Write profile JSON to disk so LangGraph nodes can read it."""
    vault_file = Config.get_user_vault_file(uid_str)
    vault_file.parent.mkdir(parents=True, exist_ok=True)
    with open(vault_file, "w", encoding="utf-8") as f:
        json.dump(profile_dict, f, indent=2, ensure_ascii=False)


# ── Users ──

def create_user(db: Session, email: str, password: str) -> User:
    """Create a new user with an empty profile."""
    user = User(email=email, password=password)
    db.add(user)
    db.flush()  # get user.id

    uid_str = user_id_str(user.id)
    profile = _empty_profile(uid_str)

    db.add(UserProfile(user_id=user.id, profile_data=json.dumps(profile)))
    db.commit()
    db.refresh(user)

    # Create user directories + sync profile to disk
    Config.get_user_vault_file(uid_str)  # creates dirs via mkdir
    sync_profile_to_disk(uid_str, profile)

    return user


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    return db.query(User).filter(User.email == email).first()


def get_user_by_id(db: Session, user_id: int) -> Optional[User]:
    return db.query(User).filter(User.id == user_id).first()


# ── Profiles ──

def get_profile(db: Session, user_id: int) -> Dict[str, Any]:
    """Get user profile as dict."""
    row = db.query(UserProfile).filter(UserProfile.user_id == user_id).first()
    if row:
        return json.loads(row.profile_data)
    return _empty_profile(user_id_str(user_id))


def update_profile(db: Session, user_id: int, profile_dict: Dict[str, Any]):
    """Save profile to DB and sync to disk."""
    profile_dict["updated_at"] = datetime.now(timezone.utc).isoformat()
    row = db.query(UserProfile).filter(UserProfile.user_id == user_id).first()
    if row:
        row.profile_data = json.dumps(profile_dict)
    else:
        db.add(UserProfile(user_id=user_id, profile_data=json.dumps(profile_dict)))
    db.commit()

    sync_profile_to_disk(user_id_str(user_id), profile_dict)


# ── Chat Messages ──

def add_chat_message(
    db: Session, user_id: int, role: str, content: str, metadata: Optional[Dict] = None
):
    msg = ChatMessage(
        user_id=user_id,
        role=role,
        content=content,
        metadata_json=json.dumps(metadata) if metadata else None,
    )
    db.add(msg)
    db.commit()


def get_chat_history(db: Session, user_id: int, limit: int = 50) -> List[Dict[str, Any]]:
    rows = (
        db.query(ChatMessage)
        .filter(ChatMessage.user_id == user_id)
        .order_by(ChatMessage.created_at.asc())
        .limit(limit)
        .all()
    )
    return [
        {
            "role": r.role,
            "content": r.content,
            "metadata": json.loads(r.metadata_json) if r.metadata_json else None,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]


def clear_chat_history(db: Session, user_id: int):
    db.query(ChatMessage).filter(ChatMessage.user_id == user_id).delete()
    db.commit()


# ── Documents ──

def add_uploaded_document(
    db: Session, user_id: int, filename: str, file_type: str, num_chunks: int = 0
):
    doc = UploadedDocument(
        user_id=user_id, filename=filename, file_type=file_type, num_chunks=num_chunks
    )
    db.add(doc)
    db.commit()


def list_uploaded_documents(db: Session, user_id: int) -> List[Dict[str, Any]]:
    rows = (
        db.query(UploadedDocument)
        .filter(UploadedDocument.user_id == user_id)
        .order_by(UploadedDocument.upload_date.desc())
        .all()
    )
    return [
        {
            "filename": r.filename,
            "file_type": r.file_type,
            "num_chunks": r.num_chunks,
            "upload_date": r.upload_date.isoformat() if r.upload_date else None,
        }
        for r in rows
    ]


def delete_uploaded_document(db: Session, user_id: int, filename: str):
    db.query(UploadedDocument).filter(
        UploadedDocument.user_id == user_id,
        UploadedDocument.filename == filename,
    ).delete()
    db.commit()
