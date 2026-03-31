"""Configuration management for SentinelFinance"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Centralized configuration management"""
    
    # Groq API (using Llama 3.1 70B)
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    
    # Vector DB Settings (FAISS)
    VECTOR_DB_PATH: str = os.getenv("VECTOR_DB_PATH", os.getenv("CHROMA_DB_PATH", "./data/chroma_db"))
    # Legacy/unused in FAISS; kept for backward compatibility
    CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "financial_knowledge")
    
    # Market Data APIs
    ALPHA_VANTAGE_API_KEY: Optional[str] = os.getenv("ALPHA_VANTAGE_API_KEY")
    YAHOO_FINANCE_ENABLED: bool = os.getenv("YAHOO_FINANCE_ENABLED", "true").lower() == "true"
    
    # User Vault Settings
    USER_VAULT_PATH: str = os.getenv("USER_VAULT_PATH", "./data/user_profiles")
    DEFAULT_USER_ID: str = os.getenv("DEFAULT_USER_ID", "default_user")
    
    # Application Settings
    MAX_ITERATIONS: int = int(os.getenv("MAX_ITERATIONS", "10"))
    DEBUG_MODE: bool = os.getenv("DEBUG_MODE", "false").lower() == "true"
    
    # Document Ingestion
    DOCUMENTS_PATH: str = os.getenv("DOCUMENTS_PATH", "./data/documents")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    
    # Cache Settings
    SEARCH_CACHE_TTL: int = int(os.getenv("SEARCH_CACHE_TTL", "3600"))  # 1 hour
    
    @classmethod
    def validate(cls) -> bool:
        """Validate that required configuration is present"""
        if not cls.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is required. Please set it in .env file")
        return True
    
    @classmethod
    def get_user_dir(cls, user_id: str) -> Path:
        """Get root directory for a user's data"""
        user_dir = Path(cls.USER_VAULT_PATH) / user_id
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir

    @classmethod
    def get_user_vault_file(cls, user_id: str) -> Path:
        """Get path to user vault JSON file"""
        user_dir = cls.get_user_dir(user_id)
        new_path = user_dir / "profile.json"

        # Migration: move old flat file into subdirectory
        old_path = Path(cls.USER_VAULT_PATH) / f"{user_id}.json"
        if old_path.exists() and not new_path.exists():
            old_path.rename(new_path)
            print(f"Migrated profile: {old_path} -> {new_path}")

        return new_path

    @classmethod
    def get_user_faiss_path(cls, user_id: str) -> Path:
        """Get path to user's personal FAISS index"""
        faiss_path = cls.get_user_dir(user_id) / "faiss_index"
        faiss_path.mkdir(parents=True, exist_ok=True)
        return faiss_path

    @classmethod
    def get_user_documents_path(cls, user_id: str) -> Path:
        """Get path to user's uploaded documents directory"""
        docs_path = cls.get_user_dir(user_id) / "documents"
        docs_path.mkdir(parents=True, exist_ok=True)
        return docs_path

    @classmethod
    def get_user_manifest_file(cls, user_id: str) -> Path:
        """Get path to user's document manifest"""
        return cls.get_user_dir(user_id) / "manifest.json"

    @classmethod
    def get_documents_path(cls) -> Path:
        """Get path to global knowledge base documents directory"""
        return Path(cls.DOCUMENTS_PATH)
