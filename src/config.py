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
    
    # ChromaDB Settings
    CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
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
    def get_user_vault_file(cls, user_id: str) -> Path:
        """Get path to user vault JSON file"""
        vault_dir = Path(cls.USER_VAULT_PATH)
        vault_dir.mkdir(parents=True, exist_ok=True)
        return vault_dir / f"{user_id}.json"
    
    @classmethod
    def get_documents_path(cls) -> Path:
        """Get path to documents directory"""
        return Path(cls.DOCUMENTS_PATH)
