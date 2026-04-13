"""SQLAlchemy database setup for MySQL."""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from src.config import Config

engine = create_engine(
    f"mysql+pymysql://{Config.MYSQL_USER}:{Config.MYSQL_PASSWORD}@{Config.MYSQL_HOST}/{Config.MYSQL_DB}",
    echo=False,
)

SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


def init_db():
    """Create all tables."""
    from . import models  # noqa: F401 — registers models with Base
    Base.metadata.create_all(bind=engine)
