import os
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Local development is zero-configuration with SQLite. Docker/production can
# override this with a PostgreSQL URL through DATABASE_URL.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SQLITE_URL = f"sqlite:///{(PROJECT_ROOT / 'finscore_local.db').as_posix()}"
SQLALCHEMY_DATABASE_URL = os.getenv(
    "DATABASE_URL",
    DEFAULT_SQLITE_URL,
)

connect_args = {"check_same_thread": False} if SQLALCHEMY_DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
