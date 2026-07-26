"""Start the FinScore API inside a Streamlit Community Cloud process.

Community Cloud exposes only the Streamlit web process. The dashboard still
uses the exact FastAPI endpoints used locally, so this module starts that API
on an internal loopback port and stores demo history in an ephemeral SQLite
database. No applicant history is committed to GitHub.
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

API_HOST = "127.0.0.1"
API_PORT = int(os.getenv("FINSCORE_INTERNAL_API_PORT", "3022"))
API_URL = f"http://{API_HOST}:{API_PORT}"

_server_thread: threading.Thread | None = None
_start_lock = threading.Lock()


def _api_is_ready() -> bool:
    try:
        with urlopen(f"{API_URL}/ready", timeout=2) as response:
            return response.status == 200
    except (OSError, URLError):
        return False


def _serve_api() -> None:
    import uvicorn

    from backend.main import app

    uvicorn.run(
        app,
        host=API_HOST,
        port=API_PORT,
        log_level=os.getenv("FINSCORE_API_LOG_LEVEL", "warning"),
        access_log=False,
    )


def ensure_backend(timeout_seconds: int = 45) -> str:
    """Return the internal API URL after its readiness endpoint succeeds."""
    global _server_thread

    if _api_is_ready():
        return API_URL

    with _start_lock:
        if _server_thread is None or not _server_thread.is_alive():
            database_path = Path(tempfile.gettempdir()) / "finscore_cloud.db"
            os.environ.setdefault("DATABASE_URL", f"sqlite:///{database_path.as_posix()}")
            _server_thread = threading.Thread(
                target=_serve_api,
                name="finscore-embedded-api",
                daemon=True,
            )
            _server_thread.start()

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if _api_is_ready():
            return API_URL
        if _server_thread is not None and not _server_thread.is_alive():
            break
        time.sleep(0.25)

    raise RuntimeError(
        "The embedded FinScore API did not become ready. Check the Streamlit deployment logs."
    )
