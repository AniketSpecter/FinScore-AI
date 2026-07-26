"""Streamlit Community Cloud entrypoint for the complete FinScore demo."""

from __future__ import annotations

import os
from pathlib import Path

from frontend.cloud_runtime import ensure_backend


os.environ["FINSCORE_DEPLOYMENT_MODE"] = "cloud"
os.environ["API_URL"] = ensure_backend()

# Execute the same dashboard used by the Windows runner and Docker deployment.
# ``exec`` is intentional: Streamlit reruns the entrypoint after every widget
# interaction, while a normal import would execute frontend.app only once.
dashboard_path = Path(__file__).with_name("app.py")
dashboard_globals = {
    "__file__": str(dashboard_path),
    "__name__": "__main__",
    "__package__": None,
}
exec(compile(dashboard_path.read_bytes(), str(dashboard_path), "exec"), dashboard_globals)
