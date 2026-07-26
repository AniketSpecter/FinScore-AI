"""Streamlit Community Cloud entrypoint for the complete FinScore demo."""

from __future__ import annotations

import os
import sys
from pathlib import Path


# Community Cloud adds the entrypoint directory (``frontend``) to sys.path,
# not necessarily the repository root. Add the root before importing the
# frontend package or backend modules.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
