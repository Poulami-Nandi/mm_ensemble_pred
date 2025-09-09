# src/mm_ensemble/utils/paths.py
from pathlib import Path
import os

def _find_project_root() -> Path:
    """
    Heuristic: walk up from this file and return the first parent
    that looks like the repo root (has pyproject.toml or .git).
    Fallback to three levels up from here.
    """
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "pyproject.toml").exists() or (p / ".git").exists():
            return p
    # Fallback: repo_root/src/mm_ensemble/utils/paths.py -> repo_root
    try:
        return here.parents[3]
    except IndexError:
        return here.parent

# Allow overrides via env vars; otherwise use repo-relative defaults
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", _find_project_root()))
DATA_DIR     = Path(os.environ.get("DATA_DIR", PROJECT_ROOT / "data"))
OUTPUTS_DIR  = Path(os.environ.get("OUTPUTS_DIR", PROJECT_ROOT / "outputs"))

# Ensure directories exist when something tries to write outputs
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
