"""Central configuration for the POS pipeline."""

from pathlib import Path
from dotenv import load_dotenv

def find_project_root(start: Path | None = None) -> Path:
    """Walk upward until pyproject.toml is found."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError("Cannot find project root (pyproject.toml).")


PROJECT_ROOT = find_project_root()
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
STOCK_MASTER_CSV = PROCESSED_DIR / "DetailGoodsStockToday.csv"