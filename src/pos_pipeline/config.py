"""Central configuration for the POS pipeline."""

from pathlib import Path
from dotenv import load_dotenv
import os


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
SALES_PARQUET_DIR = PROCESSED_DIR / "sales_daily_parquet"


FIREBASE_KEY_PATH = Path(
    os.getenv("FIREBASE_KEY_PATH", str(PROJECT_ROOT / "serviceAccountKey.json"))
)