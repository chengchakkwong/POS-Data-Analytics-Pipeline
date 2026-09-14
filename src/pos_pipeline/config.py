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
INSIGHTS_DIR = DATA_DIR / "insights"
ABC_XYZ_CSV = INSIGHTS_DIR / "abc_xyz_analysis.csv"
TARGET_STOCK_CSV = INSIGHTS_DIR / "target_stock_plan.csv"


FIREBASE_KEY_PATH = Path(
    os.getenv("FIREBASE_KEY_PATH", str(PROJECT_ROOT / "serviceAccountKey.json"))
)