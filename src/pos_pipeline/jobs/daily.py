"""Daily extract + Firestore sync job (stub)."""

from pos_pipeline.config import PROJECT_ROOT
from pos_pipeline.database.connection import check_connection


def run() -> int:
    print(f"[job:daily] project root: {PROJECT_ROOT}")

    if not check_connection():
        print("[job:daily] ERROR: cannot connect to POS database")
        return 1

    print("[job:daily] database connection OK")

    from pos_pipeline.extraction.stock import fetch_stock_master, save_stock_master

    df = fetch_stock_master()
    print(f"[job:daily] stock master rows: {len(df)}")
    if df.empty:
        print("[job:daily] ERROR: stock master is empty")
        return 1

    output = save_stock_master(df)
    print(f"[job:daily] saved: {output}")
    return 0