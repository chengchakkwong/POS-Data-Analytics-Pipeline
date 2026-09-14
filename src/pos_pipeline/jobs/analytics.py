"""Biweekly ABC/XYZ + target-stock job."""

from pos_pipeline.config import PROJECT_ROOT, SALES_PARQUET_DIR
from pos_pipeline.database.connection import check_connection
from pos_pipeline.storage.sales_cache import sync_daily_sales_parquet


def run() -> int:
    print(f"[job:analytics] project root: {PROJECT_ROOT}")

    if not check_connection():
        print("[job:analytics] ERROR: cannot connect to POS database")
        return 1

    print("[job:analytics] database connection OK")

    df = sync_daily_sales_parquet()
    print(f"[job:analytics] sales rows in affected partitions: {len(df)}")
    print(f"[job:analytics] cache: {SALES_PARQUET_DIR}")
    return 0