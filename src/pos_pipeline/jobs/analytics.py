"""Biweekly ABC/XYZ + target-stock job."""

from pos_pipeline.analysis.abc_xyz import run_abc_xyz
from pos_pipeline.config import ABC_XYZ_CSV, PROJECT_ROOT, SALES_PARQUET_DIR
from pos_pipeline.database.connection import check_connection
from pos_pipeline.storage.sales_cache import sync_daily_sales_parquet
from pos_pipeline.extraction.stock import fetch_stock_master, save_stock_master

def run() -> int:
    print(f"[job:analytics] project root: {PROJECT_ROOT}")

    if not check_connection():
        print("[job:analytics] ERROR: cannot connect to POS database")
        return 1

    print("[job:analytics] database connection OK")

    stock_df = fetch_stock_master()
    print(f"[job:analytics] stock master rows: {len(stock_df)}")
    if stock_df.empty:
        print("[job:analytics] ERROR: stock master is empty")
        return 1

    stock_path = save_stock_master(stock_df)
    print(f"[job:analytics] saved: {stock_path}")

    df = sync_daily_sales_parquet()
    print(f"[job:analytics] sales rows in affected partitions: {len(df)}")
    print(f"[job:analytics] cache: {SALES_PARQUET_DIR}")

    abc_xyz_df = run_abc_xyz()
    print(f"[job:analytics] ABC/XYZ SKUs: {len(abc_xyz_df)}")
    print(f"[job:analytics] wrote: {ABC_XYZ_CSV}")
    return 0