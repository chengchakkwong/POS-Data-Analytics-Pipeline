"""Biweekly ABC/XYZ + target-stock job."""

from __future__ import annotations

import os
import time
import traceback

from pos_pipeline.analysis.abc_xyz import run_abc_xyz
from pos_pipeline.analysis.target_stock import run_target_stock
from pos_pipeline.config import (
    ABC_XYZ_CSV,
    PROJECT_ROOT,
    SALES_PARQUET_DIR,
    TARGET_STOCK_CSV,
    TARGET_STOCK_TRACE_CSV,
)
from pos_pipeline.database.connection import check_connection
from pos_pipeline.extraction.stock import fetch_stock_master, save_stock_master
from pos_pipeline.storage.sales_cache import sync_daily_sales_parquet


def _elapsed_sec(started: float) -> float:
    return time.perf_counter() - started


def run() -> int:
    """Run the biweekly analytics job.

    Returns 0 on success. Any extract/classify/planning failure returns 1 and
    leaves the previous successful target-stock outputs untouched.
    """
    job_started = time.perf_counter()
    print(f"[job:analytics] project root: {PROJECT_ROOT}")

    if not check_connection():
        print("[job:analytics] ERROR: cannot connect to POS database")
        return 1

    print("[job:analytics] database connection OK")

    try:
        stage_started = time.perf_counter()
        stock_df = fetch_stock_master()
        print(f"[job:analytics] stock master rows: {len(stock_df)}")
        if stock_df.empty:
            print("[job:analytics] ERROR: stock master is empty")
            return 1

        stock_path = save_stock_master(stock_df)
        print(f"[job:analytics] saved: {stock_path}")
        print(f"[job:analytics] stock elapsed: {_elapsed_sec(stage_started):.1f}s")

        stage_started = time.perf_counter()
        synced = sync_daily_sales_parquet()
        print(f"[job:analytics] sales rows in affected partitions: {len(synced)}")
        print(f"[job:analytics] cache: {SALES_PARQUET_DIR}")
        print(f"[job:analytics] sales sync elapsed: {_elapsed_sec(stage_started):.1f}s")

        stage_started = time.perf_counter()
        abc_xyz_df = run_abc_xyz(sales_dir=SALES_PARQUET_DIR)
        print(f"[job:analytics] ABC/XYZ SKUs: {len(abc_xyz_df)}")
        print(f"[job:analytics] wrote: {ABC_XYZ_CSV}")
        print(f"[job:analytics] ABC/XYZ elapsed: {_elapsed_sec(stage_started):.1f}s")

        backend_name = os.getenv("FORECAST_BACKEND", "prophet")
        print(f"[job:analytics] forecast backend: {backend_name}")

        stage_started = time.perf_counter()
        target_stock_df = run_target_stock(
            sales_dir=SALES_PARQUET_DIR,
            backend_name=backend_name,
        )
        print(f"[job:analytics] target-stock SKUs: {len(target_stock_df)}")
        print(f"[job:analytics] wrote: {TARGET_STOCK_CSV}")
        print(f"[job:analytics] wrote: {TARGET_STOCK_TRACE_CSV}")
        print(
            f"[job:analytics] target-stock elapsed: {_elapsed_sec(stage_started):.1f}s"
        )
        print(f"[job:analytics] total elapsed: {_elapsed_sec(job_started):.1f}s")
        return 0
    except Exception as exc:
        print(f"[job:analytics] ERROR: {exc}")
        print(f"[job:analytics] failed after: {_elapsed_sec(job_started):.1f}s")
        traceback.print_exc()
        return 1
