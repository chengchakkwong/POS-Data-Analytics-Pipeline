"""Incremental Hive-partitioned sales parquet cache."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pandas as pd

from pos_pipeline.config import SALES_PARQUET_DIR
from pos_pipeline.extraction.sales import DEFAULT_SALES_START, fetch_daily_sales
from pos_pipeline.storage.parquet_utils import (
    load_sales_max_date,
    load_sales_partitions,
)


def sync_daily_sales_parquet(cache_dir: Path | None = None) -> pd.DataFrame:
    """Fetch new sales and merge only affected year/month partitions."""
    cache_dir = cache_dir or SALES_PARQUET_DIR

    if cache_dir.exists():
        last_date = load_sales_max_date(cache_dir)
        start_date = (last_date - timedelta(days=1)).strftime("%Y-%m-%d")
        print(f"[sales] incremental from {start_date}")
    else:
        start_date = DEFAULT_SALES_START
        print(f"[sales] no cache; full extract from {start_date}")

    df_new = fetch_daily_sales(start_date=start_date)
    if df_new.empty:
        print("[sales] no new rows")
        return df_new

    df_new["rDate"] = pd.to_datetime(df_new["rDate"])
    df_new["year"] = df_new["rDate"].dt.year
    df_new["month"] = df_new["rDate"].dt.month

    affected = df_new[["year", "month"]].drop_duplicates()
    if cache_dir.exists():
        df_old = load_sales_partitions(cache_dir, affected)
    else:
        df_old = pd.DataFrame()

    df_combined = pd.concat([df_old, df_new], ignore_index=True)
    df_combined = df_combined.drop_duplicates(
        subset=["GoodsID", "rDate"],
        keep="last",
    )

    cache_dir.mkdir(parents=True, exist_ok=True)
    df_combined.to_parquet(
        cache_dir,
        engine="pyarrow",
        partition_cols=["year", "month"],
        compression="snappy",
        existing_data_behavior="delete_matching",
    )

    print(f"[sales] wrote {len(df_combined)} rows in affected partitions")
    return df_combined