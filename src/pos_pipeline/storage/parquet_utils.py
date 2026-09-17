"""Hive-aware Parquet helpers for the sales cache."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds


def _hive_dataset(path: Path) -> ds.Dataset:
    return ds.dataset(str(path), format="parquet", partitioning="hive")


def load_sales_parquet(path: str | Path) -> pd.DataFrame:
    """Read a single parquet file or a Hive-partitioned sales cache directory."""
    path = Path(path)
    if path.is_dir():
        return _hive_dataset(path).to_table().to_pandas()
    return pd.read_parquet(path)


def load_sales_max_date(path: str | Path) -> pd.Timestamp:
    """Return the latest rDate from sales parquet cache."""
    path = Path(path)
    if not path.is_dir():
        df = pd.read_parquet(path, columns=["rDate"])
        return pd.to_datetime(df["rDate"].max())

    table = _hive_dataset(path).to_table(columns=["rDate"])
    return pd.to_datetime(pc.max(table["rDate"]).as_py())


def load_sales_partitions(path: str | Path, partitions: pd.DataFrame) -> pd.DataFrame:
    """Read only the year/month partitions listed in the partitions DataFrame."""
    path = Path(path)
    if partitions.empty:
        return pd.DataFrame()

    if not path.is_dir():
        return pd.read_parquet(path)

    expr = None
    for _, row in partitions.iterrows():
        part_expr = (ds.field("year") == int(row["year"])) & (
            ds.field("month") == int(row["month"])
        )
        expr = part_expr if expr is None else (expr | part_expr)

    return _hive_dataset(path).to_table(filter=expr).to_pandas()
