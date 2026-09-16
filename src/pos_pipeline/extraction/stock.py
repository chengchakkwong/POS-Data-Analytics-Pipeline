"""Stock master extraction."""

from __future__ import annotations

import pandas as pd

from pathlib import Path

from pos_pipeline.config import PROCESSED_DIR, STOCK_MASTER_CSV

from pos_pipeline.database.connection import execute_query

STOCK_MASTER_SQL = """
SELECT
    g.SID AS GoodsID,
    g.ID AS ProductCode,
    g.Barcode,
    g.Name,
    g.Note,
    s.CurrStock,
    g.RetailPrice,
    g.LastInCost,
    g.AvgCost,
    d.Name AS Category,
    t1.Name AS InboundLocation,
    t2.Name AS Supplier
FROM dbo.GoodsInfo g
LEFT JOIN dbo.GoodsStock s ON g.SID = s.GoodsID AND s.ShopID = 1
LEFT JOIN dbo.Dept d ON g.DeptID = d.SID
LEFT JOIN dbo.ProductType1 t1 ON g.ProductType1ID = t1.SID
LEFT JOIN dbo.ProductType2 t2 ON g.ProductType2ID = t2.SID
"""


def fetch_stock_master() -> pd.DataFrame:
    """Fetch and lightly clean the stock master table."""
    df = execute_query(STOCK_MASTER_SQL)
    if df.empty:
        return df

    text_cols = ["Name", "Note", "Category", "InboundLocation", "Supplier"]
    for col in text_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(r"[\n\r\t]+", " ", regex=True)
                .str.strip()
            )
    return df


def save_stock_master(df: pd.DataFrame, path: Path | None = None) -> Path:
    """Save stock master DataFrame to CSV and return the path."""
    output = path or STOCK_MASTER_CSV
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False, encoding="utf-8-sig")
    return output