"""Incremental extraction of inbound stock movements."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from pos_pipeline.database.connection import execute_query


INBOUND_MOVEMENTS_SQL = """
SELECT
    SID,
    BillDate,
    GoodsNo,
    Barcode,
    GoodsName1,
    OriQty,
    ChQty,
    NewQty,
    ProductType2Name1,
    invNo,
    Note
FROM dbo.GoodsStockMovement
WHERE MoveTypeID = 1
  AND SID > :last_sid
  AND BillDate >= :bill_start
ORDER BY SID
"""


def fetch_new_inbound_movements(
    last_sid: int = 0,
    days: int = 14,
    *,
    as_of: date | None = None,
) -> pd.DataFrame:
    """Fetch recent inbound movements newer than the SID watermark."""
    reference_date = as_of or date.today()
    bill_start = int(
        (reference_date - timedelta(days=days)).strftime("%Y%m%d")
    )

    return execute_query(
        INBOUND_MOVEMENTS_SQL,
        params={
            "last_sid": last_sid,
            "bill_start": bill_start,
        },
    )


INBOUND_FOR_MIN_MULTIPLE_SQL = """
SELECT
    GoodsNo,
    ChQty,
    BillDate
FROM dbo.GoodsStockMovement
WHERE MoveTypeID = 1
  AND ChQty > 0
  AND BillDate >= :bill_start
  AND BillDate <= :bill_end
ORDER BY GoodsNo, BillDate
"""


def fetch_inbound_for_min_multiple(
    years: int = 2,
    *,
    as_of: date | None = None,
) -> pd.DataFrame:
    """Fetch inbound qty history used to guess MOQ / order multiple."""
    if years < 1:
        raise ValueError(f"years must be >= 1, got {years}")

    reference_date = as_of or date.today()
    start_date = reference_date - timedelta(days=int(years) * 365)
    bill_start = int(start_date.strftime("%Y%m%d"))
    bill_end = int(reference_date.strftime("%Y%m%d"))

    return execute_query(
        INBOUND_FOR_MIN_MULTIPLE_SQL,
        params={
            "bill_start": bill_start,
            "bill_end": bill_end,
        },
    )