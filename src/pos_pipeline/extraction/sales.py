"""Extract daily sales aggregates from POS."""

from __future__ import annotations

import pandas as pd

from pos_pipeline.database.connection import execute_query

SALES_DAILY_SQL = """
SELECT D.GoodsID, M.rDate, SUM(D.Quantity) AS TotalQty, SUM(D.FinalAmt) AS TotalAmt
FROM dbo.SalesDetail AS D
JOIN dbo.SalesMaster AS M ON D.SalesMasterID = M.SID
WHERE CONVERT(date, CONVERT(varchar(8), M.rDate)) >= :start_date
GROUP BY D.GoodsID, M.rDate
"""

DEFAULT_SALES_START = "2024-01-01"


def fetch_daily_sales(start_date: str = DEFAULT_SALES_START) -> pd.DataFrame:
    """Fetch sales aggregated by GoodsID and rDate, from start_date onward."""
    return execute_query(
        SALES_DAILY_SQL,
        params={"start_date": start_date},
    )