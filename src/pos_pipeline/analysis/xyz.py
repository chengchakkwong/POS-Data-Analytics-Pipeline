"""XYZ demand-variability classification."""

from __future__ import annotations

import pandas as pd

from pos_pipeline.analysis.monthly_series import (
    build_monthly_series,
    resolve_complete_month_cutoff,
)


RESULT_COLUMNS = [
    "GoodsID",
    "CV",
    "XYZ_Class",
    "Mean_Monthly_Qty",
]


def _classify_cv(cv: float) -> str:
    """Map a coefficient of variation to X/Y/Z."""
    if pd.isna(cv) or cv > 1.0:
        return "Z"
    if cv <= 0.5:
        return "X"
    return "Y"


def analyze_xyz(
    sales_df: pd.DataFrame,
    cutoff_month: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """依每個 SKU 的完整月銷量波動標 XYZ。

    每個 SKU 從自己的首賣月開始，到共同的完整截止月為止；期間沒有
    銷售的月份補 0，首賣前的月份不納入。沒有可計算變異的資料時，
    CV 保留為缺值並保守標為 Z，而不是以任意的大數字代替。
    """
    required = {"GoodsID", "rDate", "TotalQty"}
    missing = required.difference(sales_df.columns)
    if missing:
        missing_names = ", ".join(sorted(missing))
        raise ValueError(f"sales data is missing required columns: {missing_names}")
    if sales_df.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS)

    sales_df = sales_df.copy()
    sales_df["rDate"] = pd.to_datetime(sales_df["rDate"], errors="raise")
    sales_df["TotalQty"] = pd.to_numeric(sales_df["TotalQty"], errors="raise")
    cutoff = (
        pd.Timestamp(cutoff_month).to_period("M").to_timestamp()
        if cutoff_month is not None
        else resolve_complete_month_cutoff(sales_df["rDate"])
    )

    records: list[dict] = []
    for goods_id, item_sales in sales_df.groupby("GoodsID", sort=False):
        monthly = build_monthly_series(item_sales, cutoff)
        if monthly.empty:
            continue

        mean_qty = monthly.mean()
        std_qty = monthly.std()
        cv = std_qty / mean_qty if mean_qty > 0 and pd.notna(std_qty) else float("nan")
        records.append(
            {
                "GoodsID": goods_id,
                "CV": cv,
                "XYZ_Class": _classify_cv(cv),
                "Mean_Monthly_Qty": mean_qty,
            }
        )

    if not records:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    return pd.DataFrame.from_records(records, columns=RESULT_COLUMNS)