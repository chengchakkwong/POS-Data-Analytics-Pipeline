"""Run ABC/XYZ classification from local stock and sales cache."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from parquet_utils import load_sales_parquet
from pos_pipeline.analysis.abc import analyze_profit_abc
from pos_pipeline.analysis.monthly_series import resolve_complete_month_cutoff
from pos_pipeline.analysis.xyz import analyze_xyz
from pos_pipeline.config import ABC_XYZ_CSV, SALES_PARQUET_DIR, STOCK_MASTER_CSV


def attach_xyz_and_strategy(
    abc_df: pd.DataFrame,
    xyz_df: pd.DataFrame,
) -> pd.DataFrame:
    """把 XYZ 貼上 ABC 結果，並加上策略字串。"""
    final_df = pd.merge(
        abc_df,
        xyz_df[["GoodsID", "CV", "XYZ_Class", "Mean_Monthly_Qty"]],
        on="GoodsID",
        how="left",
    )
    final_df["XYZ_Class"] = final_df["XYZ_Class"].fillna("Z")

    # 新品不進 X/Y/Z，與 ABC 的 New 對齊
    is_new_abc = final_df["ABC_Class"] == "New"
    final_df.loc[is_new_abc, "XYZ_Class"] = "New"

    def get_strategy(row):
        if row["ABC_Class"] == "New":
            return "新品觀察 (手動控貨)"
        if row["ABC_Class"] == "Excluded":
            return "排除對象"

        combo = f"{row['ABC_Class']}{row['XYZ_Class']}"
        strategy_map = {
            "AX": "自動補貨 (高頻穩定)",
            "AY": "AI 季節性預測 (重點對象)",
            "AZ": "高安全庫存 (利潤高但難抓)",
            "BX": "定期補貨",
            "BY": "季節性補貨",
            "BZ": "觀望/依訂單進貨",
            "CX": "基本品 (低庫存管理)",
            "CY": "季節品 (減少庫存)",
            "CZ": "不建議預測 (考慮汰換)",
        }
        return strategy_map.get(combo, "其他")

    final_df["Strategy"] = final_df.apply(get_strategy, axis=1)
    final_df["displayname"] = final_df["Name"] + " | " + final_df["ProductCode"]

    final_df["SortOrder"] = final_df["ABC_Class"].map(
        {"New": 0, "A": 1, "B": 2, "C": 3, "Excluded": 4}
    ).fillna(5)
    return final_df.sort_values(
        by=["SortOrder", "Monthly_Avg_Profit"],
        ascending=[True, False],
    ).drop(columns=["SortOrder"])


def run_abc_xyz(
    stock_csv: Path | None = None,
    sales_dir: Path | None = None,
    output_csv: Path | None = None,
    as_of: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Classify SKUs and write data/insights/abc_xyz_analysis.csv."""
    stock_csv = stock_csv or STOCK_MASTER_CSV
    sales_dir = sales_dir or SALES_PARQUET_DIR
    output_csv = output_csv or ABC_XYZ_CSV

    df_stock = pd.read_csv(stock_csv)
    df_sales = load_sales_parquet(sales_dir)
    required_sales = {"GoodsID", "rDate", "TotalQty", "TotalAmt"}
    missing_sales = required_sales.difference(df_sales.columns)
    if missing_sales:
        missing_names = ", ".join(sorted(missing_sales))
        raise ValueError(f"sales data is missing required columns: {missing_names}")
    if df_sales.empty:
        raise ValueError("sales data is empty; ABC/XYZ cannot be calculated")

    df_sales["rDate"] = pd.to_datetime(df_sales["rDate"], errors="raise")
    if as_of is not None:
        df_sales = df_sales[df_sales["rDate"] <= pd.Timestamp(as_of)].copy()
        if df_sales.empty:
            raise ValueError("sales data has no rows on or before as_of")

    cutoff_month = resolve_complete_month_cutoff(df_sales["rDate"], as_of=as_of)
    start_month_abc = cutoff_month - pd.DateOffset(months=11)
    cutoff_end = cutoff_month + pd.offsets.MonthEnd(0)
    df_sales_recent12 = df_sales[
        df_sales["rDate"].between(start_month_abc, cutoff_end)
    ].copy()

    first_sale_full = (
        df_sales.groupby("GoodsID")["rDate"]
        .min()
        .reset_index()
        .rename(columns={"rDate": "FirstSaleDate"})
    )
    month_age = (
        (cutoff_month.year - first_sale_full["FirstSaleDate"].dt.year) * 12
        + (cutoff_month.month - first_sale_full["FirstSaleDate"].dt.month)
    ).clip(lower=1)
    month_age_map = dict(zip(first_sale_full["GoodsID"], month_age))

    abc_df = analyze_profit_abc(
        df_stock,
        df_sales_recent12,
        month_age_map=month_age_map,
    )
    xyz_df = analyze_xyz(df_sales, cutoff_month=cutoff_month)
    abc_xyz_df = attach_xyz_and_strategy(abc_df, xyz_df)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    abc_xyz_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return abc_xyz_df