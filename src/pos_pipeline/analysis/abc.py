"""Profit-based ABC classification."""

from __future__ import annotations

import numpy as np
import pandas as pd


def analyze_profit_abc(
    stock_df: pd.DataFrame,
    sales_df: pd.DataFrame,
    conservative_cost_ratio: float = 0.80,
    month_age_map: dict | None = None,
) -> pd.DataFrame:
    """依傳入的 sales 期間做利潤 ABC。

    month_age_map 若有提供，年資用全歷史首賣日，避免只看近 12 個月
    而把舊品判成新品。未提供時，年資改由這份 sales_df 自己算。
    """
    sales_df["rDate"] = pd.to_datetime(sales_df["rDate"])

    sales_summary = sales_df.groupby("GoodsID").agg(
        {
            "TotalQty": "sum",
            "TotalAmt": "sum",
            "rDate": ["nunique", "min"],
        }
    )
    sales_summary.columns = ["TotalQty", "TotalAmt", "SalesDays", "FirstSaleDate"]
    sales_summary = sales_summary.reset_index()

    if month_age_map is None:
        today = sales_df["rDate"].max()
        sales_summary["Month_Age"] = (
            (today.year - sales_summary["FirstSaleDate"].dt.year) * 12
            + (today.month - sales_summary["FirstSaleDate"].dt.month)
        )
        # 最少算 1 個月，避免後面除以零
        sales_summary["Month_Age"] = sales_summary["Month_Age"].clip(lower=1)
    else:
        sales_summary["Month_Age"] = np.nan

    merged_df = pd.merge(sales_summary, stock_df, on="GoodsID", how="outer")

    if month_age_map is not None:
        # 年資用全歷史首賣，避免只看近 12 個月而把舊品判成新品
        merged_df["Month_Age"] = merged_df["GoodsID"].map(month_age_map)

    numeric_cols = [
        "TotalQty",
        "TotalAmt",
        "CurrStock",
        "RetailPrice",
        "LastInCost",
        "AvgCost",
    ]
    merged_df[numeric_cols] = merged_df[numeric_cols].fillna(0)
    # 沒有銷售紀錄：不當新品，當成老品
    merged_df["Month_Age"] = merged_df["Month_Age"].fillna(99).clip(lower=1)

    mask_missing_info = merged_df["ProductCode"].isna()
    if mask_missing_info.any():
        deleted_labels = (
            "Deleted (ID: "
            + merged_df["GoodsID"].astype(str).str.split(".").str[0]
            + ")"
        )
        merged_df.loc[mask_missing_info, "ProductCode"] = deleted_labels
        merged_df.loc[mask_missing_info, "Name"] = deleted_labels

    merged_df["Barcode"] = merged_df["Barcode"].fillna("DELETED")

    text_cols = ["Category", "Supplier", "Note", "InboundLocation"]
    for col in text_cols:
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].fillna("DELETED")

    # 膠袋、雜項不進 ABC 排名
    generic_conditions = (
        (merged_df["ProductCode"].astype(str) == "202320232023")
        | (merged_df["Name"].str.contains("五金家品|Deleted|膠袋徵費|塑膠袋", na=False))
    )
    merged_df["Is_Generic"] = np.where(generic_conditions, "Yes", "No")

    unit_price = np.where(
        merged_df["TotalQty"] > 0,
        merged_df["TotalAmt"] / merged_df["TotalQty"],
        0,
    )
    cost_missing = merged_df["LastInCost"] <= 0

    # 缺成本，或售價成本比 > 9：改用賣價 * 0.8
    cost_suspicious = np.zeros(len(merged_df), dtype=bool)
    valid_cost_mask = merged_df["LastInCost"] > 0
    cost_suspicious[valid_cost_mask] = (
        unit_price[valid_cost_mask]
        / merged_df.loc[valid_cost_mask, "LastInCost"]
    ) > 9

    merged_df["AdjustedCost"] = np.where(
        cost_missing | cost_suspicious,
        unit_price * conservative_cost_ratio,
        merged_df["LastInCost"],
    )

    merged_df["TotalCost"] = merged_df["AdjustedCost"] * merged_df["TotalQty"]
    merged_df["TotalProfit"] = merged_df["TotalAmt"] - merged_df["TotalCost"]
    # 時間公平：避免只因賣得久而變成 A
    merged_df["Monthly_Avg_Profit"] = merged_df["TotalProfit"] / merged_df["Month_Age"]

    is_calc = merged_df["Is_Generic"] == "No"
    df_calc = merged_df[is_calc].copy()
    df_excl = merged_df[~is_calc].copy()

    df_calc["ProfitCumulativeRatio"] = 1.0
    df_calc["ABC_Class"] = "C"

    # 先拿掉新品與泛用品；只對成熟且利潤為正的做 70/20/10
    is_new = df_calc["Month_Age"] < 4
    mature_mask = ~is_new

    mature_df = df_calc[mature_mask & (df_calc["Monthly_Avg_Profit"] > 0)].copy()
    mature_df = mature_df.sort_values(by="Monthly_Avg_Profit", ascending=False)

    total_prof_mature = mature_df["Monthly_Avg_Profit"].sum()
    if total_prof_mature > 0:
        mature_df["CumulativeProfit"] = mature_df["Monthly_Avg_Profit"].cumsum()
        mature_df["ProfitCumulativeRatio"] = (
            mature_df["CumulativeProfit"] / total_prof_mature
        )
    else:
        mature_df["ProfitCumulativeRatio"] = 1.0

    mature_df["ABC_Class"] = np.select(
        [
            (mature_df["ProfitCumulativeRatio"] <= 0.7),
            (mature_df["ProfitCumulativeRatio"] <= 0.9),
        ],
        ["A", "B"],
        default="C",
    )

    df_calc.loc[mature_df.index, "ProfitCumulativeRatio"] = mature_df[
        "ProfitCumulativeRatio"
    ]
    df_calc.loc[mature_df.index, "ABC_Class"] = mature_df["ABC_Class"]
    df_calc.loc[is_new, "ABC_Class"] = "New"

    df_excl["CumulativeProfit"] = np.nan
    df_excl["ProfitCumulativeRatio"] = np.nan
    df_excl["ABC_Class"] = "Excluded"

    return pd.concat([df_calc, df_excl], ignore_index=True)