"""Target-stock planning from ABC/XYZ labels and local caches."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pos_pipeline.config import ABC_XYZ_CSV, STOCK_MASTER_CSV, TARGET_STOCK_CSV


def plan_c_class_row(row: pd.Series) -> dict:
    """C 類長尾：用月均量與 FirstOrderQty 算目標庫存（對齊舊腳本）。"""
    mean_qty = row.get("Mean_Monthly_Qty", 0)
    if pd.isna(mean_qty):
        mean_qty = 0
    curr_stock = row.get("CurrStock", 0)
    first_order_qty = row.get("FirstOrderQty", 0)
    if pd.isna(first_order_qty):
        first_order_qty = 0
    note = row.get("Note", "")
    if pd.isna(note):
        note = ""

    if mean_qty <= 0:
        target_stock = 0
    else:
        if pd.notna(first_order_qty) and first_order_qty > 0:
            if first_order_qty > (mean_qty * 12):
                target_stock = mean_qty * 1.2
            else:
                target_stock = first_order_qty
        else:
            target_stock = mean_qty * 1.2

    return {
        "ProductCode": row.get("ProductCode", "Unknown"),
        "Name": row.get("Name", "Unknown"),
        "ABC_XYZ": f"C{row['XYZ_Class']}",
        "Strategy": row.get("Strategy", "Unknown"),
        "CurrStock": curr_stock,
        "FirstOrderQty": first_order_qty,
        "Note": note,
        "Base_Demand": round(mean_qty, 2),
        "Final_Demand": round(mean_qty, 2),
        "Target_Stock": round(target_stock, 2),
    }


def run_target_stock(
    labels_csv: Path | None = None,
    stock_csv: Path | None = None,
    output_csv: Path | None = None,
) -> pd.DataFrame:
    """從本機 ABC/XYZ 與庫存 CSV 產出目標庫存。目前只處理 C 類。"""
    labels_csv = labels_csv or ABC_XYZ_CSV
    stock_csv = stock_csv or STOCK_MASTER_CSV
    output_csv = output_csv or TARGET_STOCK_CSV

    df_labels = pd.read_csv(labels_csv)
    df_stock = pd.read_csv(stock_csv).copy()

    if "FirstOrderQty" not in df_stock.columns:
        if "Note" in df_stock.columns:
            notes = df_stock["Note"].astype(str).replace("nan", "")
            df_stock["FirstOrderQty"] = pd.to_numeric(
                notes.str.extract(r"(?:^|\s)(\d+)(?:\s|$)", expand=False),
                errors="coerce",
            )
        else:
            df_stock["FirstOrderQty"] = 0

    analysis_df = pd.merge(
        df_labels[
            ["GoodsID", "ABC_Class", "XYZ_Class", "CV", "Mean_Monthly_Qty", "Strategy"]
        ],
        df_stock,
        on="GoodsID",
        how="inner",
    )
    analysis_df["FirstOrderQty"] = pd.to_numeric(
        analysis_df["FirstOrderQty"], errors="coerce"
    )

    c_class_skus = analysis_df[analysis_df["ABC_Class"] == "C"]
    forecast_df = pd.DataFrame(
        [plan_c_class_row(row) for _, row in c_class_skus.iterrows()]
    )
    if not forecast_df.empty:
        forecast_df = forecast_df.sort_values(by="Target_Stock", ascending=False)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    forecast_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return forecast_df