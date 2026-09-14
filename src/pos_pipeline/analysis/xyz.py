"""XYZ demand-variability classification."""

from __future__ import annotations

import numpy as np
import pandas as pd


def analyze_xyz(sales_df: pd.DataFrame) -> pd.DataFrame:
    """依月銷量波動標 XYZ。回傳 GoodsID, CV, XYZ_Class, Mean_Monthly_Qty。"""
    sales_df = sales_df.copy()
    sales_df["rDate"] = pd.to_datetime(sales_df["rDate"])

    monthly_matrix = (
        sales_df.groupby(["GoodsID", pd.Grouper(key="rDate", freq="MS")])["TotalQty"]
        .sum()
        .unstack(fill_value=0)
    )

    stats = pd.DataFrame(index=monthly_matrix.index)
    stats["Mean_Monthly_Qty"] = monthly_matrix.mean(axis=1)
    stats["Std_Monthly_Qty"] = monthly_matrix.std(axis=1)
    # CV = 月銷量標準差 / 平均；均量為 0 則視為極不穩定
    stats["CV"] = np.where(
        stats["Mean_Monthly_Qty"] > 0,
        stats["Std_Monthly_Qty"] / stats["Mean_Monthly_Qty"],
        9.99,
    )

    # X 穩 (≤0.5)、Y 中 (≤1.0)、Z 亂
    conditions = [
        (stats["CV"] <= 0.5),
        (stats["CV"] <= 1.0),
        (stats["CV"] > 1.0),
    ]
    stats["XYZ_Class"] = np.select(conditions, ["X", "Y", "Z"], default="Z")

    return stats.reset_index()[
        ["GoodsID", "CV", "XYZ_Class", "Mean_Monthly_Qty"]
    ]