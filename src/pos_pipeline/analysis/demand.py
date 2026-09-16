"""Demand-rule helpers for New / X-Y / Z planning (no model backends yet)."""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from pos_pipeline.analysis.monthly_series import (
    build_monthly_series,
    resolve_complete_month_cutoff,
    trailing_complete_months,
)

_NOTE_NUMBER_RE = re.compile(r"\d+")
_NEW_SEASONAL_MIN_COMPLETE_MONTHS = 24
_NEW_SEASONAL_FLOOR = 0.8
_NEW_SEASONAL_CEIL = 1.2


def parse_first_order_qty(note: object) -> tuple[float | None, str]:
    """Parse FirstOrderQty from a free-text Note.

    Policy (human-reviewed):
    - exactly one positive integer sequence → use it
    - no numbers or two+ numbers → missing (do not guess)
    """
    if note is None or (isinstance(note, float) and pd.isna(note)):
        return None, "no_number"
    text = str(note).strip()
    if not text or text.lower() == "nan":
        return None, "no_number"

    numbers = _NOTE_NUMBER_RE.findall(text)
    if not numbers:
        return None, "no_number"
    if len(numbers) >= 2:
        return None, "multiple_numbers"
    return float(numbers[0]), "single_number"


def base_demand_for_new(
    item_sales: pd.DataFrame,
    mean_qty: float = 0.0,
    *,
    as_of: str | pd.Timestamp | None = None,
) -> float:
    """新品：日均銷量 × 30；活躍天數 < 7 則上限為總銷 × 3。

    活躍天數分母從首賣日算到 ``as_of``（最後同步日）；未提供時用最後銷售日。
    沒有銷售列時回傳 0（不退回月均）。
    ``mean_qty`` 保留給後續防爆上限使用。
    """
    del mean_qty  # reserved for later caps; not used in the base rule
    if item_sales is None or item_sales.empty:
        return 0.0

    item_sales = item_sales.copy()
    item_sales["rDate"] = pd.to_datetime(item_sales["rDate"], errors="raise")
    first_sale_date = item_sales["rDate"].min()
    last_sale_date = item_sales["rDate"].max()
    end_date = pd.Timestamp(as_of) if as_of is not None else last_sale_date
    if end_date < first_sale_date:
        raise ValueError("as_of must not be before the first sale date")

    active_days = max((end_date - first_sale_date).days, 1)
    total_sold = float(pd.to_numeric(item_sales["TotalQty"], errors="raise").sum())
    base_pred = (total_sold / active_days) * 30
    if active_days < 7:
        base_pred = min(base_pred, total_sold * 3)
    return float(base_pred)


def base_demand_for_recent_mean(
    item_sales: pd.DataFrame,
    cutoff_month: str | pd.Timestamp,
    *,
    months: int = 3,
) -> float:
    """X/Y 資料不足時的基準：最近 N 個完整日曆月平均（含 0）。"""
    if item_sales is None or item_sales.empty:
        return 0.0

    monthly = build_monthly_series(item_sales, cutoff_month)
    if monthly.empty:
        return 0.0
    window = trailing_complete_months(monthly, months)
    return float(window.mean())


def base_demand_for_z(
    item_sales: pd.DataFrame,
    mean_qty: float = 0.0,
    *,
    cutoff_month: str | pd.Timestamp | None = None,
    as_of: str | pd.Timestamp | None = None,
) -> float:
    """Z 類：近 6 個完整日曆月高點；若等於歷史最高且至少兩個月有銷售，改用次高。

    沒有可用銷售時回傳 0（不退回月均）。
    ``mean_qty`` 保留給後續防爆上限使用。
    """
    del mean_qty
    if item_sales is None or item_sales.empty:
        return 0.0

    item_sales = item_sales.copy()
    item_sales["rDate"] = pd.to_datetime(item_sales["rDate"], errors="raise")
    resolved_cutoff = (
        pd.Timestamp(cutoff_month).to_period("M").to_timestamp()
        if cutoff_month is not None
        else resolve_complete_month_cutoff(item_sales["rDate"], as_of=as_of)
    )
    monthly = build_monthly_series(item_sales, resolved_cutoff)
    if monthly.empty:
        return 0.0

    historical_max = float(monthly.max())
    last_6m = trailing_complete_months(monthly, 6)
    if last_6m.empty:
        return 0.0

    last_6m_max = float(last_6m.max())
    nonzero_months = int((last_6m > 0).sum())
    if (
        len(last_6m) >= 2
        and last_6m_max == historical_max
        and last_6m_max > 0
        and nonzero_months >= 2
    ):
        return float(last_6m.nlargest(2).iloc[1])
    return last_6m_max


@dataclass(frozen=True)
class CategorySeasonalProfile:
    """Category-month seasonal indices plus data-quality gates."""

    index_map: dict[tuple[object, int], float]
    complete_month_counts: dict[object, int]


def calculate_category_seasonal_profile(
    sales_df: pd.DataFrame,
    stock_df: pd.DataFrame,
    cutoff_month: str | pd.Timestamp | None = None,
) -> CategorySeasonalProfile:
    """Build category × month seasonal indices on a complete-month calendar.

    Each category starts at its first sales month and is padded with zeros through
    ``cutoff_month``. Categories with fewer than 24 complete months remain in the
    map for inspection, but :func:`new_seasonal_factor` will ignore them.
    """
    required_sales = {"GoodsID", "rDate", "TotalQty"}
    missing_sales = required_sales.difference(sales_df.columns)
    if missing_sales:
        missing_names = ", ".join(sorted(missing_sales))
        raise ValueError(f"sales data is missing required columns: {missing_names}")
    if "GoodsID" not in stock_df.columns or "Category" not in stock_df.columns:
        raise ValueError("stock data must include GoodsID and Category")

    if sales_df.empty:
        return CategorySeasonalProfile(index_map={}, complete_month_counts={})

    sales_df = sales_df.copy()
    sales_df["rDate"] = pd.to_datetime(sales_df["rDate"], errors="raise")
    resolved_cutoff = (
        pd.Timestamp(cutoff_month).to_period("M").to_timestamp()
        if cutoff_month is not None
        else resolve_complete_month_cutoff(sales_df["rDate"])
    )

    df_with_cat = sales_df.merge(
        stock_df[["GoodsID", "Category"]],
        on="GoodsID",
        how="left",
    )
    index_map: dict[tuple[object, int], float] = {}
    complete_month_counts: dict[object, int] = {}

    for category, cat_sales in df_with_cat.groupby("Category", dropna=False):
        monthly = build_monthly_series(cat_sales, resolved_cutoff)
        if monthly.empty:
            complete_month_counts[category] = 0
            continue

        complete_month_counts[category] = int(len(monthly))
        cat_avg = float(monthly.mean())
        if cat_avg == 0:
            cat_avg = 1.0

        month_factors: dict[int, list[float]] = {}
        for month_start, qty in monthly.items():
            month_num = int(pd.Timestamp(month_start).month)
            month_factors.setdefault(month_num, []).append(float(qty) / cat_avg)
        for month_num, factors in month_factors.items():
            index_map[(category, month_num)] = float(sum(factors) / len(factors))

    return CategorySeasonalProfile(
        index_map=index_map,
        complete_month_counts=complete_month_counts,
    )


def calculate_category_seasonal_indices(
    sales_df: pd.DataFrame,
    stock_df: pd.DataFrame,
    cutoff_month: str | pd.Timestamp | None = None,
) -> dict[tuple[object, int], float]:
    """Compatibility wrapper returning only the category-month index map."""
    return calculate_category_seasonal_profile(
        sales_df,
        stock_df,
        cutoff_month=cutoff_month,
    ).index_map


def new_seasonal_factor(
    category: object,
    month: int,
    profile: CategorySeasonalProfile,
) -> float:
    """Seasonal factor for New SKUs only.

    Requires at least 24 complete category months; otherwise returns 1.0.
    When applied, the factor is clipped to [0.8, 1.2]. Z never uses this.
    """
    if profile.complete_month_counts.get(category, 0) < _NEW_SEASONAL_MIN_COMPLETE_MONTHS:
        return 1.0
    raw = float(profile.index_map.get((category, month), 1.0))
    return min(_NEW_SEASONAL_CEIL, max(_NEW_SEASONAL_FLOOR, raw))
