"""Shared calendar-month handling for analytics and forecasting."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def _month_start(value: str | pd.Timestamp) -> pd.Timestamp:
    """Normalize a date-like value to the first day of its month."""
    return pd.Timestamp(value).to_period("M").to_timestamp()


def resolve_complete_month_cutoff(
    dates: Iterable,
    as_of: str | pd.Timestamp | None = None,
) -> pd.Timestamp:
    """Return the latest month that is known to be complete.

    The usable data boundary is the earlier of ``as_of`` and the latest
    available date. A month is complete only when that boundary is month-end;
    otherwise the preceding month is returned.
    """
    parsed_dates = pd.to_datetime(pd.Series(dates), errors="raise").dropna()
    if parsed_dates.empty:
        raise ValueError("cannot resolve a complete-month cutoff without dates")

    data_boundary = parsed_dates.max()
    if as_of is not None:
        data_boundary = min(data_boundary, pd.Timestamp(as_of))

    boundary_month = _month_start(data_boundary)
    if data_boundary.is_month_end:
        return boundary_month
    return boundary_month - pd.DateOffset(months=1)


def build_monthly_series(
    sales_df: pd.DataFrame,
    cutoff_month: str | pd.Timestamp,
    *,
    date_col: str = "rDate",
    value_col: str = "TotalQty",
    start_month: str | pd.Timestamp | None = None,
) -> pd.Series:
    """Aggregate sales into a continuous month-start series.

    By default the series starts at the first available sales row and ends at
    ``cutoff_month``. Missing months inside that observed lifetime, including
    trailing months, are represented by zero. Months before the first sales
    row are not invented unless ``start_month`` is explicitly supplied.
    """
    required = {date_col, value_col}
    missing = required.difference(sales_df.columns)
    if missing:
        missing_names = ", ".join(sorted(missing))
        raise ValueError(f"sales data is missing required columns: {missing_names}")

    cutoff = _month_start(cutoff_month)
    if sales_df.empty:
        return pd.Series(
            dtype="float64",
            index=pd.DatetimeIndex([], freq="MS"),
            name=value_col,
        )

    working = sales_df[[date_col, value_col]].copy()
    working[date_col] = pd.to_datetime(working[date_col], errors="raise")
    working[value_col] = pd.to_numeric(working[value_col], errors="raise")
    working = working.dropna(subset=[date_col, value_col])
    working["_Month"] = working[date_col].dt.to_period("M").dt.to_timestamp()
    working = working[working["_Month"] <= cutoff]

    if working.empty:
        return pd.Series(
            dtype="float64",
            index=pd.DatetimeIndex([], freq="MS"),
            name=value_col,
        )

    start = (
        _month_start(start_month)
        if start_month is not None
        else working["_Month"].min()
    )
    if start > cutoff:
        raise ValueError("start_month must not be after cutoff_month")

    working = working[working["_Month"] >= start]
    monthly = working.groupby("_Month")[value_col].sum()
    calendar = pd.date_range(start=start, end=cutoff, freq="MS")
    result = monthly.reindex(calendar, fill_value=0)
    result.index.name = date_col
    result.name = value_col
    return result


def trailing_complete_months(series: pd.Series, months: int) -> pd.Series:
    """Return at most the latest ``months`` rows from a monthly series."""
    if months <= 0:
        raise ValueError("months must be greater than zero")
    return series.tail(months)


def count_sales_months(series: pd.Series) -> int:
    """Count months whose net sales quantity is positive."""
    numeric = pd.to_numeric(series, errors="coerce")
    return int((numeric > 0).sum())
