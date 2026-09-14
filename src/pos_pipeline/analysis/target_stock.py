"""Target-stock planning from ABC/XYZ labels and local caches."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pos_pipeline.analysis.demand import (
    CategorySeasonalProfile,
    base_demand_for_new,
    base_demand_for_recent_mean,
    base_demand_for_z,
    calculate_category_seasonal_indices,
    calculate_category_seasonal_profile,
    new_seasonal_factor,
    parse_first_order_qty,
)
from pos_pipeline.analysis.forecasting import (
    ForecastBackend,
    ForecastResult,
    forecast_xy_base_demand,
    get_forecast_backend,
)
from pos_pipeline.analysis.monthly_series import (
    build_monthly_series,
    count_sales_months,
    resolve_complete_month_cutoff,
)
from pos_pipeline.config import (
    ABC_XYZ_CSV,
    SALES_PARQUET_DIR,
    STOCK_MASTER_CSV,
    TARGET_STOCK_CSV,
    TARGET_STOCK_TRACE_CSV,
)

# Re-export demand helpers so existing imports keep working during the refactor.
__all__ = [
    "JoinAudit",
    "base_demand_for_new",
    "base_demand_for_recent_mean",
    "base_demand_for_z",
    "calculate_category_seasonal_indices",
    "calculate_category_seasonal_profile",
    "new_seasonal_factor",
    "parse_first_order_qty",
    "plan_c_class_row",
    "plan_focus_sku_row",
    "run_target_stock",
    "write_csv_atomic",
]

PLAN_COLUMNS = [
    "ProductCode",
    "Name",
    "ABC_XYZ",
    "Strategy",
    "CurrStock",
    "FirstOrderQty",
    "Note",
    "Base_Demand",
    "Final_Demand",
    "Target_Stock",
]

TRACE_COLUMNS = [
    "GoodsID",
    "ProductCode",
    "ABC_Class",
    "XYZ_Class",
    "ABC_XYZ",
    "Calendar_Start",
    "Calendar_End",
    "Complete_Months",
    "Nonzero_Months",
    "Forecast_Method",
    "Forecast_Status",
    "Decision_Source",
    "CV",
    "CV_Status",
    "Mean_Monthly_Qty",
    "Base_Demand",
    "Seasonal_Factor",
    "Final_Demand",
    "Safety_Ratio",
    "Target_Before_Cap",
    "Target_Cap",
    "Cap_Applied",
    "Target_Stock",
    "FirstOrderQty",
    "FirstOrderQty_Parse_Status",
]


@dataclass(frozen=True)
class JoinAudit:
    """Counts for SKUs dropped or rejected by the labels/stock join."""

    labels_without_stock: int
    stock_without_labels: int
    invalid_goods_id: int


def write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    """Write CSV via a temp file, then replace the destination."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        df.to_csv(tmp_path, index=False, encoding="utf-8-sig")
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def normalize_goods_id_series(series: pd.Series) -> pd.Series:
    """Normalize GoodsID values for reliable joins."""

    def _one(value: object) -> object:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return pd.NA
        if isinstance(value, bool):
            return pd.NA
        if isinstance(value, int):
            return int(value)
        if isinstance(value, float):
            if value.is_integer():
                return int(value)
            return pd.NA
        text = str(value).strip()
        if not text or text.lower() == "nan":
            return pd.NA
        try:
            number = float(text)
        except ValueError:
            return text
        if number.is_integer():
            return int(number)
        return pd.NA

    return series.map(_one)


def _mean_qty(row: pd.Series) -> float:
    value = row.get("Mean_Monthly_Qty", 0)
    if pd.isna(value):
        return 0.0
    return float(value)


def _note_text(row: pd.Series) -> str:
    note = row.get("Note", "")
    if pd.isna(note):
        return ""
    return str(note)


def _resolve_first_order(row: pd.Series) -> tuple[float | None, str]:
    first_order_qty = row.get("FirstOrderQty")
    parse_status = row.get("FirstOrderQty_Parse_Status")
    note = _note_text(row)

    if pd.isna(first_order_qty) and "FirstOrderQty_Parse_Status" not in row.index:
        return parse_first_order_qty(note)
    if pd.isna(first_order_qty):
        return None, str(parse_status or "no_number")
    return float(first_order_qty), str(parse_status or "single_number")


def _empty_calendar_fields() -> dict:
    return {
        "Calendar_Start": pd.NaT,
        "Calendar_End": pd.NaT,
        "Complete_Months": 0,
        "Nonzero_Months": 0,
    }


def _calendar_fields(
    item_sales: pd.DataFrame,
    cutoff_month: pd.Timestamp,
) -> dict:
    if item_sales is None or item_sales.empty:
        return _empty_calendar_fields()
    monthly = build_monthly_series(item_sales, cutoff_month)
    if monthly.empty:
        return _empty_calendar_fields()
    return {
        "Calendar_Start": monthly.index.min(),
        "Calendar_End": monthly.index.max(),
        "Complete_Months": int(len(monthly)),
        "Nonzero_Months": count_sales_months(monthly),
    }


def _safety_ratio(abc: str, cv: float | None, *, require_cv: bool) -> tuple[float, str]:
    if cv is None or pd.isna(cv):
        if require_cv:
            raise ValueError("missing CV for safety-stock calculation")
        return 0.0, "missing"
    limit = 0.5 if abc == "A" else 0.3
    return float(min(limit, float(cv) * 0.5)), "ok"


def _apply_target_cap(
    *,
    abc: str,
    mean_qty: float,
    target_before_cap: float,
) -> tuple[float | None, bool, float]:
    """Return target_cap, cap_applied, target."""
    if abc == "New" or mean_qty <= 0:
        return None, False, target_before_cap
    target_cap = max(mean_qty * 4, 2)
    if target_before_cap > target_cap:
        return float(target_cap), True, float(target_cap)
    return float(target_cap), False, target_before_cap


def plan_c_class_row(row: pd.Series) -> dict:
    """C 類長尾：用月均量與 FirstOrderQty 算目標庫存。"""
    mean_qty = _mean_qty(row)
    curr_stock = row.get("CurrStock", 0)
    first_order_qty, parse_status = _resolve_first_order(row)
    note = _note_text(row)

    decision_source = "mean_times_1_2"
    if mean_qty <= 0:
        target_stock = 0
        decision_source = "zero_mean"
    elif first_order_qty is not None and first_order_qty > 0:
        if first_order_qty > (mean_qty * 12):
            target_stock = mean_qty * 1.2
            decision_source = "first_order_oversized"
        else:
            target_stock = first_order_qty
            decision_source = "first_order_qty"
    else:
        target_stock = mean_qty * 1.2
        decision_source = "mean_times_1_2"

    xyz = row.get("XYZ_Class", "Z")
    plan = {
        "GoodsID": row.get("GoodsID"),
        "ProductCode": row.get("ProductCode", "Unknown"),
        "Name": row.get("Name", "Unknown"),
        "ABC_Class": "C",
        "XYZ_Class": xyz,
        "ABC_XYZ": f"C{xyz}",
        "Strategy": row.get("Strategy", "Unknown"),
        "CurrStock": curr_stock,
        "FirstOrderQty": first_order_qty if first_order_qty is not None else 0,
        "FirstOrderQty_Parse_Status": parse_status,
        "Decision_Source": decision_source,
        "Note": note,
        "Base_Demand": round(mean_qty, 2),
        "Final_Demand": round(mean_qty, 2),
        "Target_Stock": round(target_stock, 2),
        "Forecast_Method": "c_class_rule",
        "Forecast_Status": "ok",
        "CV": row.get("CV"),
        "CV_Status": "ok" if pd.notna(row.get("CV")) else "missing",
        "Mean_Monthly_Qty": mean_qty,
        "Seasonal_Factor": 1.0,
        "Safety_Ratio": 0.0,
        "Target_Before_Cap": round(target_stock, 2),
        "Target_Cap": None,
        "Cap_Applied": False,
        **_empty_calendar_fields(),
    }
    return plan


def plan_focus_sku_row(
    row: pd.Series,
    item_sales: pd.DataFrame,
    *,
    cutoff_month: pd.Timestamp,
    next_month: int,
    seasonal_profile: CategorySeasonalProfile,
    as_of: str | pd.Timestamp | None = None,
    backend: ForecastBackend | None = None,
    backend_name: str = "prophet",
) -> dict:
    """Plan one A/B/New SKU into plan+trace fields."""
    abc = str(row["ABC_Class"])
    xyz = str(row["XYZ_Class"])
    mean_qty = _mean_qty(row)
    cv = row.get("CV")
    category = row.get("Category", "Unknown")
    goods_id = row.get("GoodsID")
    calendar = _calendar_fields(item_sales, cutoff_month)

    if abc == "New":
        base_pred = base_demand_for_new(item_sales, mean_qty, as_of=as_of)
        forecast_method = "new_run_rate"
        forecast_status = "empty_series" if base_pred == 0 and (
            item_sales is None or item_sales.empty
        ) else "ok"
        seasonal_factor = new_seasonal_factor(category, next_month, seasonal_profile)
        safety_ratio, cv_status = _safety_ratio(abc, cv, require_cv=False)
        decision_source = "new_run_rate"
    elif xyz in {"X", "Y"}:
        if pd.isna(cv):
            raise ValueError(
                f"GoodsID {goods_id}: missing CV for {abc}{xyz} planning"
            )
        forecast: ForecastResult = forecast_xy_base_demand(
            item_sales,
            cutoff_month,
            backend=backend,
            backend_name=backend_name,
        )
        base_pred = forecast.value
        forecast_method = forecast.method
        forecast_status = forecast.status
        seasonal_factor = 1.0
        safety_ratio, cv_status = _safety_ratio(abc, cv, require_cv=True)
        decision_source = forecast_method
        calendar = {
            "Calendar_Start": calendar["Calendar_Start"],
            "Calendar_End": calendar["Calendar_End"],
            "Complete_Months": forecast.complete_months,
            "Nonzero_Months": forecast.nonzero_months,
        }
    else:
        if pd.isna(cv):
            raise ValueError(
                f"GoodsID {goods_id}: missing CV for {abc}{xyz} planning"
            )
        base_pred = base_demand_for_z(
            item_sales,
            mean_qty,
            cutoff_month=cutoff_month,
            as_of=as_of,
        )
        forecast_method = "z_recent_max"
        forecast_status = (
            "empty_series"
            if base_pred == 0 and (item_sales is None or item_sales.empty)
            else "ok"
        )
        seasonal_factor = 1.0
        safety_ratio, cv_status = _safety_ratio(abc, cv, require_cv=True)
        decision_source = "z_recent_max"

    if abc != "New" and mean_qty > 0:
        base_pred = min(base_pred, mean_qty * 3)

    final_demand = base_pred * seasonal_factor
    target_before = final_demand * (1.0 + safety_ratio)
    target_cap, cap_applied, target = _apply_target_cap(
        abc=abc,
        mean_qty=mean_qty,
        target_before_cap=target_before,
    )

    return {
        "GoodsID": goods_id,
        "ProductCode": row.get("ProductCode", "Unknown"),
        "Name": row.get("Name", "Unknown"),
        "ABC_Class": abc,
        "XYZ_Class": xyz,
        "ABC_XYZ": f"{abc}{xyz}",
        "Strategy": row.get("Strategy", "Unknown"),
        "CurrStock": row.get("CurrStock", 0),
        "FirstOrderQty": 0,
        "FirstOrderQty_Parse_Status": "not_applicable",
        "Decision_Source": decision_source,
        "Note": _note_text(row),
        "Base_Demand": round(base_pred, 2),
        "Final_Demand": round(final_demand, 2),
        "Target_Stock": round(target, 2),
        "Forecast_Method": forecast_method,
        "Forecast_Status": forecast_status,
        "CV": cv,
        "CV_Status": cv_status,
        "Mean_Monthly_Qty": mean_qty,
        "Seasonal_Factor": round(seasonal_factor, 4),
        "Safety_Ratio": round(safety_ratio, 4),
        "Target_Before_Cap": round(target_before, 2),
        "Target_Cap": None if target_cap is None else round(float(target_cap), 2),
        "Cap_Applied": cap_applied,
        **calendar,
    }


def _prepare_stock(df_stock: pd.DataFrame) -> pd.DataFrame:
    stock = df_stock.copy()
    if "FirstOrderQty" not in stock.columns:
        if "Note" in stock.columns:
            parsed = stock["Note"].map(parse_first_order_qty)
            stock["FirstOrderQty"] = [qty for qty, _status in parsed]
            stock["FirstOrderQty_Parse_Status"] = [
                status for _qty, status in parsed
            ]
        else:
            stock["FirstOrderQty"] = None
            stock["FirstOrderQty_Parse_Status"] = "no_number"
    elif "FirstOrderQty_Parse_Status" not in stock.columns:
        stock["FirstOrderQty_Parse_Status"] = stock["FirstOrderQty"].map(
            lambda value: "single_number"
            if pd.notna(value) and float(value) > 0
            else "no_number"
        )
    return stock


def _join_labels_and_stock(
    df_labels: pd.DataFrame,
    df_stock: pd.DataFrame,
) -> tuple[pd.DataFrame, JoinAudit]:
    labels = df_labels.copy()
    stock = df_stock.copy()
    labels["GoodsID"] = normalize_goods_id_series(labels["GoodsID"])
    stock["GoodsID"] = normalize_goods_id_series(stock["GoodsID"])

    invalid_goods_id = int(
        labels["GoodsID"].isna().sum() + stock["GoodsID"].isna().sum()
    )
    labels = labels.dropna(subset=["GoodsID"])
    stock = stock.dropna(subset=["GoodsID"])

    label_ids = set(labels["GoodsID"])
    stock_ids = set(stock["GoodsID"])
    audit = JoinAudit(
        labels_without_stock=len(label_ids - stock_ids),
        stock_without_labels=len(stock_ids - label_ids),
        invalid_goods_id=invalid_goods_id,
    )

    analysis_df = pd.merge(
        labels[
            ["GoodsID", "ABC_Class", "XYZ_Class", "CV", "Mean_Monthly_Qty", "Strategy"]
        ],
        stock,
        on="GoodsID",
        how="inner",
    )
    analysis_df["FirstOrderQty"] = pd.to_numeric(
        analysis_df["FirstOrderQty"], errors="coerce"
    )
    return analysis_df, audit


def _to_plan_frame(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=PLAN_COLUMNS)
    frame = pd.DataFrame(rows)
    for column in PLAN_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[PLAN_COLUMNS].sort_values(
        by="Target_Stock",
        ascending=False,
    ).reset_index(drop=True)


def _to_trace_frame(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=TRACE_COLUMNS)
    frame = pd.DataFrame(rows)
    for column in TRACE_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[TRACE_COLUMNS]


def run_target_stock(
    labels_csv: Path | None = None,
    stock_csv: Path | None = None,
    output_csv: Path | None = None,
    trace_csv: Path | None = None,
    sales_df: pd.DataFrame | None = None,
    sales_dir: Path | None = None,
    as_of: str | pd.Timestamp | None = None,
    backend: ForecastBackend | None = None,
    backend_name: str = "prophet",
) -> pd.DataFrame:
    """Build the target-stock plan for A/B/New/C and write plan + trace CSVs.

    All SKUs are planned in memory first. CSV outputs are replaced only after
    every row succeeds, so a model/data failure leaves the previous successful
    files untouched.
    """
    labels_csv = labels_csv or ABC_XYZ_CSV
    stock_csv = stock_csv or STOCK_MASTER_CSV
    output_csv = output_csv or TARGET_STOCK_CSV
    trace_csv = trace_csv or TARGET_STOCK_TRACE_CSV

    df_labels = pd.read_csv(labels_csv)
    df_stock = _prepare_stock(pd.read_csv(stock_csv))
    analysis_df, audit = _join_labels_and_stock(df_labels, df_stock)

    print(
        "[target-stock] join audit: "
        f"labels_without_stock={audit.labels_without_stock}, "
        f"stock_without_labels={audit.stock_without_labels}, "
        f"invalid_goods_id={audit.invalid_goods_id}"
    )

    target_skus = analysis_df[
        analysis_df["ABC_Class"].isin(["A", "B", "New", "C"])
    ].copy()
    focus_skus = target_skus[target_skus["ABC_Class"].isin(["A", "B", "New"])]
    c_class_skus = target_skus[target_skus["ABC_Class"] == "C"]

    if sales_df is None:
        if focus_skus.empty:
            sales_df = pd.DataFrame(columns=["GoodsID", "rDate", "TotalQty"])
        else:
            from parquet_utils import load_sales_parquet

            sales_df = load_sales_parquet(sales_dir or SALES_PARQUET_DIR)

    if sales_df is None:
        raise ValueError("sales_df is required when A/B/New SKUs are present")

    sales_df = sales_df.copy()
    if not sales_df.empty:
        required_sales = {"GoodsID", "rDate", "TotalQty"}
        missing = required_sales.difference(sales_df.columns)
        if missing:
            missing_names = ", ".join(sorted(missing))
            raise ValueError(f"sales data is missing required columns: {missing_names}")
        sales_df["GoodsID"] = normalize_goods_id_series(sales_df["GoodsID"])
        sales_df["rDate"] = pd.to_datetime(sales_df["rDate"], errors="raise")
        if as_of is not None:
            sales_df = sales_df[sales_df["rDate"] <= pd.Timestamp(as_of)].copy()
        elif not sales_df.empty:
            # New run-rate denominator uses the shared last sync day, not each
            # SKU's own last sale date (which overstates dormant new items).
            as_of = sales_df["rDate"].max()

    if focus_skus.empty:
        cutoff_month = pd.Timestamp("1970-01-01")
        next_month = 1
        seasonal_profile = CategorySeasonalProfile({}, {})
        sales_by_sku = {}
    else:
        if sales_df.empty:
            raise ValueError("sales data is empty; cannot plan A/B/New SKUs")
        cutoff_month = resolve_complete_month_cutoff(sales_df["rDate"], as_of=as_of)
        next_month = int((cutoff_month + pd.DateOffset(months=1)).month)
        seasonal_profile = calculate_category_seasonal_profile(
            sales_df,
            analysis_df if "Category" in analysis_df.columns else df_stock,
            cutoff_month=cutoff_month,
        )
        sales_by_sku = {
            goods_id: group
            for goods_id, group in sales_df.groupby("GoodsID", sort=False)
        }

    selected_backend = backend
    needs_model = bool(
        (
            focus_skus["ABC_Class"].isin(["A", "B"])
            & focus_skus["XYZ_Class"].isin(["X", "Y"])
        ).any()
    )
    if selected_backend is None and needs_model:
        # Resolve once so missing Prophet installs fail before partial writes.
        selected_backend = get_forecast_backend(backend_name)

    planned_rows: list[dict] = []
    for _, row in focus_skus.iterrows():
        item_sales = sales_by_sku.get(
            row["GoodsID"],
            pd.DataFrame(columns=["rDate", "TotalQty"]),
        )
        planned_rows.append(
            plan_focus_sku_row(
                row,
                item_sales,
                cutoff_month=cutoff_month,
                next_month=next_month,
                seasonal_profile=seasonal_profile,
                as_of=as_of,
                backend=selected_backend,
                backend_name=backend_name,
            )
        )

    for _, row in c_class_skus.iterrows():
        planned_rows.append(plan_c_class_row(row))

    plan_df = _to_plan_frame(planned_rows)
    trace_df = _to_trace_frame(planned_rows)

    write_csv_atomic(plan_df, output_csv)
    write_csv_atomic(trace_df, trace_csv)
    return plan_df
