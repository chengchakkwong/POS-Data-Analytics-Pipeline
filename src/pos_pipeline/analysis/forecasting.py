"""Pluggable monthly forecasting backends for X/Y demand planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from pos_pipeline.analysis.demand import base_demand_for_recent_mean
from pos_pipeline.analysis.monthly_series import (
    build_monthly_series,
    count_sales_months,
)

MIN_COMPLETE_MONTHS_FOR_MODEL = 24
MIN_NONZERO_MONTHS_FOR_MODEL = 12


@dataclass(frozen=True)
class ForecastResult:
    """One monthly demand forecast with audit metadata."""

    value: float
    method: str
    status: str
    complete_months: int
    nonzero_months: int


class ForecastBackend(Protocol):
    """Predict the next month-start demand from a complete monthly series."""

    name: str

    def predict_next_month(self, monthly: pd.Series) -> float:
        """Return a non-negative next-month prediction."""


def model_data_is_eligible(monthly: pd.Series) -> tuple[bool, int, int]:
    """Return eligibility plus complete/nonzero month counts."""
    complete_months = int(len(monthly))
    nonzero_months = count_sales_months(monthly)
    eligible = (
        complete_months >= MIN_COMPLETE_MONTHS_FOR_MODEL
        and nonzero_months >= MIN_NONZERO_MONTHS_FOR_MODEL
    )
    return eligible, complete_months, nonzero_months


def _monthly_frame(monthly: pd.Series) -> pd.DataFrame:
    frame = monthly.rename("y").reset_index()
    frame.columns = ["ds", "y"]
    frame["ds"] = pd.to_datetime(frame["ds"], errors="raise")
    frame["y"] = pd.to_numeric(frame["y"], errors="raise")
    return frame


class RecentMeanBackend:
    """Baseline backend: mean of the latest complete calendar months."""

    name = "recent_3m"

    def __init__(self, months: int = 3) -> None:
        if months <= 0:
            raise ValueError("months must be greater than zero")
        self.months = months
        if months != 3:
            self.name = f"recent_{months}m"

    def predict_next_month(self, monthly: pd.Series) -> float:
        if monthly.empty:
            return 0.0
        window = monthly.tail(self.months)
        return float(max(0.0, window.mean()))


def _load_prophet_class():
    """Lazy-load Prophet so production installs stay optional until used."""
    try:
        from prophet import Prophet
    except ImportError as exc:  # pragma: no cover - depends on local install
        raise RuntimeError(
            "Prophet is not installed; install requirements-forecast.txt"
        ) from exc
    return Prophet


def _load_neuralprophet_class():
    """Lazy-load NeuralProphet so experimental deps stay out of production."""
    try:
        from neuralprophet import NeuralProphet
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "NeuralProphet is not installed; install "
            "requirements-neuralprophet.txt for experimental runs"
        ) from exc
    return NeuralProphet


class ProphetBackend:
    """Official production backend using Prophet on month-start series."""

    name = "prophet"

    def predict_next_month(self, monthly: pd.Series) -> float:
        prophet_cls = _load_prophet_class()
        frame = _monthly_frame(monthly)
        model = prophet_cls(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            uncertainty_samples=0,
        )
        model.fit(frame)
        future = model.make_future_dataframe(periods=1, freq="MS")
        forecast = model.predict(future)
        return float(max(0.0, forecast.iloc[-1]["yhat"]))


class NeuralProphetBackend:
    """Optional experimental backend; not required for production installs."""

    name = "neuralprophet"

    def predict_next_month(self, monthly: pd.Series) -> float:
        neural_cls = _load_neuralprophet_class()
        frame = _monthly_frame(monthly)
        model = neural_cls(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            learning_rate=0.1,
            accelerator="cpu",
            epochs=30,
            batch_size=None,
            trainer_config={
                "logger": False,
                "enable_checkpointing": False,
                "enable_progress_bar": False,
            },
        )
        model.fit(frame, freq="MS", progress=None)
        future = model.make_future_dataframe(frame, periods=1)
        forecast = model.predict(future)
        value_col = "yhat1" if "yhat1" in forecast.columns else "yhat"
        return float(max(0.0, forecast.iloc[-1][value_col]))


def get_forecast_backend(name: str = "prophet") -> ForecastBackend:
    """Resolve a backend by name."""
    normalized = name.strip().lower()
    if normalized in {"prophet", "production"}:
        return ProphetBackend()
    if normalized in {"neuralprophet", "neural"}:
        return NeuralProphetBackend()
    if normalized in {"recent_3m", "recent", "baseline"}:
        return RecentMeanBackend()
    raise ValueError(f"unknown forecast backend: {name}")


def forecast_xy_base_demand(
    item_sales: pd.DataFrame,
    cutoff_month: str | pd.Timestamp,
    *,
    backend: ForecastBackend | None = None,
    backend_name: str = "prophet",
) -> ForecastResult:
    """Forecast next-month demand for X/Y SKUs.

    Eligibility gate:
    - at least 24 complete calendar months, and
    - at least 12 months with positive sales

    If the series is not eligible, use the recent-3-month mean.
    If an eligible model run fails, the exception propagates: callers must not
    silently replace the failure with a monthly average or zero.
    """
    selected = backend or get_forecast_backend(backend_name)
    monthly = build_monthly_series(item_sales, cutoff_month)
    eligible, complete_months, nonzero_months = model_data_is_eligible(monthly)

    if monthly.empty:
        return ForecastResult(
            value=0.0,
            method="recent_3m",
            status="empty_series",
            complete_months=0,
            nonzero_months=0,
        )

    if not eligible:
        value = base_demand_for_recent_mean(
            item_sales,
            cutoff_month,
            months=3,
        )
        return ForecastResult(
            value=float(value),
            method="recent_3m",
            status="insufficient_history",
            complete_months=complete_months,
            nonzero_months=nonzero_months,
        )

    value = selected.predict_next_month(monthly)
    return ForecastResult(
        value=float(max(0.0, value)),
        method=selected.name,
        status="ok",
        complete_months=complete_months,
        nonzero_months=nonzero_months,
    )
