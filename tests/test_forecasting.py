"""Tests for pluggable X/Y forecasting backends."""

from unittest import TestCase
from unittest.mock import MagicMock, patch

import pandas as pd

from pos_pipeline.analysis.forecasting import (
    ForecastResult,
    NeuralProphetBackend,
    ProphetBackend,
    RecentMeanBackend,
    forecast_xy_base_demand,
    get_forecast_backend,
    model_data_is_eligible,
)


def _monthly_sales(months: int, nonzero: int, start: str = "2024-01-01") -> pd.DataFrame:
    rows = []
    start_ts = pd.Timestamp(start)
    for offset in range(months):
        month = start_ts + pd.DateOffset(months=offset)
        qty = 10 if offset < nonzero else 0
        rows.append({"rDate": month + pd.Timedelta(days=1), "TotalQty": qty})
    return pd.DataFrame(rows)


class ModelEligibilityTests(TestCase):
    def test_requires_24_complete_and_12_nonzero_months(self) -> None:
        monthly = pd.Series(
            [10] * 12 + [0] * 12,
            index=pd.date_range("2024-01-01", periods=24, freq="MS"),
        )
        eligible, complete, nonzero = model_data_is_eligible(monthly)
        self.assertTrue(eligible)
        self.assertEqual(complete, 24)
        self.assertEqual(nonzero, 12)

    def test_rejects_short_complete_history(self) -> None:
        monthly = pd.Series(
            [10] * 20,
            index=pd.date_range("2024-01-01", periods=20, freq="MS"),
        )
        eligible, complete, nonzero = model_data_is_eligible(monthly)
        self.assertFalse(eligible)
        self.assertEqual(complete, 20)
        self.assertEqual(nonzero, 20)

    def test_rejects_sparse_nonzero_months(self) -> None:
        monthly = pd.Series(
            [10] * 11 + [0] * 13,
            index=pd.date_range("2024-01-01", periods=24, freq="MS"),
        )
        eligible, _complete, nonzero = model_data_is_eligible(monthly)
        self.assertFalse(eligible)
        self.assertEqual(nonzero, 11)


class BackendFactoryTests(TestCase):
    def test_resolves_known_backends(self) -> None:
        self.assertIsInstance(get_forecast_backend("prophet"), ProphetBackend)
        self.assertIsInstance(
            get_forecast_backend("neuralprophet"),
            NeuralProphetBackend,
        )
        self.assertIsInstance(get_forecast_backend("recent_3m"), RecentMeanBackend)

    def test_unknown_backend_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown forecast backend"):
            get_forecast_backend("xgboost")


class RecentMeanBackendTests(TestCase):
    def test_averages_latest_months_including_zeros(self) -> None:
        monthly = pd.Series(
            [30, 0, 0],
            index=pd.date_range("2026-01-01", periods=3, freq="MS"),
        )
        self.assertEqual(RecentMeanBackend().predict_next_month(monthly), 10)


class ForecastXyBaseDemandTests(TestCase):
    def test_insufficient_history_uses_recent_mean_without_calling_model(
        self,
    ) -> None:
        sales = _monthly_sales(months=6, nonzero=6)
        backend = MagicMock()
        backend.name = "prophet"

        result = forecast_xy_base_demand(
            sales,
            cutoff_month="2024-06-01",
            backend=backend,
        )

        self.assertIsInstance(result, ForecastResult)
        self.assertEqual(result.method, "recent_3m")
        self.assertEqual(result.status, "insufficient_history")
        self.assertEqual(result.complete_months, 6)
        self.assertEqual(result.nonzero_months, 6)
        self.assertEqual(result.value, 10)
        backend.predict_next_month.assert_not_called()

    def test_eligible_history_uses_selected_backend(self) -> None:
        sales = _monthly_sales(months=24, nonzero=12)
        backend = MagicMock()
        backend.name = "prophet"
        backend.predict_next_month.return_value = 42.5

        result = forecast_xy_base_demand(
            sales,
            cutoff_month="2025-12-01",
            backend=backend,
        )

        self.assertEqual(result.method, "prophet")
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.complete_months, 24)
        self.assertEqual(result.nonzero_months, 12)
        self.assertEqual(result.value, 42.5)
        backend.predict_next_month.assert_called_once()

    def test_model_failure_propagates_without_silent_fallback(self) -> None:
        sales = _monthly_sales(months=24, nonzero=12)
        backend = MagicMock()
        backend.name = "prophet"
        backend.predict_next_month.side_effect = RuntimeError("fit failed")

        with self.assertRaisesRegex(RuntimeError, "fit failed"):
            forecast_xy_base_demand(
                sales,
                cutoff_month="2025-12-01",
                backend=backend,
            )

    def test_empty_sales_returns_zero_recent_mean(self) -> None:
        result = forecast_xy_base_demand(
            pd.DataFrame(columns=["rDate", "TotalQty"]),
            cutoff_month="2025-12-01",
            backend=MagicMock(name="prophet"),
        )
        self.assertEqual(result.value, 0)
        self.assertEqual(result.method, "recent_3m")
        self.assertEqual(result.status, "empty_series")


class LazyImportBackendTests(TestCase):
    def test_prophet_backend_reports_missing_dependency(self) -> None:
        backend = ProphetBackend()
        monthly = pd.Series(
            [1.0, 2.0],
            index=pd.date_range("2026-01-01", periods=2, freq="MS"),
        )
        with patch(
            "pos_pipeline.analysis.forecasting._load_prophet_class",
            side_effect=RuntimeError("Prophet is not installed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "Prophet is not installed"):
                backend.predict_next_month(monthly)

    def test_neuralprophet_backend_reports_missing_dependency(self) -> None:
        backend = NeuralProphetBackend()
        monthly = pd.Series(
            [1.0, 2.0],
            index=pd.date_range("2026-01-01", periods=2, freq="MS"),
        )
        with patch(
            "pos_pipeline.analysis.forecasting._load_neuralprophet_class",
            side_effect=RuntimeError("NeuralProphet is not installed"),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "NeuralProphet is not installed",
            ):
                backend.predict_next_month(monthly)

    def test_prophet_backend_uses_loaded_class(self) -> None:
        backend = ProphetBackend()
        monthly = pd.Series(
            [1.0, 2.0],
            index=pd.date_range("2026-01-01", periods=2, freq="MS"),
        )
        fake_model = MagicMock()
        fake_model.make_future_dataframe.return_value = pd.DataFrame(
            {"ds": pd.date_range("2026-01-01", periods=3, freq="MS")}
        )
        fake_model.predict.return_value = pd.DataFrame({"yhat": [1.0, 2.0, 9.0]})
        fake_cls = MagicMock(return_value=fake_model)

        with patch(
            "pos_pipeline.analysis.forecasting._load_prophet_class",
            return_value=fake_cls,
        ):
            value = backend.predict_next_month(monthly)

        self.assertEqual(value, 9.0)
        fake_model.fit.assert_called_once()


if __name__ == "__main__":
    import unittest

    unittest.main()
