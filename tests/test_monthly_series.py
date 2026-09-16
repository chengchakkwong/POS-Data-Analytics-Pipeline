"""Tests for shared complete-calendar monthly series helpers."""

from unittest import TestCase

import pandas as pd

from pos_pipeline.analysis.monthly_series import (
    build_monthly_series,
    count_sales_months,
    resolve_complete_month_cutoff,
    trailing_complete_months,
)


class ResolveCompleteMonthCutoffTests(TestCase):
    def test_mid_month_boundary_uses_previous_month(self) -> None:
        cutoff = resolve_complete_month_cutoff(
            ["2026-09-01", "2026-09-15"],
        )

        self.assertEqual(cutoff, pd.Timestamp("2026-08-01"))

    def test_month_end_boundary_includes_that_month(self) -> None:
        cutoff = resolve_complete_month_cutoff(
            ["2026-09-01", "2026-09-30"],
        )

        self.assertEqual(cutoff, pd.Timestamp("2026-09-01"))

    def test_as_of_never_looks_beyond_requested_date(self) -> None:
        cutoff = resolve_complete_month_cutoff(
            ["2026-07-01", "2026-09-30"],
            as_of="2026-08-15",
        )

        self.assertEqual(cutoff, pd.Timestamp("2026-07-01"))

    def test_empty_dates_raise_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "without dates"):
            resolve_complete_month_cutoff([])


class BuildMonthlySeriesTests(TestCase):
    def test_fills_internal_and_trailing_months_after_first_sale(self) -> None:
        sales = pd.DataFrame(
            [
                {"rDate": "2026-01-10", "TotalQty": 10},
                {"rDate": "2026-03-20", "TotalQty": 5},
            ]
        )

        result = build_monthly_series(sales, cutoff_month="2026-05-01")

        self.assertEqual(
            result.index.tolist(),
            pd.date_range("2026-01-01", "2026-05-01", freq="MS").tolist(),
        )
        self.assertEqual(result.tolist(), [10, 0, 5, 0, 0])

    def test_does_not_add_months_before_first_sale_by_default(self) -> None:
        sales = pd.DataFrame(
            [{"rDate": "2026-03-20", "TotalQty": 5}]
        )

        result = build_monthly_series(sales, cutoff_month="2026-05-01")

        self.assertEqual(
            result.index.tolist(),
            pd.date_range("2026-03-01", "2026-05-01", freq="MS").tolist(),
        )
        self.assertEqual(result.tolist(), [5, 0, 0])

    def test_explicit_start_month_can_build_a_shared_calendar(self) -> None:
        sales = pd.DataFrame(
            [{"rDate": "2026-03-20", "TotalQty": 5}]
        )

        result = build_monthly_series(
            sales,
            cutoff_month="2026-05-01",
            start_month="2026-01-01",
        )

        self.assertEqual(result.tolist(), [0, 0, 5, 0, 0])

    def test_excludes_rows_after_cutoff_month(self) -> None:
        sales = pd.DataFrame(
            [
                {"rDate": "2026-05-20", "TotalQty": 5},
                {"rDate": "2026-06-01", "TotalQty": 99},
            ]
        )

        result = build_monthly_series(sales, cutoff_month="2026-05-01")

        self.assertEqual(result.tolist(), [5])

    def test_empty_sales_returns_an_empty_named_series(self) -> None:
        sales = pd.DataFrame(columns=["rDate", "TotalQty"])

        result = build_monthly_series(sales, cutoff_month="2026-05-01")

        self.assertTrue(result.empty)
        self.assertEqual(result.name, "TotalQty")

    def test_missing_required_column_raises_clear_error(self) -> None:
        sales = pd.DataFrame([{"rDate": "2026-05-01"}])

        with self.assertRaisesRegex(ValueError, "TotalQty"):
            build_monthly_series(sales, cutoff_month="2026-05-01")


class MonthlySeriesSummaryTests(TestCase):
    def test_trailing_months_uses_calendar_rows_including_zeros(self) -> None:
        series = pd.Series(
            [10, 0, 20, 0],
            index=pd.date_range("2026-01-01", periods=4, freq="MS"),
        )

        result = trailing_complete_months(series, months=3)

        self.assertEqual(result.tolist(), [0, 20, 0])

    def test_count_sales_months_counts_only_positive_net_sales(self) -> None:
        series = pd.Series([10, 0, -2, 5, float("nan")])

        self.assertEqual(count_sales_months(series), 2)

    def test_non_positive_month_window_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            trailing_complete_months(pd.Series([1]), months=0)


if __name__ == "__main__":
    import unittest

    unittest.main()
