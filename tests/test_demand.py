"""Tests for demand-rule helpers (New / X-Y / Z / seasonal / FirstOrderQty)."""

from unittest import TestCase

import pandas as pd

from pos_pipeline.analysis.demand import (
    base_demand_for_new,
    base_demand_for_recent_mean,
    base_demand_for_z,
    calculate_category_seasonal_profile,
    new_seasonal_factor,
    parse_first_order_qty,
)


class ParseFirstOrderQtyTests(TestCase):
    def test_single_number_is_accepted(self) -> None:
        qty, status = parse_first_order_qty("24 箱規")
        self.assertEqual(qty, 24)
        self.assertEqual(status, "single_number")

    def test_year_only_note_is_accepted_under_single_number_policy(self) -> None:
        qty, status = parse_first_order_qty("2024 新款")
        self.assertEqual(qty, 2024)
        self.assertEqual(status, "single_number")

    def test_multiple_numbers_are_rejected(self) -> None:
        qty, status = parse_first_order_qty("2024 新款 24")
        self.assertIsNone(qty)
        self.assertEqual(status, "multiple_numbers")

    def test_no_number_is_missing(self) -> None:
        qty, status = parse_first_order_qty("沒有指定")
        self.assertIsNone(qty)
        self.assertEqual(status, "no_number")


class BaseDemandForNewTests(TestCase):
    def test_empty_sales_returns_zero(self) -> None:
        self.assertEqual(base_demand_for_new(pd.DataFrame(), mean_qty=8), 0)

    def test_run_rate_times_30_when_active_days_at_least_7(self) -> None:
        sales = pd.DataFrame(
            [
                {"rDate": "2026-01-01", "TotalQty": 10},
                {"rDate": "2026-01-11", "TotalQty": 20},
            ]
        )
        self.assertEqual(base_demand_for_new(sales, mean_qty=8), 90)

    def test_caps_at_total_sold_times_3_when_under_7_days(self) -> None:
        sales = pd.DataFrame([{"rDate": "2026-01-01", "TotalQty": 10}])
        self.assertEqual(base_demand_for_new(sales, mean_qty=8), 30)

    def test_as_of_extends_denominator_beyond_last_sale(self) -> None:
        sales = pd.DataFrame(
            [
                {"rDate": "2026-01-01", "TotalQty": 10},
                {"rDate": "2026-01-11", "TotalQty": 20},
            ]
        )
        # 首賣到 as_of = 20 天、總銷 30 → 日均 1.5 → ×30 = 45
        self.assertEqual(
            base_demand_for_new(sales, mean_qty=8, as_of="2026-01-21"),
            45,
        )


class BaseDemandForRecentMeanTests(TestCase):
    def test_includes_trailing_zero_months_in_average(self) -> None:
        sales = pd.DataFrame(
            [
                {"rDate": "2026-01-10", "TotalQty": 30},
                {"rDate": "2026-02-10", "TotalQty": 0},
            ]
        )
        # Jan=30, Feb=0, Mar=0 → mean of last 3 = 10
        result = base_demand_for_recent_mean(
            sales,
            cutoff_month="2026-03-01",
            months=3,
        )
        self.assertEqual(result, 10)

    def test_empty_sales_returns_zero(self) -> None:
        self.assertEqual(
            base_demand_for_recent_mean(
                pd.DataFrame(columns=["rDate", "TotalQty"]),
                cutoff_month="2026-03-01",
            ),
            0,
        )


class BaseDemandForZTests(TestCase):
    def test_empty_sales_returns_zero(self) -> None:
        self.assertEqual(base_demand_for_z(pd.DataFrame(), mean_qty=8), 0)

    def test_uses_recent_max_when_it_is_not_all_time_high(self) -> None:
        sales = pd.DataFrame(
            [
                {"rDate": "2025-01-01", "TotalQty": 100},
                {"rDate": "2026-01-01", "TotalQty": 10},
                {"rDate": "2026-02-01", "TotalQty": 20},
                {"rDate": "2026-03-01", "TotalQty": 15},
                {"rDate": "2026-04-01", "TotalQty": 12},
                {"rDate": "2026-05-01", "TotalQty": 8},
                {"rDate": "2026-06-01", "TotalQty": 5},
            ]
        )
        self.assertEqual(
            base_demand_for_z(sales, mean_qty=8, cutoff_month="2026-06-01"),
            20,
        )

    def test_drops_all_time_high_spike_to_second_max(self) -> None:
        sales = pd.DataFrame(
            [
                {"rDate": "2026-01-01", "TotalQty": 10},
                {"rDate": "2026-02-01", "TotalQty": 10},
                {"rDate": "2026-03-01", "TotalQty": 100},
            ]
        )
        self.assertEqual(
            base_demand_for_z(sales, mean_qty=8, cutoff_month="2026-03-01"),
            10,
        )

    def test_keeps_single_nonzero_month_instead_of_falling_to_zero(self) -> None:
        sales = pd.DataFrame([{"rDate": "2026-03-01", "TotalQty": 60}])
        # 首賣後補零：Jan 不存在；Mar=60, Apr=0, May=0 → 只有一個非零月
        result = base_demand_for_z(
            sales,
            mean_qty=8,
            cutoff_month="2026-05-01",
        )
        self.assertEqual(result, 60)


class CategorySeasonalProfileTests(TestCase):
    def test_pads_trailing_zero_months_before_indexing(self) -> None:
        sales = pd.DataFrame(
            [
                {"GoodsID": 1, "rDate": "2026-01-15", "TotalQty": 10},
                {"GoodsID": 1, "rDate": "2026-02-15", "TotalQty": 30},
            ]
        )
        stock = pd.DataFrame([{"GoodsID": 1, "Category": "Cups"}])

        profile = calculate_category_seasonal_profile(
            sales,
            stock,
            cutoff_month="2026-04-01",
        )

        # Jan..Apr = 10,30,0,0 → mean=10 → indices 1.0, 3.0, 0.0, 0.0
        self.assertEqual(profile.complete_month_counts["Cups"], 4)
        self.assertEqual(profile.index_map[("Cups", 1)], 1.0)
        self.assertEqual(profile.index_map[("Cups", 2)], 3.0)
        self.assertEqual(profile.index_map[("Cups", 3)], 0.0)
        self.assertEqual(profile.index_map[("Cups", 4)], 0.0)

    def test_new_seasonal_factor_requires_24_complete_months(self) -> None:
        sales = pd.DataFrame(
            [
                {"GoodsID": 1, "rDate": "2026-01-15", "TotalQty": 10},
                {"GoodsID": 1, "rDate": "2026-02-15", "TotalQty": 30},
            ]
        )
        stock = pd.DataFrame([{"GoodsID": 1, "Category": "Cups"}])
        profile = calculate_category_seasonal_profile(
            sales,
            stock,
            cutoff_month="2026-04-01",
        )

        self.assertEqual(new_seasonal_factor("Cups", 2, profile), 1.0)

    def test_new_seasonal_factor_clips_when_history_is_long_enough(self) -> None:
        rows = []
        for offset in range(24):
            month = pd.Timestamp("2024-01-01") + pd.DateOffset(months=offset)
            qty = 40 if month.month == 2 else 10
            rows.append(
                {
                    "GoodsID": 1,
                    "rDate": month + pd.Timedelta(days=1),
                    "TotalQty": qty,
                }
            )
        sales = pd.DataFrame(rows)
        stock = pd.DataFrame([{"GoodsID": 1, "Category": "Cups"}])
        profile = calculate_category_seasonal_profile(
            sales,
            stock,
            cutoff_month="2025-12-01",
        )

        self.assertGreaterEqual(profile.complete_month_counts["Cups"], 24)
        # February is strong; clipped to 1.2 instead of the raw >1.2 value
        self.assertEqual(new_seasonal_factor("Cups", 2, profile), 1.2)
        # A quiet month is clipped up to 0.8 instead of falling below it
        self.assertEqual(new_seasonal_factor("Cups", 1, profile), 0.8)


if __name__ == "__main__":
    import unittest

    unittest.main()
