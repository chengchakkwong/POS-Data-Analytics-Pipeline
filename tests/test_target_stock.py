"""Tests for C-class target-stock rules."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import pandas as pd

from pos_pipeline.analysis.target_stock import (
    base_demand_for_new,
    base_demand_for_z,
    calculate_category_seasonal_indices,
    plan_c_class_row,
    run_target_stock,
)


def _c_row(**overrides) -> pd.Series:
    base = {
        "ProductCode": "P001",
        "Name": "Test SKU",
        "XYZ_Class": "Z",
        "Strategy": "不建議預測 (考慮汰換)",
        "CurrStock": 5,
        "Mean_Monthly_Qty": 10,
        "FirstOrderQty": 0,
        "FirstOrderQty_Parse_Status": "no_number",
        "Note": "",
    }
    base.update(overrides)
    return pd.Series(base)


class PlanCClassRowTests(TestCase):
    def test_zero_mean_qty_returns_zero_target(self) -> None:
        result = plan_c_class_row(_c_row(Mean_Monthly_Qty=0, FirstOrderQty=24))
        self.assertEqual(result["Target_Stock"], 0)
        self.assertEqual(result["Decision_Source"], "zero_mean")

    def test_oversized_first_order_falls_back_to_mean_times_1_2(self) -> None:
        result = plan_c_class_row(_c_row(Mean_Monthly_Qty=10, FirstOrderQty=200))
        self.assertEqual(result["Target_Stock"], 12.0)
        self.assertEqual(result["Decision_Source"], "first_order_oversized")

    def test_reasonable_first_order_is_used(self) -> None:
        result = plan_c_class_row(_c_row(Mean_Monthly_Qty=10, FirstOrderQty=24))
        self.assertEqual(result["Target_Stock"], 24)
        self.assertEqual(result["Decision_Source"], "first_order_qty")

    def test_missing_first_order_uses_mean_times_1_2(self) -> None:
        result = plan_c_class_row(_c_row(Mean_Monthly_Qty=10, FirstOrderQty=None))
        self.assertEqual(result["Target_Stock"], 12.0)
        self.assertEqual(result["Decision_Source"], "mean_times_1_2")

    def test_parses_single_number_note_when_first_order_missing(self) -> None:
        result = plan_c_class_row(
            _c_row(
                FirstOrderQty=None,
                Note="24 箱規",
            ).drop(labels=["FirstOrderQty_Parse_Status"])
        )
        self.assertEqual(result["FirstOrderQty"], 24)
        self.assertEqual(result["FirstOrderQty_Parse_Status"], "single_number")
        self.assertEqual(result["Target_Stock"], 24)

    def test_rejects_multiple_number_note(self) -> None:
        result = plan_c_class_row(
            _c_row(
                FirstOrderQty=None,
                Note="2024 新款 24",
            ).drop(labels=["FirstOrderQty_Parse_Status"])
        )
        self.assertEqual(result["FirstOrderQty"], 0)
        self.assertEqual(result["FirstOrderQty_Parse_Status"], "multiple_numbers")
        self.assertEqual(result["Target_Stock"], 12.0)


class RunTargetStockTests(TestCase):
    def test_keeps_only_c_class_and_parses_single_number_note(self) -> None:
        labels = pd.DataFrame(
            [
                {
                    "GoodsID": 1,
                    "ABC_Class": "C",
                    "XYZ_Class": "Z",
                    "CV": 1.5,
                    "Mean_Monthly_Qty": 10,
                    "Strategy": "不建議預測 (考慮汰換)",
                },
                {
                    "GoodsID": 2,
                    "ABC_Class": "A",
                    "XYZ_Class": "X",
                    "CV": 0.2,
                    "Mean_Monthly_Qty": 50,
                    "Strategy": "自動補貨 (高頻穩定)",
                },
            ]
        )
        stock = pd.DataFrame(
            [
                {
                    "GoodsID": 1,
                    "ProductCode": "C001",
                    "Name": "C SKU",
                    "CurrStock": 3,
                    "Note": "24 箱規",
                },
                {
                    "GoodsID": 2,
                    "ProductCode": "A001",
                    "Name": "A SKU",
                    "CurrStock": 20,
                    "Note": "",
                },
            ]
        )

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            labels_csv = tmp_path / "abc_xyz.csv"
            stock_csv = tmp_path / "stock.csv"
            output_csv = tmp_path / "target_stock.csv"
            labels.to_csv(labels_csv, index=False)
            stock.to_csv(stock_csv, index=False)

            result = run_target_stock(
                labels_csv=labels_csv,
                stock_csv=stock_csv,
                output_csv=output_csv,
            )

            self.assertTrue(output_csv.exists())
            self.assertEqual(len(result), 1)
            self.assertEqual(result.iloc[0]["ProductCode"], "C001")
            self.assertEqual(result.iloc[0]["ABC_XYZ"], "CZ")
            self.assertEqual(result.iloc[0]["Target_Stock"], 24)
            self.assertEqual(result.iloc[0]["FirstOrderQty"], 24)
            self.assertEqual(
                result.iloc[0]["FirstOrderQty_Parse_Status"],
                "single_number",
            )
            self.assertEqual(result.iloc[0]["Decision_Source"], "first_order_qty")
            self.assertEqual(result.iloc[0]["Note"], "24 箱規")

    def test_multiple_number_note_does_not_become_first_order(self) -> None:
        labels = pd.DataFrame(
            [
                {
                    "GoodsID": 1,
                    "ABC_Class": "C",
                    "XYZ_Class": "Z",
                    "CV": 1.5,
                    "Mean_Monthly_Qty": 10,
                    "Strategy": "不建議預測 (考慮汰換)",
                }
            ]
        )
        stock = pd.DataFrame(
            [
                {
                    "GoodsID": 1,
                    "ProductCode": "C001",
                    "Name": "C SKU",
                    "CurrStock": 3,
                    "Note": "2024 新款 24",
                }
            ]
        )

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            labels_csv = tmp_path / "abc_xyz.csv"
            stock_csv = tmp_path / "stock.csv"
            output_csv = tmp_path / "target_stock.csv"
            labels.to_csv(labels_csv, index=False)
            stock.to_csv(stock_csv, index=False)

            result = run_target_stock(
                labels_csv=labels_csv,
                stock_csv=stock_csv,
                output_csv=output_csv,
            )

            self.assertEqual(result.iloc[0]["FirstOrderQty"], 0)
            self.assertEqual(
                result.iloc[0]["FirstOrderQty_Parse_Status"],
                "multiple_numbers",
            )
            self.assertEqual(result.iloc[0]["Target_Stock"], 12.0)


class CategorySeasonalIndexTests(TestCase):
    def test_month_index_uses_complete_calendar_with_zeros(self) -> None:
        sales = pd.DataFrame(
            [
                {"GoodsID": 1, "rDate": "2026-01-15", "TotalQty": 10},
                {"GoodsID": 1, "rDate": "2026-02-15", "TotalQty": 30},
            ]
        )
        stock = pd.DataFrame([{"GoodsID": 1, "Category": "Cups"}])

        index_map = calculate_category_seasonal_indices(
            sales,
            stock,
            cutoff_month="2026-02-01",
        )

        # 月均 = (10+30)/2 = 20 → 1 月 10/20=0.5，2 月 30/20=1.5
        self.assertEqual(index_map[("Cups", 1)], 0.5)
        self.assertEqual(index_map[("Cups", 2)], 1.5)


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


if __name__ == "__main__":
    import unittest

    unittest.main()
