"""Tests for C-class and A/B/New target-stock planning."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import MagicMock

import pandas as pd

from pos_pipeline.analysis.demand import CategorySeasonalProfile
from pos_pipeline.analysis.forecasting import RecentMeanBackend
from pos_pipeline.analysis.target_stock import (
    base_demand_for_new,
    base_demand_for_z,
    calculate_category_seasonal_indices,
    plan_c_class_row,
    plan_focus_sku_row,
    run_target_stock,
    write_csv_atomic,
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
        "CV": 1.5,
        "GoodsID": 1,
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


class PlanFocusSkuRowTests(TestCase):
    def test_z_class_applies_safety_and_target_cap(self) -> None:
        row = pd.Series(
            {
                "GoodsID": 1,
                "ProductCode": "A001",
                "Name": "A Z SKU",
                "ABC_Class": "A",
                "XYZ_Class": "Z",
                "Strategy": "高安全庫存",
                "CurrStock": 1,
                "Category": "Cups",
                "CV": 2.0,
                "Mean_Monthly_Qty": 10,
                "Note": "",
            }
        )
        sales = pd.DataFrame(
            [
                {"rDate": "2025-01-01", "TotalQty": 100},
                {"rDate": "2026-01-01", "TotalQty": 10},
                {"rDate": "2026-02-01", "TotalQty": 40},
                {"rDate": "2026-03-01", "TotalQty": 20},
            ]
        )
        result = plan_focus_sku_row(
            row,
            sales,
            cutoff_month=pd.Timestamp("2026-03-01"),
            next_month=4,
            seasonal_profile=CategorySeasonalProfile({}, {}),
        )
        # recent max=40 (not all-time high 100), then base capped by mean*3=30
        # safety=min(0.5, 1.0)=0.5 → before cap=45 → target cap=40
        self.assertEqual(result["Base_Demand"], 30)
        self.assertEqual(result["Seasonal_Factor"], 1.0)
        self.assertEqual(result["Safety_Ratio"], 0.5)
        self.assertEqual(result["Target_Before_Cap"], 45)
        self.assertEqual(result["Target_Cap"], 40)
        self.assertTrue(result["Cap_Applied"])
        self.assertEqual(result["Target_Stock"], 40)
        self.assertEqual(result["Forecast_Method"], "z_recent_max")

    def test_missing_cv_for_ab_raises(self) -> None:
        row = pd.Series(
            {
                "GoodsID": 1,
                "ProductCode": "A001",
                "Name": "A X SKU",
                "ABC_Class": "A",
                "XYZ_Class": "X",
                "Strategy": "自動補貨",
                "CurrStock": 1,
                "Category": "Cups",
                "CV": float("nan"),
                "Mean_Monthly_Qty": 10,
                "Note": "",
            }
        )
        with self.assertRaisesRegex(ValueError, "missing CV"):
            plan_focus_sku_row(
                row,
                pd.DataFrame([{"rDate": "2026-01-01", "TotalQty": 10}]),
                cutoff_month=pd.Timestamp("2026-01-01"),
                next_month=2,
                seasonal_profile=CategorySeasonalProfile({}, {}),
                backend=RecentMeanBackend(),
            )

    def test_xy_uses_injected_backend(self) -> None:
        row = pd.Series(
            {
                "GoodsID": 1,
                "ProductCode": "B001",
                "Name": "B Y SKU",
                "ABC_Class": "B",
                "XYZ_Class": "Y",
                "Strategy": "季節性補貨",
                "CurrStock": 2,
                "Category": "Cups",
                "CV": 0.4,
                "Mean_Monthly_Qty": 20,
                "Note": "",
            }
        )
        sales = pd.DataFrame(
            [
                {
                    "rDate": (pd.Timestamp("2024-01-01") + pd.DateOffset(months=i)),
                    "TotalQty": 10,
                }
                for i in range(24)
            ]
        )
        backend = MagicMock()
        backend.name = "prophet"
        backend.predict_next_month.return_value = 15

        result = plan_focus_sku_row(
            row,
            sales,
            cutoff_month=pd.Timestamp("2025-12-01"),
            next_month=1,
            seasonal_profile=CategorySeasonalProfile({}, {}),
            backend=backend,
        )
        self.assertEqual(result["Forecast_Method"], "prophet")
        self.assertEqual(result["Base_Demand"], 15)
        self.assertEqual(result["Safety_Ratio"], 0.2)
        self.assertEqual(result["Target_Stock"], 18)


class RunTargetStockTests(TestCase):
    def test_keeps_c_and_focus_classes_and_writes_trace(self) -> None:
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
                    "XYZ_Class": "Z",
                    "CV": 0.8,
                    "Mean_Monthly_Qty": 10,
                    "Strategy": "高安全庫存",
                },
                {
                    "GoodsID": 3,
                    "ABC_Class": "Excluded",
                    "XYZ_Class": "Z",
                    "CV": 0.1,
                    "Mean_Monthly_Qty": 1,
                    "Strategy": "排除對象",
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
                    "Category": "Cups",
                },
                {
                    "GoodsID": 2,
                    "ProductCode": "A001",
                    "Name": "A SKU",
                    "CurrStock": 20,
                    "Note": "",
                    "Category": "Cups",
                },
                {
                    "GoodsID": 3,
                    "ProductCode": "E001",
                    "Name": "Excluded",
                    "CurrStock": 0,
                    "Note": "",
                    "Category": "Cups",
                },
            ]
        )
        sales = pd.DataFrame(
            [
                {"GoodsID": 2, "rDate": "2026-01-01", "TotalQty": 10},
                {"GoodsID": 2, "rDate": "2026-02-01", "TotalQty": 12},
                {"GoodsID": 2, "rDate": "2026-03-01", "TotalQty": 8},
            ]
        )

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            labels_csv = tmp_path / "abc_xyz.csv"
            stock_csv = tmp_path / "stock.csv"
            output_csv = tmp_path / "target_stock.csv"
            trace_csv = tmp_path / "target_stock_trace.csv"
            labels.to_csv(labels_csv, index=False)
            stock.to_csv(stock_csv, index=False)

            result = run_target_stock(
                labels_csv=labels_csv,
                stock_csv=stock_csv,
                output_csv=output_csv,
                trace_csv=trace_csv,
                sales_df=sales,
                backend=RecentMeanBackend(),
            )

            self.assertTrue(output_csv.exists())
            self.assertTrue(trace_csv.exists())
            self.assertEqual(len(result), 2)
            self.assertNotIn("E001", set(result["ProductCode"]))
            self.assertEqual(
                set(result["ProductCode"]),
                {"C001", "A001"},
            )
            trace = pd.read_csv(trace_csv)
            self.assertIn("Forecast_Method", trace.columns)
            self.assertIn("Cap_Applied", trace.columns)

    def test_model_failure_does_not_overwrite_previous_outputs(self) -> None:
        labels = pd.DataFrame(
            [
                {
                    "GoodsID": 1,
                    "ABC_Class": "A",
                    "XYZ_Class": "X",
                    "CV": 0.2,
                    "Mean_Monthly_Qty": 10,
                    "Strategy": "自動補貨",
                }
            ]
        )
        stock = pd.DataFrame(
            [
                {
                    "GoodsID": 1,
                    "ProductCode": "A001",
                    "Name": "A SKU",
                    "CurrStock": 1,
                    "Note": "",
                    "Category": "Cups",
                }
            ]
        )
        sales = pd.DataFrame(
            [
                {
                    "GoodsID": 1,
                    "rDate": (
                        pd.Timestamp("2024-01-01") + pd.DateOffset(months=i)
                    )
                    + pd.offsets.MonthEnd(0),
                    "TotalQty": 10,
                }
                for i in range(24)
            ]
        )

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            labels_csv = tmp_path / "abc_xyz.csv"
            stock_csv = tmp_path / "stock.csv"
            output_csv = tmp_path / "target_stock.csv"
            trace_csv = tmp_path / "target_stock_trace.csv"
            labels.to_csv(labels_csv, index=False)
            stock.to_csv(stock_csv, index=False)
            output_csv.write_text("old_plan\n", encoding="utf-8")
            trace_csv.write_text("old_trace\n", encoding="utf-8")

            backend = MagicMock()
            backend.name = "prophet"
            backend.predict_next_month.side_effect = RuntimeError("fit failed")

            with self.assertRaisesRegex(RuntimeError, "fit failed"):
                run_target_stock(
                    labels_csv=labels_csv,
                    stock_csv=stock_csv,
                    output_csv=output_csv,
                    trace_csv=trace_csv,
                    sales_df=sales,
                    backend=backend,
                )

            self.assertEqual(output_csv.read_text(encoding="utf-8"), "old_plan\n")
            self.assertEqual(trace_csv.read_text(encoding="utf-8"), "old_trace\n")

    def test_new_run_rate_uses_global_sales_max_as_as_of(self) -> None:
        """Without as_of, New active days end at the shared sales sync date."""
        labels = pd.DataFrame(
            [
                {
                    "GoodsID": 1,
                    "ABC_Class": "New",
                    "XYZ_Class": "New",
                    "CV": float("nan"),
                    "Mean_Monthly_Qty": 0,
                    "Strategy": "新品觀察 (手動控貨)",
                }
            ]
        )
        stock = pd.DataFrame(
            [
                {
                    "GoodsID": 1,
                    "ProductCode": "N001",
                    "Name": "New SKU",
                    "CurrStock": 1,
                    "Note": "",
                    "Category": "Cups",
                }
            ]
        )
        # New SKU last sold on Jan 11; another SKU keeps the cache alive until Jan 21.
        # Active days = 20 → (30/20)*30 = 45, not (30/10)*30 = 90.
        sales = pd.DataFrame(
            [
                {"GoodsID": 1, "rDate": "2026-01-01", "TotalQty": 10},
                {"GoodsID": 1, "rDate": "2026-01-11", "TotalQty": 20},
                {"GoodsID": 99, "rDate": "2026-01-21", "TotalQty": 1},
            ]
        )

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            labels_csv = tmp_path / "abc_xyz.csv"
            stock_csv = tmp_path / "stock.csv"
            output_csv = tmp_path / "target_stock.csv"
            trace_csv = tmp_path / "target_stock_trace.csv"
            labels.to_csv(labels_csv, index=False)
            stock.to_csv(stock_csv, index=False)

            result = run_target_stock(
                labels_csv=labels_csv,
                stock_csv=stock_csv,
                output_csv=output_csv,
                trace_csv=trace_csv,
                sales_df=sales,
            )

            self.assertEqual(len(result), 1)
            self.assertEqual(result.iloc[0]["Base_Demand"], 45)

    def test_normalizes_goods_id_float_strings_for_join(self) -> None:
        labels = pd.DataFrame(
            [
                {
                    "GoodsID": "1.0",
                    "ABC_Class": "C",
                    "XYZ_Class": "Z",
                    "CV": 1.5,
                    "Mean_Monthly_Qty": 10,
                    "Strategy": "不建議預測",
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
                    "Note": "",
                    "Category": "Cups",
                }
            ]
        )
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            labels_csv = tmp_path / "abc_xyz.csv"
            stock_csv = tmp_path / "stock.csv"
            output_csv = tmp_path / "target_stock.csv"
            trace_csv = tmp_path / "target_stock_trace.csv"
            labels.to_csv(labels_csv, index=False)
            stock.to_csv(stock_csv, index=False)

            result = run_target_stock(
                labels_csv=labels_csv,
                stock_csv=stock_csv,
                output_csv=output_csv,
                trace_csv=trace_csv,
                sales_df=pd.DataFrame(columns=["GoodsID", "rDate", "TotalQty"]),
            )
            self.assertEqual(len(result), 1)
            self.assertEqual(result.iloc[0]["ProductCode"], "C001")


class WriteCsvAtomicTests(TestCase):
    def test_replaces_destination_only_after_successful_write(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.csv"
            write_csv_atomic(pd.DataFrame([{"a": 1}]), path)
            self.assertTrue(path.exists())
            self.assertEqual(pd.read_csv(path).iloc[0]["a"], 1)


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
