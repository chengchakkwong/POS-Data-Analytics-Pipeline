"""Tests for C-class target-stock rules."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import pandas as pd

from pos_pipeline.analysis.target_stock import plan_c_class_row, run_target_stock


def _c_row(**overrides) -> pd.Series:
    base = {
        "ProductCode": "P001",
        "Name": "Test SKU",
        "XYZ_Class": "Z",
        "Strategy": "不建議預測 (考慮汰換)",
        "CurrStock": 5,
        "Mean_Monthly_Qty": 10,
        "FirstOrderQty": 0,
    }
    base.update(overrides)
    return pd.Series(base)


class PlanCClassRowTests(TestCase):
    def test_zero_mean_qty_returns_zero_target(self) -> None:
        result = plan_c_class_row(_c_row(Mean_Monthly_Qty=0, FirstOrderQty=24))
        self.assertEqual(result["Target_Stock"], 0)

    def test_oversized_first_order_falls_back_to_mean_times_1_2(self) -> None:
        # mean 10 → 一年量 120；FirstOrderQty 200 過大
        result = plan_c_class_row(_c_row(Mean_Monthly_Qty=10, FirstOrderQty=200))
        self.assertEqual(result["Target_Stock"], 12.0)

    def test_reasonable_first_order_is_used(self) -> None:
        result = plan_c_class_row(_c_row(Mean_Monthly_Qty=10, FirstOrderQty=24))
        self.assertEqual(result["Target_Stock"], 24)

    def test_missing_first_order_uses_mean_times_1_2(self) -> None:
        result = plan_c_class_row(_c_row(Mean_Monthly_Qty=10, FirstOrderQty=0))
        self.assertEqual(result["Target_Stock"], 12.0)

class RunTargetStockTests(TestCase):
    def test_keeps_only_c_class_and_parses_note(self) -> None:
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
            self.assertEqual(result.iloc[0]["Note"], "24 箱規")

if __name__ == "__main__":
    import unittest

    unittest.main()