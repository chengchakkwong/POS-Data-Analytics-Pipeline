"""Tests for ABC/XYZ analysis runner."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import pandas as pd

from pos_pipeline.analysis.abc_xyz import run_abc_xyz


class RunAbcXyzTests(TestCase):
    @patch("pos_pipeline.analysis.abc_xyz.analyze_xyz")
    @patch("pos_pipeline.analysis.abc_xyz.analyze_profit_abc")
    @patch("pos_pipeline.analysis.abc_xyz.load_sales_parquet")
    @patch("pos_pipeline.analysis.abc_xyz.pd.read_csv")
    def test_splits_recent_sales_for_abc_and_keeps_full_history(
        self,
        mock_read_csv,
        mock_load_sales,
        mock_analyze_profit_abc,
        mock_analyze_xyz,
    ) -> None:
        mock_read_csv.return_value = pd.DataFrame([{"GoodsID": 1}])
        mock_load_sales.return_value = pd.DataFrame(
            [
                {
                    "GoodsID": 1,
                    "rDate": "2025-01-01",
                    "TotalQty": 1,
                    "TotalAmt": 10,
                },
                {
                    "GoodsID": 1,
                    "rDate": "2026-09-01",
                    "TotalQty": 2,
                    "TotalAmt": 20,
                },
            ]
        )
        mock_analyze_profit_abc.return_value = pd.DataFrame(
            [
                {
                    "GoodsID": 1,
                    "ABC_Class": "A",
                    "Name": "Test",
                    "ProductCode": "P001",
                    "Monthly_Avg_Profit": 10,
                }
            ]
        )
        mock_analyze_xyz.return_value = pd.DataFrame(
            [
                {
                    "GoodsID": 1,
                    "CV": 0.2,
                    "XYZ_Class": "X",
                    "Mean_Monthly_Qty": 2,
                }
            ]
        )

        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "abc_xyz_analysis.csv"
            result = run_abc_xyz(
                stock_csv=Path("fake-stock.csv"),
                sales_dir=Path("fake-sales"),
                output_csv=output,
            )

            self.assertTrue(output.exists())
            self.assertEqual(result.iloc[0]["ABC_Class"], "A")
            self.assertEqual(result.iloc[0]["XYZ_Class"], "X")
            self.assertEqual(result.iloc[0]["Strategy"], "自動補貨 (高頻穩定)")

        recent_sales = mock_analyze_profit_abc.call_args.args[1]
        month_age_map = mock_analyze_profit_abc.call_args.kwargs["month_age_map"]
        full_sales = mock_analyze_xyz.call_args.args[0]

        self.assertEqual(len(recent_sales), 1)
        self.assertEqual(
            pd.Timestamp(recent_sales.iloc[0]["rDate"]),
            pd.Timestamp("2026-09-01"),
        )
        self.assertEqual(month_age_map[1], 20)
        self.assertEqual(len(full_sales), 2)


if __name__ == "__main__":
    import unittest

    unittest.main()