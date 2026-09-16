"""Tests for per-SKU calendar-based XYZ classification."""

from unittest import TestCase

import pandas as pd

from pos_pipeline.analysis.xyz import analyze_xyz


class AnalyzeXyzTests(TestCase):
    def test_classifies_x_y_z_from_complete_per_sku_series(self) -> None:
        sales = pd.DataFrame(
            [
                {"GoodsID": 1, "rDate": "2026-01-01", "TotalQty": 10},
                {"GoodsID": 1, "rDate": "2026-02-01", "TotalQty": 10},
                {"GoodsID": 1, "rDate": "2026-03-01", "TotalQty": 10},
                {"GoodsID": 2, "rDate": "2026-01-01", "TotalQty": 5},
                {"GoodsID": 2, "rDate": "2026-02-01", "TotalQty": 15},
                {"GoodsID": 2, "rDate": "2026-03-01", "TotalQty": 5},
                {"GoodsID": 3, "rDate": "2026-01-01", "TotalQty": 10},
            ]
        )

        result = analyze_xyz(sales, cutoff_month="2026-03-01").set_index(
            "GoodsID"
        )

        self.assertEqual(result.loc[1, "XYZ_Class"], "X")
        self.assertEqual(result.loc[2, "XYZ_Class"], "Y")
        self.assertEqual(result.loc[3, "XYZ_Class"], "Z")
        self.assertAlmostEqual(result.loc[3, "Mean_Monthly_Qty"], 10 / 3)

    def test_new_sku_is_not_padded_with_pre_sale_global_months(self) -> None:
        sales = pd.DataFrame(
            [
                {"GoodsID": 1, "rDate": "2026-01-01", "TotalQty": 10},
                {"GoodsID": 1, "rDate": "2026-03-01", "TotalQty": 10},
                {"GoodsID": 2, "rDate": "2026-03-01", "TotalQty": 20},
            ]
        )

        result = analyze_xyz(sales, cutoff_month="2026-03-01").set_index(
            "GoodsID"
        )

        self.assertEqual(result.loc[2, "Mean_Monthly_Qty"], 20)
        self.assertTrue(pd.isna(result.loc[2, "CV"]))
        self.assertEqual(result.loc[2, "XYZ_Class"], "Z")

    def test_zero_mean_keeps_cv_missing_instead_of_magic_number(self) -> None:
        sales = pd.DataFrame(
            [{"GoodsID": 1, "rDate": "2026-01-01", "TotalQty": 0}]
        )

        result = analyze_xyz(sales, cutoff_month="2026-03-01")

        self.assertEqual(result.iloc[0]["Mean_Monthly_Qty"], 0)
        self.assertTrue(pd.isna(result.iloc[0]["CV"]))
        self.assertEqual(result.iloc[0]["XYZ_Class"], "Z")

    def test_sales_after_cutoff_do_not_create_an_xyz_row(self) -> None:
        sales = pd.DataFrame(
            [{"GoodsID": 1, "rDate": "2026-04-01", "TotalQty": 10}]
        )

        result = analyze_xyz(sales, cutoff_month="2026-03-01")

        self.assertTrue(result.empty)
        self.assertEqual(
            result.columns.tolist(),
            ["GoodsID", "CV", "XYZ_Class", "Mean_Monthly_Qty"],
        )

    def test_empty_sales_returns_empty_result_with_contract_columns(self) -> None:
        sales = pd.DataFrame(columns=["GoodsID", "rDate", "TotalQty"])

        result = analyze_xyz(sales)

        self.assertTrue(result.empty)
        self.assertEqual(
            result.columns.tolist(),
            ["GoodsID", "CV", "XYZ_Class", "Mean_Monthly_Qty"],
        )

    def test_missing_required_column_raises_clear_error(self) -> None:
        sales = pd.DataFrame([{"GoodsID": 1, "rDate": "2026-01-01"}])

        with self.assertRaisesRegex(ValueError, "TotalQty"):
            analyze_xyz(sales, cutoff_month="2026-01-01")


if __name__ == "__main__":
    import unittest

    unittest.main()
