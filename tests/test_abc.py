"""Tests for profit-based ABC classification edge cases."""

from unittest import TestCase

import pandas as pd

from pos_pipeline.analysis.abc import analyze_profit_abc
from pos_pipeline.analysis.abc_xyz import attach_xyz_and_strategy


def _stock_row(goods_id: int, name: str = "SKU", product_code: str = "P001") -> dict:
    return {
        "GoodsID": goods_id,
        "ProductCode": product_code,
        "Barcode": f"B{goods_id}",
        "Name": name,
        "Note": "",
        "CurrStock": 1,
        "RetailPrice": 10,
        "LastInCost": 4,
        "AvgCost": 4,
        "Category": "CAT",
        "InboundLocation": "LOC",
        "Supplier": "SUP",
    }


class AnalyzeProfitAbcTests(TestCase):
    def test_monthly_avg_profit_uses_min_month_age_and_window(self) -> None:
        stock = pd.DataFrame(
            [
                _stock_row(1, name="Old"),
                _stock_row(2, name="Young", product_code="P002"),
            ]
        )
        # 近窗各自總利潤相同：售價 10、成本 4 → 單件利潤 6
        sales = pd.DataFrame(
            [
                {
                    "GoodsID": 1,
                    "rDate": "2026-01-01",
                    "TotalQty": 10,
                    "TotalAmt": 100,
                },
                {
                    "GoodsID": 2,
                    "rDate": "2026-01-01",
                    "TotalQty": 10,
                    "TotalAmt": 100,
                },
            ]
        )
        month_age_map = {1: 24, 2: 6}

        result = analyze_profit_abc(
            stock,
            sales,
            month_age_map=month_age_map,
            abc_window_months=12,
        ).set_index("GoodsID")

        # 老品：60 / min(24,12) = 5；年輕成熟品：60 / min(6,12) = 10
        self.assertEqual(result.loc[1, "Monthly_Avg_Profit"], 5)
        self.assertEqual(result.loc[2, "Monthly_Avg_Profit"], 10)

    def test_sku_that_crosses_70_percent_stays_in_class_a(self) -> None:
        stock = pd.DataFrame(
            [
                _stock_row(1, name="Dominant", product_code="P001"),
                _stock_row(2, name="Small", product_code="P002"),
            ]
        )
        sales = pd.DataFrame(
            [
                {
                    "GoodsID": 1,
                    "rDate": "2026-01-01",
                    "TotalQty": 75,
                    "TotalAmt": 750,
                },
                {
                    "GoodsID": 2,
                    "rDate": "2026-01-01",
                    "TotalQty": 25,
                    "TotalAmt": 250,
                },
            ]
        )
        month_age_map = {1: 12, 2: 12}

        result = analyze_profit_abc(
            stock,
            sales,
            month_age_map=month_age_map,
        ).set_index("GoodsID")

        # 單件利潤 6 → 月均 75*6/12=37.5 與 25*6/12=12.5；第一名累積 75% 仍應為 A
        self.assertEqual(result.loc[1, "ABC_Class"], "A")
        self.assertEqual(result.loc[2, "ABC_Class"], "B")
        self.assertAlmostEqual(result.loc[1, "ProfitCumulativeRatio"], 0.75)

    def test_abc_window_months_must_be_positive(self) -> None:
        stock = pd.DataFrame([_stock_row(1)])
        sales = pd.DataFrame(
            [{"GoodsID": 1, "rDate": "2026-01-01", "TotalQty": 1, "TotalAmt": 10}]
        )
        with self.assertRaisesRegex(ValueError, "abc_window_months"):
            analyze_profit_abc(stock, sales, abc_window_months=0)


class StrategyLabelTests(TestCase):
    def test_y_class_strategy_does_not_claim_detected_seasonality(self) -> None:
        abc = pd.DataFrame(
            [
                {
                    "GoodsID": 1,
                    "ABC_Class": "A",
                    "Name": "A Y",
                    "ProductCode": "AY1",
                    "Monthly_Avg_Profit": 10,
                },
                {
                    "GoodsID": 2,
                    "ABC_Class": "B",
                    "Name": "B Y",
                    "ProductCode": "BY1",
                    "Monthly_Avg_Profit": 5,
                },
                {
                    "GoodsID": 3,
                    "ABC_Class": "C",
                    "Name": "C Y",
                    "ProductCode": "CY1",
                    "Monthly_Avg_Profit": 1,
                },
            ]
        )
        xyz = pd.DataFrame(
            [
                {
                    "GoodsID": 1,
                    "CV": 0.7,
                    "XYZ_Class": "Y",
                    "Mean_Monthly_Qty": 8,
                },
                {
                    "GoodsID": 2,
                    "CV": 0.8,
                    "XYZ_Class": "Y",
                    "Mean_Monthly_Qty": 4,
                },
                {
                    "GoodsID": 3,
                    "CV": 0.9,
                    "XYZ_Class": "Y",
                    "Mean_Monthly_Qty": 2,
                },
            ]
        )

        result = attach_xyz_and_strategy(abc, xyz).set_index("GoodsID")

        self.assertEqual(
            result.loc[1, "Strategy"],
            "AI 趨勢／波動預測 (重點對象)",
        )
        self.assertEqual(result.loc[2, "Strategy"], "中度波動補貨")
        self.assertEqual(
            result.loc[3, "Strategy"],
            "中度波動長尾 (減少庫存)",
        )
        for strategy in result["Strategy"]:
            self.assertNotIn("季節", strategy)


if __name__ == "__main__":
    import unittest

    unittest.main()
