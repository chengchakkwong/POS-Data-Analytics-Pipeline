"""Tests for the biweekly analytics job orchestration."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import pandas as pd

from pos_pipeline.jobs.analytics import run


class AnalyticsJobTests(TestCase):
    @patch("pos_pipeline.jobs.analytics.check_connection", return_value=False)
    def test_returns_1_when_database_unavailable(self, _mock_check) -> None:
        self.assertEqual(run(), 1)

    @patch("pos_pipeline.jobs.analytics.check_connection", return_value=True)
    @patch("pos_pipeline.jobs.analytics.fetch_stock_master")
    def test_returns_1_when_stock_master_is_empty(
        self,
        mock_fetch_stock,
        _mock_check,
    ) -> None:
        mock_fetch_stock.return_value = pd.DataFrame()
        self.assertEqual(run(), 1)

    @patch("pos_pipeline.jobs.analytics.check_connection", return_value=True)
    @patch("pos_pipeline.jobs.analytics.fetch_stock_master")
    @patch("pos_pipeline.jobs.analytics.save_stock_master")
    @patch("pos_pipeline.jobs.analytics.sync_daily_sales_parquet")
    @patch("pos_pipeline.jobs.analytics.run_abc_xyz")
    @patch("pos_pipeline.jobs.analytics.run_target_stock")
    def test_happy_path_passes_sales_dir_and_backend(
        self,
        mock_target_stock,
        mock_abc_xyz,
        mock_sync_sales,
        mock_save_stock,
        mock_fetch_stock,
        _mock_check,
    ) -> None:
        mock_fetch_stock.return_value = pd.DataFrame([{"GoodsID": 1}])
        mock_save_stock.return_value = Path("stock.csv")
        mock_sync_sales.return_value = pd.DataFrame([{"GoodsID": 1}])
        mock_abc_xyz.return_value = pd.DataFrame([{"GoodsID": 1}])
        mock_target_stock.return_value = pd.DataFrame([{"ProductCode": "A001"}])

        with patch.dict("os.environ", {"FORECAST_BACKEND": "recent_3m"}):
            code = run()

        self.assertEqual(code, 0)
        mock_abc_xyz.assert_called_once()
        self.assertIn("sales_dir", mock_abc_xyz.call_args.kwargs)
        mock_target_stock.assert_called_once()
        kwargs = mock_target_stock.call_args.kwargs
        self.assertIn("sales_dir", kwargs)
        self.assertEqual(kwargs["backend_name"], "recent_3m")

    @patch("pos_pipeline.jobs.analytics.check_connection", return_value=True)
    @patch("pos_pipeline.jobs.analytics.fetch_stock_master")
    @patch("pos_pipeline.jobs.analytics.save_stock_master")
    @patch("pos_pipeline.jobs.analytics.sync_daily_sales_parquet")
    @patch("pos_pipeline.jobs.analytics.run_abc_xyz")
    @patch("pos_pipeline.jobs.analytics.run_target_stock")
    def test_returns_1_when_target_stock_raises(
        self,
        mock_target_stock,
        mock_abc_xyz,
        mock_sync_sales,
        mock_save_stock,
        mock_fetch_stock,
        _mock_check,
    ) -> None:
        mock_fetch_stock.return_value = pd.DataFrame([{"GoodsID": 1}])
        mock_save_stock.return_value = Path("stock.csv")
        mock_sync_sales.return_value = pd.DataFrame([{"GoodsID": 1}])
        mock_abc_xyz.return_value = pd.DataFrame([{"GoodsID": 1}])
        mock_target_stock.side_effect = RuntimeError("fit failed")

        self.assertEqual(run(), 1)

    @patch("pos_pipeline.jobs.analytics.check_connection", return_value=True)
    @patch("pos_pipeline.jobs.analytics.fetch_stock_master")
    @patch("pos_pipeline.jobs.analytics.save_stock_master")
    @patch("pos_pipeline.jobs.analytics.sync_daily_sales_parquet")
    @patch("pos_pipeline.jobs.analytics.run_abc_xyz")
    def test_returns_1_when_abc_xyz_raises(
        self,
        mock_abc_xyz,
        mock_sync_sales,
        mock_save_stock,
        mock_fetch_stock,
        _mock_check,
    ) -> None:
        mock_fetch_stock.return_value = pd.DataFrame([{"GoodsID": 1}])
        mock_save_stock.return_value = Path("stock.csv")
        mock_sync_sales.return_value = pd.DataFrame([{"GoodsID": 1}])
        mock_abc_xyz.side_effect = ValueError("sales data is empty")

        self.assertEqual(run(), 1)


class OfflineAnalyticsPipelineTests(TestCase):
    """End-to-end classify + plan using anonymized sample_data (no SQL)."""

    def test_sample_data_produces_plan_and_trace(self) -> None:
        from pos_pipeline.analysis.abc_xyz import run_abc_xyz
        from pos_pipeline.analysis.forecasting import RecentMeanBackend
        from pos_pipeline.analysis.target_stock import PLAN_COLUMNS, run_target_stock
        from parquet_utils import load_sales_parquet

        sample_dir = Path("sample_data")
        stock_path = sample_dir / "stock.csv"
        sales_path = sample_dir / "sales.parquet"
        if not stock_path.exists() or not sales_path.exists():
            self.skipTest("sample_data stock/sales files are not available")

        sales_df = load_sales_parquet(sales_path).copy()
        sales_df["rDate"] = pd.to_datetime(
            sales_df["rDate"].astype(str),
            format="%Y%m%d",
        )
        self.assertFalse(sales_df.empty)

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            prepared_stock = tmp_path / "stock.csv"
            prepared_sales = tmp_path / "sales.parquet"
            labels_csv = tmp_path / "abc_xyz.csv"
            plan_csv = tmp_path / "target_stock.csv"
            trace_csv = tmp_path / "target_stock_trace.csv"

            pd.read_csv(stock_path, encoding="utf-8-sig").to_csv(
                prepared_stock,
                index=False,
                encoding="utf-8-sig",
            )
            sales_df.to_parquet(prepared_sales, index=False)

            labels = run_abc_xyz(
                stock_csv=prepared_stock,
                sales_dir=prepared_sales,
                output_csv=labels_csv,
            )
            self.assertGreater(len(labels), 0)
            self.assertTrue({"A", "B", "C", "New"} & set(labels["ABC_Class"]))

            plan = run_target_stock(
                labels_csv=labels_csv,
                stock_csv=prepared_stock,
                output_csv=plan_csv,
                trace_csv=trace_csv,
                sales_df=sales_df,
                backend=RecentMeanBackend(),
            )

            self.assertTrue(plan_csv.exists())
            self.assertTrue(trace_csv.exists())
            self.assertGreater(len(plan), 0)
            for column in PLAN_COLUMNS:
                self.assertIn(column, plan.columns)
            self.assertTrue(set(plan["ABC_XYZ"].astype(str).str[0]).issubset(
                {"A", "B", "C", "N"}
            ))
            trace = pd.read_csv(trace_csv)
            self.assertIn("Forecast_Method", trace.columns)
            self.assertIn("Decision_Source", trace.columns)
            self.assertEqual(len(trace), len(plan))


if __name__ == "__main__":
    import unittest

    unittest.main()
