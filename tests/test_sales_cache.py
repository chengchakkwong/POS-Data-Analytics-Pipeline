"""Tests for incremental sales parquet sync."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import pandas as pd

from pos_pipeline.extraction.sales import DEFAULT_SALES_START
from pos_pipeline.storage.sales_cache import sync_daily_sales_parquet


class SyncDailySalesParquetTests(TestCase):
    @patch("pos_pipeline.storage.sales_cache.fetch_daily_sales")
    def test_uses_default_start_when_cache_missing(
        self,
        mock_fetch,
    ) -> None:
        mock_fetch.return_value = pd.DataFrame()
        missing = Path("sales-cache-does-not-exist")

        sync_daily_sales_parquet(cache_dir=missing)

        mock_fetch.assert_called_once_with(start_date=DEFAULT_SALES_START)

    @patch("pos_pipeline.storage.sales_cache.load_sales_max_date")
    @patch("pos_pipeline.storage.sales_cache.fetch_daily_sales")
    def test_looks_back_one_day_when_cache_exists(
        self,
        mock_fetch,
        mock_max_date,
    ) -> None:
        mock_fetch.return_value = pd.DataFrame()
        mock_max_date.return_value = pd.Timestamp("2026-09-10")

        with TemporaryDirectory() as tmp:
            sync_daily_sales_parquet(cache_dir=Path(tmp))

        mock_fetch.assert_called_once_with(start_date="2026-09-09")


if __name__ == "__main__":
    import unittest

    unittest.main()