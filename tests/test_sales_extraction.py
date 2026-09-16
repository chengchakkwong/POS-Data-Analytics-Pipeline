"""Tests for daily sales extraction."""

from unittest import TestCase
from unittest.mock import patch

import pandas as pd

from pos_pipeline.extraction.sales import (
    DEFAULT_SALES_START,
    SALES_DAILY_SQL,
    fetch_daily_sales,
)


class FetchDailySalesTests(TestCase):
    @patch("pos_pipeline.extraction.sales.execute_query")
    def test_uses_default_start_date(self, mock_execute_query) -> None:
        expected = pd.DataFrame([{"GoodsID": 1}])
        mock_execute_query.return_value = expected

        result = fetch_daily_sales()

        self.assertIs(result, expected)
        mock_execute_query.assert_called_once_with(
            SALES_DAILY_SQL,
            params={"start_date": DEFAULT_SALES_START},
        )

    @patch("pos_pipeline.extraction.sales.execute_query")
    def test_passes_custom_start_date(self, mock_execute_query) -> None:
        mock_execute_query.return_value = pd.DataFrame()

        fetch_daily_sales(start_date="2026-09-01")

        mock_execute_query.assert_called_once_with(
            SALES_DAILY_SQL,
            params={"start_date": "2026-09-01"},
        )


if __name__ == "__main__":
    import unittest

    unittest.main()