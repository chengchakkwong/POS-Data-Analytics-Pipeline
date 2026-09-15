"""Tests for inbound history extraction used by min-multiple."""

from datetime import date
from unittest import TestCase
from unittest.mock import patch

import pandas as pd

from pos_pipeline.extraction.inbound import (
    INBOUND_FOR_MIN_MULTIPLE_SQL,
    fetch_inbound_for_min_multiple,
)


class FetchInboundForMinMultipleTests(TestCase):
    @patch("pos_pipeline.extraction.inbound.execute_query")
    def test_builds_year_window_parameters(
        self,
        mock_execute_query,
    ) -> None:
        expected = pd.DataFrame([{"GoodsNo": "A", "ChQty": 12}])
        mock_execute_query.return_value = expected

        result = fetch_inbound_for_min_multiple(
            years=2,
            as_of=date(2026, 9, 9),
        )

        self.assertIs(result, expected)
        mock_execute_query.assert_called_once_with(
            INBOUND_FOR_MIN_MULTIPLE_SQL,
            params={
                "bill_start": 20240909,
                "bill_end": 20260909,
            },
        )

    def test_rejects_invalid_years(self) -> None:
        with self.assertRaises(ValueError):
            fetch_inbound_for_min_multiple(years=0)
