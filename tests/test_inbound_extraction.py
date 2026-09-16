"""Tests for inbound movement extraction."""

from datetime import date
from unittest import TestCase
from unittest.mock import patch

import pandas as pd

from pos_pipeline.extraction.inbound import (
    INBOUND_MOVEMENTS_SQL,
    fetch_new_inbound_movements,
)


class FetchNewInboundMovementsTests(TestCase):
    @patch("pos_pipeline.extraction.inbound.execute_query")
    def test_builds_watermark_and_date_parameters(
        self,
        mock_execute_query,
    ) -> None:
        expected = pd.DataFrame([{"SID": 124}])
        mock_execute_query.return_value = expected

        result = fetch_new_inbound_movements(
            last_sid=123,
            days=14,
            as_of=date(2026, 9, 9),
        )

        self.assertIs(result, expected)
        mock_execute_query.assert_called_once_with(
            INBOUND_MOVEMENTS_SQL,
            params={
                "last_sid": 123,
                "bill_start": 20260826,
            },
        )


if __name__ == "__main__":
    import unittest

    unittest.main()