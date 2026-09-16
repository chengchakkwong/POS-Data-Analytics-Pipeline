"""Tests for inbound Firestore delivery."""

from unittest import TestCase
from unittest.mock import patch

import pandas as pd

from pos_pipeline.delivery_to_firebase.inbound import (
    upload_inbound_movements,
)


class UploadInboundMovementsTests(TestCase):
    @patch(
        "pos_pipeline.delivery_to_firebase.inbound.save_last_sid"
    )
    @patch(
        "pos_pipeline.delivery_to_firebase.inbound.get_firestore_client"
    )
    def test_uploads_record_and_advances_watermark(
        self,
        mock_get_firestore_client,
        mock_save_last_sid,
    ) -> None:
        df = pd.DataFrame(
            [
                {
                    "SID": 124,
                    "BillDate": 20260913,
                    "GoodsNo": "G001",
                    "ChQty": 5,
                }
            ]
        )

        db = mock_get_firestore_client.return_value
        batch = db.batch.return_value
        doc_ref = db.collection.return_value.document.return_value

        written = upload_inbound_movements(df)

        self.assertEqual(written, 1)
        db.collection.assert_called_once_with("inbound_movements")
        db.collection.return_value.document.assert_called_once_with("124")

        actual_doc_ref, payload = batch.set.call_args.args
        self.assertIs(actual_doc_ref, doc_ref)
        self.assertEqual(payload["SID"], 124)
        self.assertEqual(payload["GoodsNo"], "G001")
        self.assertEqual(payload["ChQty"], 5)

        batch.commit.assert_called_once_with()
        mock_save_last_sid.assert_called_once_with(124)


if __name__ == "__main__":
    import unittest

    unittest.main()