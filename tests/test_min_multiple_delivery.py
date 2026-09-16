"""Tests for guessed min/multiple Firestore delivery."""

from unittest import TestCase
from unittest.mock import patch

import pandas as pd

from pos_pipeline.delivery_to_firebase.min_multiple import (
    MIN_MULTIPLE_STATE,
    prepare_min_multiple_df,
    upload_guessed_min_multiple,
)
from pos_pipeline.delivery_to_firebase.sync_state import content_hash


class PrepareMinMultipleDfTests(TestCase):
    def test_strips_product_code(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "ProductCode": " A001 ",
                    "guessed_min": 12,
                    "guessed_multiple": 6,
                }
            ]
        )
        result = prepare_min_multiple_df(df)
        self.assertEqual(result.iloc[0]["ProductCode"], "A001")
        self.assertEqual(result.iloc[0]["guessed_min"], 12)

    def test_requires_columns(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            prepare_min_multiple_df(pd.DataFrame([{"ProductCode": "A"}]))


class UploadGuessedMinMultipleTests(TestCase):
    @patch(
        "pos_pipeline.delivery_to_firebase.min_multiple.save_hashes"
    )
    @patch(
        "pos_pipeline.delivery_to_firebase.min_multiple.load_hashes",
        return_value={},
    )
    @patch(
        "pos_pipeline.delivery_to_firebase.min_multiple.get_firestore_client"
    )
    def test_merges_only_guessed_fields(
        self,
        mock_get_client,
        _mock_load_hashes,
        mock_save_hashes,
    ) -> None:
        db = mock_get_client.return_value
        batch = db.batch.return_value
        doc_ref = db.collection.return_value.document.return_value
        df = pd.DataFrame(
            [
                {
                    "ProductCode": "A001",
                    "guessed_min": 12,
                    "guessed_multiple": 6,
                }
            ]
        )

        written = upload_guessed_min_multiple(df)

        self.assertEqual(written, 1)
        db.collection.assert_called_with("replenishment")
        batch.set.assert_called_once_with(
            doc_ref,
            {"guessed_min": 12, "guessed_multiple": 6},
            merge=True,
        )
        mock_save_hashes.assert_called_once()
        state_name, hashes = mock_save_hashes.call_args.args
        self.assertEqual(state_name, MIN_MULTIPLE_STATE)
        self.assertIn("A001", hashes)

    @patch(
        "pos_pipeline.delivery_to_firebase.min_multiple.save_hashes"
    )
    @patch(
        "pos_pipeline.delivery_to_firebase.min_multiple.get_firestore_client"
    )
    def test_skips_unchanged_hash(
        self,
        mock_get_client,
        mock_save_hashes,
    ) -> None:
        item = {
            "ProductCode": "A001",
            "guessed_min": 12,
            "guessed_multiple": 6,
        }
        digest = content_hash(item, ignore_keys=set())

        with patch(
            "pos_pipeline.delivery_to_firebase.min_multiple.load_hashes",
            return_value={"A001": digest},
        ):
            written = upload_guessed_min_multiple(pd.DataFrame([item]))

        self.assertEqual(written, 0)
        mock_get_client.return_value.batch.return_value.set.assert_not_called()
        mock_save_hashes.assert_not_called()
