"""Tests for analytics result Firestore delivery."""

from unittest import TestCase
from unittest.mock import patch

import pandas as pd

from pos_pipeline.delivery_to_firebase.analytics_results import (
    prepare_classification_df,
    prepare_target_stock_df,
    upload_analytics_results,
    upload_classification,
    upload_target_stock,
)


class PrepareClassificationDfTests(TestCase):
    def test_maps_note_and_skips_blank_product_code(self) -> None:
        labels = pd.DataFrame(
            [
                {
                    "ProductCode": " A001 ",
                    "ABC_Class": "A",
                    "XYZ_Class": "X",
                    "Note": "24 箱規",
                },
                {
                    "ProductCode": "  ",
                    "ABC_Class": "C",
                    "XYZ_Class": "Z",
                    "Note": "skip",
                },
            ]
        )
        result = prepare_classification_df(labels)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["ProductCode"], "A001")
        self.assertEqual(result.iloc[0]["note"], "24 箱規")
        self.assertEqual(result.iloc[0]["ABC_Class"], "A")

    def test_requires_abc_xyz_columns(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            prepare_classification_df(
                pd.DataFrame([{"ProductCode": "A001", "ABC_Class": "A"}])
            )


class PrepareTargetStockDfTests(TestCase):
    def test_rounds_target_stock_to_int(self) -> None:
        plan = pd.DataFrame(
            [
                {"ProductCode": "A001", "Target_Stock": 12.6},
                {"ProductCode": "B001", "Target_Stock": 3.2},
            ]
        )
        result = prepare_target_stock_df(plan)
        self.assertEqual(result.iloc[0]["Target_Stock"], 13)
        self.assertEqual(result.iloc[1]["Target_Stock"], 3)

    def test_falls_back_to_base_demand(self) -> None:
        plan = pd.DataFrame([{"ProductCode": "A001", "Base_Demand": 8.4}])
        result = prepare_target_stock_df(plan)
        self.assertEqual(result.iloc[0]["Target_Stock"], 8)


class UploadAnalyticsResultsTests(TestCase):
    @patch(
        "pos_pipeline.delivery_to_firebase.analytics_results.save_hashes"
    )
    @patch(
        "pos_pipeline.delivery_to_firebase.analytics_results.load_hashes",
        return_value={},
    )
    @patch(
        "pos_pipeline.delivery_to_firebase.analytics_results.get_firestore_client"
    )
    def test_classification_writes_products_and_replenishment(
        self,
        mock_get_client,
        _mock_load_hashes,
        mock_save_hashes,
    ) -> None:
        db = mock_get_client.return_value
        batch = db.batch.return_value
        labels = pd.DataFrame(
            [
                {
                    "ProductCode": "A001",
                    "ABC_Class": "A",
                    "XYZ_Class": "X",
                    "Note": "hot",
                }
            ]
        )

        written = upload_classification(labels)

        self.assertEqual(written, 1)
        db.collection.assert_any_call("products")
        db.collection.assert_any_call("replenishment")
        self.assertEqual(batch.set.call_count, 2)
        for set_call in batch.set.call_args_list:
            _doc_ref, payload = set_call.args
            self.assertEqual(payload["ABC_Class"], "A")
            self.assertEqual(payload["XYZ_Class"], "X")
            self.assertEqual(payload["note"], "hot")
            self.assertNotIn("ProductCode", payload)
            self.assertTrue(set_call.kwargs["merge"])
        batch.commit.assert_called_once_with()
        mock_save_hashes.assert_called_once()
        self.assertEqual(mock_save_hashes.call_args.args[0], "classification")

    @patch(
        "pos_pipeline.delivery_to_firebase.analytics_results.save_hashes"
    )
    @patch(
        "pos_pipeline.delivery_to_firebase.analytics_results.load_hashes",
        return_value={"A001": "same"},
    )
    @patch(
        "pos_pipeline.delivery_to_firebase.analytics_results.content_hash",
        return_value="same",
    )
    @patch(
        "pos_pipeline.delivery_to_firebase.analytics_results.get_firestore_client"
    )
    def test_skips_unchanged_target_stock(
        self,
        mock_get_client,
        _mock_hash,
        _mock_load_hashes,
        mock_save_hashes,
    ) -> None:
        plan = pd.DataFrame([{"ProductCode": "A001", "Target_Stock": 10}])
        written = upload_target_stock(plan)
        self.assertEqual(written, 0)
        mock_get_client.return_value.batch.return_value.set.assert_not_called()
        mock_save_hashes.assert_not_called()

    @patch(
        "pos_pipeline.delivery_to_firebase.analytics_results.upload_target_stock",
        return_value=2,
    )
    @patch(
        "pos_pipeline.delivery_to_firebase.analytics_results.upload_classification",
        return_value=3,
    )
    def test_upload_analytics_results_returns_both_counts(
        self,
        mock_class,
        mock_target,
    ) -> None:
        labels = pd.DataFrame([{"ProductCode": "A001"}])
        plan = pd.DataFrame([{"ProductCode": "A001"}])
        class_written, target_written = upload_analytics_results(labels, plan)
        self.assertEqual(class_written, 3)
        self.assertEqual(target_written, 2)
        mock_class.assert_called_once_with(labels)
        mock_target.assert_called_once_with(plan)


if __name__ == "__main__":
    import unittest

    unittest.main()
