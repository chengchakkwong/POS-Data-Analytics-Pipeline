"""Unit tests for guessed min / multiple factor voting."""

from unittest import TestCase

import pandas as pd

from pos_pipeline.analysis.min_multiple import (
    compute_guessed_min_multiple,
    guess_min_and_multiple,
)


class GuessMinAndMultipleTests(TestCase):
    def test_case_a_regular_pattern(self) -> None:
        self.assertEqual(guess_min_and_multiple([24, 36, 12, 48]), (12, 12))

    def test_case_b_noise_typo(self) -> None:
        self.assertEqual(
            guess_min_and_multiple([24, 24, 25, 36, 12]),
            (12, 12),
        )

    def test_case_c_sparse(self) -> None:
        self.assertEqual(guess_min_and_multiple([2, 3, 4]), (2, 2))

    def test_case_d_sample_order_mixed(self) -> None:
        self.assertEqual(
            guess_min_and_multiple([2, 24, 48, 24]),
            (24, 24),
        )

    def test_empty_and_single(self) -> None:
        self.assertEqual(guess_min_and_multiple([]), (None, None))
        self.assertEqual(guess_min_and_multiple([12]), (12, 12))


class ComputeGuessedMinMultipleTests(TestCase):
    def test_groups_by_goods_no(self) -> None:
        df = pd.DataFrame(
            {
                "GoodsNo": ["A", "A", "B"],
                "ChQty": [24, 12, 10],
            }
        )
        result = compute_guessed_min_multiple(df)
        self.assertEqual(set(result.columns), {
            "ProductCode",
            "guessed_min",
            "guessed_multiple",
        })
        by_code = result.set_index("ProductCode")
        self.assertEqual(
            (int(by_code.loc["A", "guessed_min"]), int(by_code.loc["A", "guessed_multiple"])),
            (12, 12),
        )
        self.assertEqual(
            (int(by_code.loc["B", "guessed_min"]), int(by_code.loc["B", "guessed_multiple"])),
            (10, 10),
        )

    def test_empty_or_missing_columns(self) -> None:
        self.assertTrue(compute_guessed_min_multiple(pd.DataFrame()).empty)
        self.assertTrue(
            compute_guessed_min_multiple(
                pd.DataFrame({"GoodsNo": ["A"]})
            ).empty
        )
