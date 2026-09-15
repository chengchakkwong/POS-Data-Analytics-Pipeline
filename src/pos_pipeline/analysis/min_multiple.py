"""Guessed min / multiple from inbound quantity history (factor voting)."""

from __future__ import annotations

import math
from collections import defaultdict

import pandas as pd


def guess_min_and_multiple(
    history_records: list,
    threshold_ratio: float = 0.6,
) -> tuple[int | None, int | None]:
    """Infer Min (MOQ) and Multiple from inbound qty history via factor voting.

    Factors that appear in at least ``threshold_ratio`` of records become
    candidates; the largest such factor is Multiple. Min is the smallest
    historical qty divisible by Multiple (else the global minimum).
    """
    clean = [
        int(x)
        for x in history_records
        if x is not None and str(x).strip() != ""
    ]
    try:
        clean = [x for x in clean if x > 0]
    except (TypeError, ValueError):
        return None, None

    if not clean:
        return None, None
    if len(clean) == 1:
        return clean[0], clean[0]

    total_count = len(clean)
    factor_votes: dict[int, int] = defaultdict(int)

    for num in clean:
        limit = int(math.sqrt(num)) + 1
        for i in range(1, limit):
            if num % i == 0:
                factor_votes[i] += 1
                if i != num // i:
                    factor_votes[num // i] += 1

    required_votes = math.ceil(total_count * threshold_ratio)

    guessed_multiple = 1
    for factor, votes in factor_votes.items():
        if votes >= required_votes and factor > guessed_multiple:
            guessed_multiple = factor

    valid_candidates = [x for x in set(clean) if x % guessed_multiple == 0]
    if valid_candidates:
        guessed_min = min(valid_candidates)
    else:
        guessed_min = min(clean)

    return guessed_min, guessed_multiple


def compute_guessed_min_multiple(df_inbound: pd.DataFrame) -> pd.DataFrame:
    """Group inbound rows by GoodsNo and compute guessed_min / guessed_multiple.

    Expects columns ``GoodsNo`` and ``ChQty``. Returns
    ``ProductCode``, ``guessed_min``, ``guessed_multiple`` (empty if none).
    """
    empty = pd.DataFrame(
        columns=["ProductCode", "guessed_min", "guessed_multiple"]
    )
    if df_inbound is None or df_inbound.empty:
        return empty

    required = {"GoodsNo", "ChQty"}
    if not required.issubset(df_inbound.columns):
        return empty

    df = df_inbound.copy()
    df["ChQty"] = pd.to_numeric(df["ChQty"], errors="coerce").fillna(0).astype(int)
    df = df[df["ChQty"] > 0]
    if df.empty:
        return empty

    rows: list[dict] = []
    for goods_no, group in df.groupby("GoodsNo", sort=False):
        guessed_min, guessed_multiple = guess_min_and_multiple(
            group["ChQty"].tolist()
        )
        if guessed_min is None or guessed_multiple is None:
            continue
        product_code = str(goods_no).strip()
        if not product_code or product_code.lower() == "nan":
            continue
        rows.append(
            {
                "ProductCode": product_code,
                "guessed_min": guessed_min,
                "guessed_multiple": guessed_multiple,
            }
        )

    if not rows:
        return empty
    return pd.DataFrame(rows)[
        ["ProductCode", "guessed_min", "guessed_multiple"]
    ]
