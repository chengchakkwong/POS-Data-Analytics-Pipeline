"""Upload guessed_min / guessed_multiple to Firestore replenishment."""

from __future__ import annotations

from typing import Any

import pandas as pd

from pos_pipeline.delivery_to_firebase.client import get_firestore_client
from pos_pipeline.delivery_to_firebase.sync_state import (
    content_hash,
    load_hashes,
    save_hashes,
)

BATCH_SIZE = 400
MIN_MULTIPLE_STATE = "min_multiple"
PAYLOAD_COLUMNS = ["guessed_min", "guessed_multiple"]


def prepare_min_multiple_df(df: pd.DataFrame) -> pd.DataFrame:
    """Keep ProductCode + guessed_min / guessed_multiple for merge upload."""
    empty = pd.DataFrame(
        columns=["ProductCode", "guessed_min", "guessed_multiple"]
    )
    if df is None or df.empty:
        return empty

    required = {"ProductCode", "guessed_min", "guessed_multiple"}
    missing = required.difference(df.columns)
    if missing:
        missing_names = ", ".join(sorted(missing))
        raise ValueError(
            f"min-multiple data is missing required columns: {missing_names}"
        )

    out = pd.DataFrame(
        {
            "ProductCode": df["ProductCode"].astype(str).str.strip(),
            "guessed_min": df["guessed_min"],
            "guessed_multiple": df["guessed_multiple"],
        }
    )
    out = out[out["ProductCode"] != ""]
    out = out.where(pd.notnull(out), None)
    return out.reset_index(drop=True)


def upload_guessed_min_multiple(df: pd.DataFrame) -> int:
    """Merge guessed_min / guessed_multiple into replenishment only.

    Skips docs whose hash matches ``sync_state/min_multiple``.
    Does not touch MinOrderQty / OrderMultiple or other replenishment fields.
    Returns number of product documents written.
    """
    prepared = prepare_min_multiple_df(df)
    if prepared.empty:
        return 0

    records: list[dict[str, Any]] = prepared.to_dict(orient="records")
    db = get_firestore_client()
    known_hashes = load_hashes(MIN_MULTIPLE_STATE)
    new_hashes = dict(known_hashes)
    written = 0
    batch = db.batch()
    batch_count = 0

    for item in records:
        product_code = str(item.get("ProductCode", "")).strip()
        if not product_code:
            continue

        digest = content_hash(item, ignore_keys=set())
        if new_hashes.get(product_code) == digest:
            continue

        payload = {key: item.get(key) for key in PAYLOAD_COLUMNS}
        doc_ref = db.collection("replenishment").document(product_code)
        batch.set(doc_ref, payload, merge=True)
        batch_count += 1

        new_hashes[product_code] = digest
        written += 1

        if batch_count >= BATCH_SIZE:
            batch.commit()
            batch = db.batch()
            batch_count = 0

    if batch_count:
        batch.commit()

    if new_hashes != known_hashes:
        save_hashes(MIN_MULTIPLE_STATE, new_hashes)

    return written
