"""Upload ABC/XYZ labels and Target_Stock to Firestore."""

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
CLASSIFICATION_STATE = "classification"
TARGET_STOCK_STATE = "target_stock"


def prepare_classification_df(labels_df: pd.DataFrame) -> pd.DataFrame:
    """Build ProductCode + ABC/XYZ + note payloads for Firestore merge."""
    if labels_df is None or labels_df.empty:
        return pd.DataFrame(
            columns=["ProductCode", "ABC_Class", "XYZ_Class", "note"]
        )

    required = {"ProductCode", "ABC_Class", "XYZ_Class"}
    missing = required.difference(labels_df.columns)
    if missing:
        missing_names = ", ".join(sorted(missing))
        raise ValueError(
            f"classification data is missing required columns: {missing_names}"
        )

    frame = labels_df.copy()
    if "Note" in frame.columns:
        note_source = frame["Note"]
    elif "note" in frame.columns:
        note_source = frame["note"]
    else:
        note_source = pd.Series([None] * len(frame), index=frame.index)

    out = pd.DataFrame(
        {
            "ProductCode": frame["ProductCode"].astype(str).str.strip(),
            "ABC_Class": frame["ABC_Class"],
            "XYZ_Class": frame["XYZ_Class"],
            "note": note_source,
        }
    )
    out = out[out["ProductCode"] != ""]
    out = out.where(pd.notnull(out), None)
    return out.reset_index(drop=True)


def prepare_target_stock_df(plan_df: pd.DataFrame) -> pd.DataFrame:
    """Build ProductCode + integer Target_Stock payloads for Firestore merge."""
    if plan_df is None or plan_df.empty:
        return pd.DataFrame(columns=["ProductCode", "Target_Stock"])

    if "ProductCode" not in plan_df.columns:
        raise ValueError("target-stock plan is missing required column: ProductCode")

    if "Target_Stock" in plan_df.columns:
        source = plan_df["Target_Stock"]
    elif "Base_Demand" in plan_df.columns:
        source = plan_df["Base_Demand"]
    else:
        raise ValueError(
            "target-stock plan must contain either Target_Stock or Base_Demand"
        )

    target_stock = pd.to_numeric(source, errors="coerce").fillna(0).round().astype(int)
    out = pd.DataFrame(
        {
            "ProductCode": plan_df["ProductCode"].astype(str).str.strip(),
            "Target_Stock": target_stock,
        }
    )
    out = out[out["ProductCode"] != ""]
    return out.reset_index(drop=True)


def _upload_hashed_payloads(
    *,
    records: list[dict[str, Any]],
    state_name: str,
    collections: list[str],
) -> int:
    """Merge payloads into collections, skipping unchanged hashes."""
    if not records:
        return 0

    db = get_firestore_client()
    known_hashes = load_hashes(state_name)
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

        payload = {key: value for key, value in item.items() if key != "ProductCode"}
        for collection_name in collections:
            doc_ref = db.collection(collection_name).document(product_code)
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
        save_hashes(state_name, new_hashes)

    return written


def upload_classification(labels_df: pd.DataFrame) -> int:
    """Upload ABC/XYZ + note to products and replenishment. Returns docs written."""
    prepared = prepare_classification_df(labels_df)
    records = prepared.to_dict(orient="records")
    return _upload_hashed_payloads(
        records=records,
        state_name=CLASSIFICATION_STATE,
        collections=["products", "replenishment"],
    )


def upload_target_stock(plan_df: pd.DataFrame) -> int:
    """Upload Target_Stock to replenishment only. Returns docs written."""
    prepared = prepare_target_stock_df(plan_df)
    records = prepared.to_dict(orient="records")
    return _upload_hashed_payloads(
        records=records,
        state_name=TARGET_STOCK_STATE,
        collections=["replenishment"],
    )


def upload_analytics_results(
    labels_df: pd.DataFrame,
    plan_df: pd.DataFrame,
) -> tuple[int, int]:
    """Upload classification then Target_Stock. Returns (class_written, target_written)."""
    class_written = upload_classification(labels_df)
    target_written = upload_target_stock(plan_df)
    return class_written, target_written
