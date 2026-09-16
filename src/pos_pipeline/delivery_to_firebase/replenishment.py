"""Prepare replenishment fields from stock master."""

from __future__ import annotations
from pos_pipeline.delivery_to_firebase.client import get_firestore_client

import pandas as pd

from pos_pipeline.delivery_to_firebase.sync_state import (
    content_hash,
    load_hashes,
    save_hashes,
)

REPLENISHMENT_COLS = [
    "ProductCode",
    "LastInCost",
    "AvgCost",
    "InboundLocation",
    "FirstOrderQty",
    "NoteDescription",
]


def prepare_replenishment_df(df_stock: pd.DataFrame) -> pd.DataFrame:
    """Parse Note into FirstOrderQty / NoteDescription; keep full stock columns."""
    if df_stock is None or df_stock.empty:
        return pd.DataFrame()

    df = df_stock.copy()

    if "Note" not in df.columns:
        df["Note"] = ""

    note_series = df["Note"].astype(str).replace("nan", "")
    df["FirstOrderQty"] = note_series.str.extract(
        r"(?:^|\s)(\d+)(?:\s|$)", expand=False
    )
    df["NoteDescription"] = (
        note_series.str.replace(r"(?:^|\s)\d+(?:\s|$)", " ", regex=True)
        .str.strip()
        .replace("", None)
    )

    return df


def upload_replenishment(df_stock: pd.DataFrame, *, limit: int | None = None) -> int:
    """
    Upload replenishment docs with merge=True.
    Skip docs whose content hash matches sync_state/replenishment.
    Returns number of documents written.
    """
    df = prepare_replenishment_df(df_stock)
    if df.empty:
        return 0

    if limit is not None:
        df = df.head(limit)

    df = df.where(pd.notnull(df), None)
    db = get_firestore_client()
    known_hashes = load_hashes("replenishment")
    new_hashes = dict(known_hashes)
    written = 0

    for item in df.to_dict(orient="records"):
        product_code = str(item.get("ProductCode", "")).strip()
        if not product_code:
            product_code = str(item.get("Barcode", "")).strip()
        if not product_code:
            continue

        digest = content_hash(item)
        if new_hashes.get(product_code) == digest:
            continue

        db.collection("replenishment").document(product_code).set(item, merge=True)
        new_hashes[product_code] = digest
        written += 1
        
    if new_hashes != known_hashes:
        save_hashes("replenishment", new_hashes)

    return written    