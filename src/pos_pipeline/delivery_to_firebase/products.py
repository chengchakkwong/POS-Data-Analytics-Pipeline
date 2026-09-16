"""Upload stock rows to Firestore products collection."""

from __future__ import annotations

import pandas as pd

from pos_pipeline.delivery_to_firebase.client import get_firestore_client

from pos_pipeline.delivery_to_firebase.sync_state import (
    content_hash,
    load_hashes,
    save_hashes,
)

PRODUCT_COLUMNS = [
    "ProductCode",
    "Barcode",
    "Name",
    "CurrStock",
    "RetailPrice",
    "Category",
    "Supplier",
]


def prepare_products_df(df_stock: pd.DataFrame) -> pd.DataFrame:
    """Keep only columns used by the products collection."""
    cols = [c for c in PRODUCT_COLUMNS if c in df_stock.columns]
    return df_stock[cols].copy()


def upload_products(df_stock: pd.DataFrame, *, limit: int | None = None) -> int:
    """
    Upload products with merge=True.
    Skip docs whose content hash matches sync_state/products.
    Returns number of documents written.
    """
    df = prepare_products_df(df_stock)
    if df.empty:
        return 0

    if limit is not None:
        df = df.head(limit)

    df = df.where(pd.notnull(df), None)
    db = get_firestore_client()
    known_hashes = load_hashes("products")
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

        db.collection("products").document(product_code).set(item, merge=True)
        new_hashes[product_code] = digest
        written += 1

    if new_hashes != known_hashes:
        save_hashes("products", new_hashes)

    return written