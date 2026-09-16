"""Upload inbound movements to Firestore."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from pos_pipeline.delivery_to_firebase.client import get_firestore_client
from pos_pipeline.delivery_to_firebase.sync_state import save_last_sid

REQUIRED_COLUMNS = ["SID", "GoodsNo", "ChQty", "BillDate"]


def upload_inbound_movements(df: pd.DataFrame) -> int:
    """Upload new inbound movements and advance the SID watermark."""
    if df is None or df.empty:
        return 0

    if not all(column in df.columns for column in REQUIRED_COLUMNS):
        return 0

    db = get_firestore_client()
    records = df.where(pd.notnull(df), None).to_dict(orient="records")
    synced_at = datetime.now(timezone.utc).isoformat()

    batch = db.batch()
    batch_count = 0
    written = 0
    uploaded_sids: list[int] = []

    for item in records:
        sid = item.get("SID")
        if sid is None:
            continue

        sid = int(sid)
        payload = {
            "SID": sid,
            "BillDate": item.get("BillDate"),
            "GoodsNo": item.get("GoodsNo"),
            "Barcode": item.get("Barcode"),
            "GoodsName1": item.get("GoodsName1"),
            "OriQty": item.get("OriQty"),
            "ChQty": item.get("ChQty"),
            "NewQty": item.get("NewQty"),
            "ProductType2Name1": item.get("ProductType2Name1"),
            "invNo": item.get("invNo"),
            "Note": item.get("Note"),
            "syncedAt": synced_at,
        }

        doc_ref = db.collection("inbound_movements").document(str(sid))
        batch.set(doc_ref, payload)

        uploaded_sids.append(sid)
        batch_count += 1
        written += 1

        if batch_count >= 400:
            batch.commit()
            batch = db.batch()
            batch_count = 0

    if batch_count:
        batch.commit()

    if uploaded_sids:
        save_last_sid(max(uploaded_sids))

    return written