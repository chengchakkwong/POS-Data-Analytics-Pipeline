"""Firestore-backed sync hashes for incremental uploads."""

from __future__ import annotations

import hashlib
from typing import Any

from pos_pipeline.delivery_to_firebase.client import get_firestore_client

SYNC_STATE_COLLECTION = "sync_state"


def content_hash(item: dict[str, Any], *, ignore_keys: set[str] | None = None) -> str:
    """Stable MD5 over sorted fields; ignore volatile keys like AvgCost by default."""
    ignored = ignore_keys or {"AvgCost"}
    parts = [
        f"{key}={item.get(key)}"
        for key in sorted(item.keys())
        if key not in ignored
    ]
    return hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()


def load_hashes(state_name: str) -> dict[str, str]:
    """Load ProductCode -> hash map from sync_state/{state_name}."""
    db = get_firestore_client()
    snap = db.collection(SYNC_STATE_COLLECTION).document(state_name).get()
    if not snap.exists:
        return {}
    data = snap.to_dict() or {}
    hashes = data.get("hashes") or {}
    return {str(k): str(v) for k, v in hashes.items()}


def save_hashes(state_name: str, hashes: dict[str, str]) -> None:
    """Overwrite sync_state/{state_name}.hashes with the given map."""
    db = get_firestore_client()
    db.collection(SYNC_STATE_COLLECTION).document(state_name).set(
        {"hashes": hashes},
        merge=True,
    )


def load_last_sid(state_name: str = "inbound_movements") -> int:
    """Load the last processed SID from sync_state/{state_name}."""
    db = get_firestore_client()
    snap = db.collection(SYNC_STATE_COLLECTION).document(state_name).get()

    if not snap.exists:
        return 0

    data = snap.to_dict() or {}
    raw_value = data.get("lastSid", 0)

    try:
        last_sid = int(raw_value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid lastSid in sync_state/{state_name}: {raw_value!r}"
        ) from error

    if last_sid < 0:
        raise ValueError(
            f"lastSid cannot be negative: {last_sid}"
        )

    return last_sid


def save_last_sid(
    last_sid: int,
    state_name: str = "inbound_movements",
) -> None:
    """Save the last processed SID after successful uploads."""
    if last_sid < 0:
        raise ValueError(
            f"lastSid cannot be negative: {last_sid}"
        )

    db = get_firestore_client()
    db.collection(SYNC_STATE_COLLECTION).document(state_name).set(
        {"lastSid": int(last_sid)},
        merge=True,
    )