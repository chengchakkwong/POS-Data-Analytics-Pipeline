"""Firebase client helpers."""

from __future__ import annotations

from pathlib import Path

import firebase_admin
from firebase_admin import credentials, firestore

from pos_pipeline.config import FIREBASE_KEY_PATH


def get_firebase_key_path() -> Path:
    return FIREBASE_KEY_PATH


def firebase_key_exists() -> bool:
    return FIREBASE_KEY_PATH.is_file()


def get_firestore_client():
    """Initialize Firebase app (once) and return Firestore client."""
    if not firebase_key_exists():
        raise FileNotFoundError(f"Firebase key not found: {FIREBASE_KEY_PATH}")

    if not firebase_admin._apps:
        cred = credentials.Certificate(str(FIREBASE_KEY_PATH))
        firebase_admin.initialize_app(cred)

    return firestore.client()