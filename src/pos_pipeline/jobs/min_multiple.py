"""Guessed min / multiple job: inbound history → Firestore replenishment."""

from __future__ import annotations

import time

from pos_pipeline.analysis.min_multiple import compute_guessed_min_multiple
from pos_pipeline.config import PROJECT_ROOT
from pos_pipeline.database.connection import check_connection
from pos_pipeline.delivery_to_firebase.min_multiple import (
    upload_guessed_min_multiple,
)
from pos_pipeline.extraction.inbound import fetch_inbound_for_min_multiple


def run(years: int = 2) -> int:
    """Extract inbound history, guess MOQ/multiple, merge to Firestore.

    Returns 0 on success (including empty result sets). Connection / extract /
    compute / upload failures return 1.
    """
    job_started = time.perf_counter()
    print(f"[job:min-multiple] project root: {PROJECT_ROOT}")
    print(f"[job:min-multiple] years: {years}")

    if not check_connection():
        print("[job:min-multiple] ERROR: cannot connect to POS database")
        return 1

    print("[job:min-multiple] database connection OK")

    try:
        inbound_df = fetch_inbound_for_min_multiple(years=years)
        print(f"[job:min-multiple] inbound rows: {len(inbound_df)}")
        if inbound_df.empty:
            print("[job:min-multiple] no inbound rows; skip upload")
            return 0

        guessed_df = compute_guessed_min_multiple(inbound_df)
        print(f"[job:min-multiple] guessed SKUs: {len(guessed_df)}")
        if guessed_df.empty:
            print("[job:min-multiple] no guessed min/multiple; skip upload")
            return 0

        written = upload_guessed_min_multiple(guessed_df)
        print(f"[job:min-multiple] replenishment docs written: {written}")
        print(
            f"[job:min-multiple] elapsed: {time.perf_counter() - job_started:.1f}s"
        )
        return 0
    except Exception as error:
        print(f"[job:min-multiple] ERROR: {error}")
        return 1
