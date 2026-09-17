"""
demo_firebase_payload.py

Offline simulation of the Firestore "Load" stage, with NO network access.

It reads the offline demo analytics output (demo_output/abc_xyz_analysis.csv),
builds the same document payloads that firebase_service.py would upload, and
writes them to local JSON files instead of Firestore. It also reproduces the
incremental MD5 de-duplication so reviewers can see how unchanged documents are
skipped to save write quota.

This script intentionally does NOT import FirebaseManager (which would try to
connect to Firestore) and does NOT modify any production code.

離線模擬 Firestore「Load」階段，完全不連網路。讀取離線 demo 分析結果，組出與
firebase_service.py 相同的文件 payload，輸出成本地 JSON，並重現增量 MD5 去重邏輯。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from logger_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Hash helpers — formulas aligned with firebase_service.py
# (FirebaseManager._generate_hash / _generate_classification_hash)
# ---------------------------------------------------------------------------


def hash_full_record(item: dict) -> str:
    """Mirror FirebaseManager._generate_hash: all fields except AvgCost."""
    parts = []
    for key in sorted(item.keys()):
        if key == "AvgCost":
            continue
        parts.append(f"{key}={item.get(key)}")
    return hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()


def hash_classification(item: dict) -> str:
    """Mirror FirebaseManager._generate_classification_hash."""
    unique_str = (
        f"{item.get('ProductCode')}"
        f"{item.get('ABC_Class')}"
        f"{item.get('XYZ_Class')}"
        f"{item.get('note')}"
    )
    return hashlib.md5(unique_str.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Payload builders
# ---------------------------------------------------------------------------


def _clean_str(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _clean_code(value) -> str:
    """Clean an identifier and drop the trailing '.0' pandas adds to numeric codes."""
    s = _clean_str(value)
    if s.endswith(".0") and s[:-2].isdigit():
        return s[:-2]
    return s


def _to_number(value) -> float:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return 0
        return round(float(value), 2)
    except (TypeError, ValueError):
        return 0


def build_products(df: pd.DataFrame) -> dict[str, dict]:
    """products collection: stock master for price & stock lookup."""
    docs: dict[str, dict] = {}
    for _, row in df.iterrows():
        product_code = _clean_code(row.get("ProductCode")) or _clean_code(row.get("Barcode"))
        if not product_code:
            continue
        docs[product_code] = {
            "ProductCode": product_code,
            "Barcode": _clean_code(row.get("Barcode")),
            "Name": _clean_str(row.get("Name")),
            "CurrStock": int(_to_number(row.get("CurrStock"))),
            "RetailPrice": _to_number(row.get("RetailPrice")),
        }
    return docs


def build_classification(df: pd.DataFrame) -> dict[str, dict]:
    """classification payload: ABC/XYZ labels for the replenishment board."""
    docs: dict[str, dict] = {}
    for _, row in df.iterrows():
        product_code = _clean_code(row.get("ProductCode"))
        if not product_code:
            continue
        docs[product_code] = {
            "ABC_Class": _clean_str(row.get("ABC_Class")),
            "XYZ_Class": _clean_str(row.get("XYZ_Class")),
            "note": _clean_str(row.get("Note")),
        }
    return docs


# ---------------------------------------------------------------------------
# Incremental de-dup (mirrors firebase_service.py cache behavior)
# ---------------------------------------------------------------------------


def diff_against_cache(
    docs: dict[str, dict],
    cache: dict[str, str],
    cache_prefix: str,
    hash_fn,
) -> tuple[int, int, dict[str, str]]:
    """Return (would_upload, skipped, updated_cache) without any network call."""
    would_upload = 0
    skipped = 0
    new_cache = dict(cache)
    for doc_id, payload in docs.items():
        hash_input = {**payload, "ProductCode": doc_id} if cache_prefix == "class" else payload
        current_hash = hash_fn(hash_input)
        cache_key = f"{cache_prefix}:{doc_id}"
        if cache.get(cache_key) == current_hash:
            skipped += 1
            continue
        would_upload += 1
        new_cache[cache_key] = current_hash
    return would_upload, skipped, new_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline simulation of Firestore upload using demo analytics output."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("demo_output/abc_xyz_analysis.csv"),
        help="Analytics CSV produced by demo_pipeline.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("demo_output/firebase_payload"),
        help="Directory for simulated Firestore payloads.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise SystemExit(
            f"Missing analytics input: {args.input}\n"
            "Run `python demo_pipeline.py` first to generate it."
        )

    df = pd.read_csv(args.input, encoding="utf-8-sig")
    logger.info("Loaded %d analytics rows from %s", len(df), args.input)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.output_dir / ".sync_cache.json"
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        cache = {}

    products = build_products(df)
    classification = build_classification(df)

    prod_upload, prod_skip, cache = diff_against_cache(
        products, cache, "prod", hash_full_record
    )
    class_upload, class_skip, cache = diff_against_cache(
        classification, cache, "class", hash_classification
    )

    (args.output_dir / "products.json").write_text(
        json.dumps(products, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "classification.json").write_text(
        json.dumps(classification, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    cache_path.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== Simulated Firestore Upload (offline, no network) ===")
    print(f"products       : would upload {prod_upload:>4} | skipped {prod_skip:>4} (unchanged)")
    print(f"classification : would upload {class_upload:>4} | skipped {class_skip:>4} (unchanged)")
    print(f"\nPayloads written to: {args.output_dir}/")
    print("  - products.json")
    print("  - classification.json")
    print("Re-run this command to see all documents skipped via MD5 de-dup.")


if __name__ == "__main__":
    main()
