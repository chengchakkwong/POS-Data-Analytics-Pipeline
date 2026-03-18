import argparse
import logging
import os
from typing import Optional

import pandas as pd

from firebase_service import FirebaseManager


logger = logging.getLogger(__name__)


def _pick_target_stock_source_column(df: pd.DataFrame) -> str:
    """
    Source for Target_Stock: Base_Demand or existing Target_Stock (no decimals).
    Upload only ProductCode and Target_Stock.
    """
    if "Base_Demand" in df.columns:
        return "Base_Demand"
    if "Target_Stock" in df.columns:
        return "Target_Stock"
    raise KeyError("CSV must contain either 'Base_Demand' or 'Target_Stock' column.")


def build_upload_df(
    csv_path: str,
    *,
    goods_id_col: str = "GoodsID",
    product_code_col: str = "ProductCode",
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Map identifier to ProductCode
    if product_code_col in df.columns:
        df[product_code_col] = df[product_code_col].astype(str).str.strip()
    elif goods_id_col in df.columns:
        df[product_code_col] = df[goods_id_col].astype(str).str.strip()
    else:
        raise KeyError(f"CSV must contain either '{product_code_col}' or '{goods_id_col}'.")

    src_col = _pick_target_stock_source_column(df)
    target_stock = pd.to_numeric(df[src_col], errors="coerce").fillna(0)

    # No decimals: convert to integer units (rounded to nearest).
    df_out = pd.DataFrame(
        {
            "ProductCode": df[product_code_col],
            "Target_Stock": target_stock.round().astype(int),
        }
    )

    # Drop empty ProductCode rows
    df_out["ProductCode"] = df_out["ProductCode"].astype(str).str.strip()
    df_out = df_out[df_out["ProductCode"] != ""]

    return df_out


def upload_inventory_plan(
    csv_path: str,
    *,
    key_path: str,
    cache_file: Optional[str] = "data/sync_cache.json",
) -> None:
    df_upload = build_upload_df(csv_path)
    if df_upload.empty:
        logger.warning("⚠️ No rows to upload.")
        return

    fb_mgr = FirebaseManager(key_path=key_path, cache_file=cache_file)
    fb_mgr.upload_replenishment_data(df_upload)

    logger.info("✅ Upload complete.")
    logger.info(f"   - CSV: {csv_path}")
    logger.info(f"   - Rows: {len(df_upload)}")
    logger.info("   - Collection: replenishment")
    logger.info("   - Fields: ProductCode, Target_Stock")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload final_inventory_plan.csv to Firestore replenishment (ProductCode + Target_Stock only)."
    )
    parser.add_argument(
        "--csv",
        default=os.path.join("data", "insights", "final_inventory_plan.csv"),
        help="Path to final_inventory_plan.csv",
    )
    parser.add_argument(
        "--key",
        default="serviceAccountKey.json",
        help="Path to Firebase service account key JSON",
    )
    parser.add_argument(
        "--cache",
        default=os.path.join("data", "sync_cache.json"),
        help="Local cache file path for incremental uploads",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    upload_inventory_plan(args.csv, key_path=args.key, cache_file=args.cache)


if __name__ == "__main__":
    main()

