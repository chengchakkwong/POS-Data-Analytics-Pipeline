"""Daily extract + Firestore sync job."""

from pos_pipeline.config import PROJECT_ROOT
from pos_pipeline.database.connection import check_connection
from pos_pipeline.delivery_to_firebase.products import upload_products
from pos_pipeline.delivery_to_firebase.replenishment import upload_replenishment

from pos_pipeline.delivery_to_firebase.inbound import upload_inbound_movements
from pos_pipeline.delivery_to_firebase.sync_state import load_last_sid
from pos_pipeline.extraction.inbound import fetch_new_inbound_movements

def run() -> int:
    print(f"[job:daily] project root: {PROJECT_ROOT}")

    if not check_connection():
        print("[job:daily] ERROR: cannot connect to POS database")
        return 1

    print("[job:daily] database connection OK")

    from pos_pipeline.extraction.stock import fetch_stock_master, save_stock_master

    df = fetch_stock_master()
    print(f"[job:daily] stock master rows: {len(df)}")
    if df.empty:
        print("[job:daily] ERROR: stock master is empty")
        return 1

    output = save_stock_master(df)
    print(f"[job:daily] saved: {output}")



    products_written = upload_products(df)
    print(f"[job:daily] products uploaded: {products_written}")

    replenishment_written = upload_replenishment(df)
    print(f"[job:daily] replenishment uploaded: {replenishment_written}")

    last_sid = load_last_sid()
    inbound_df = fetch_new_inbound_movements(last_sid=last_sid)
    print(f"[job:daily] inbound rows: {len(inbound_df)}")

    inbound_written = upload_inbound_movements(inbound_df)
    print(f"[job:daily] inbound uploaded: {inbound_written}")
    return 0