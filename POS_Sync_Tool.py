import time
import logging
from db_utils import POSDatabaseManager
from pos_service import POSDataService
from firebase_service import FirebaseManager
from replenishment_service import prepare as prepare_replenishment
import logger_config  # 導入統一的日誌配置

# 使用統一的日誌配置
logger = logging.getLogger(__name__)
PRODUCT_TARGET_COLUMNS = [
    "ProductCode",
    "Barcode",
    "Name",
    "CurrStock",
    "RetailPrice",
    "Category",
    "Supplier",
]
SEPARATOR = "=" * 40


def initialize_services():
    """初始化資料庫、POS 服務與 Firebase 服務。"""
    logger.info("初始化資料庫連接和服務...")
    db_mgr = POSDatabaseManager(timeout=3)
    service = POSDataService(db_mgr)
    fb_mgr = FirebaseManager(key_path="serviceAccountKey.json")
    return db_mgr, service, fb_mgr


def fetch_stock_data(service):
    """從 POS 取得商品庫存主檔；失敗時回傳 None。"""
    try:
        return service.get_stock_master_data()
    except Exception as e:
        logger.error(f"❌ 抓取商品庫存主檔失敗: {e}", exc_info=True)
        return None


def select_upload_columns(df_stock):
    """過濾 products 目標欄位，並回傳可上傳資料。"""
    valid_cols = [col for col in PRODUCT_TARGET_COLUMNS if col in df_stock.columns]
    if len(valid_cols) < len(PRODUCT_TARGET_COLUMNS):
        missing = set(PRODUCT_TARGET_COLUMNS) - set(valid_cols)
        logger.warning(f"⚠️ 注意: SQL 查詢結果缺少以下欄位，將略過: {missing}")
    return df_stock[valid_cols]


def upload_to_firebase(fb_mgr, df_stock):
    """
    上傳 products 與 replenishment 到 Firebase。
    回傳 True 表示流程執行完成（包含部分跳過但非錯誤情境）。
    """
    if df_stock is None or df_stock.empty:
        logger.warning("⚠️ 查無資料，跳過上傳步驟。")
        return True

    try:
        df_products = select_upload_columns(df_stock)
        logger.info(f"準備上傳 {len(df_products)} 筆資料到 Firebase (products + replenishment)...")
        fb_mgr.upload_stock_data(df_products)

        df_repl = prepare_replenishment(df_stock)
        if df_repl.empty:
            logger.warning("⚠️ 補貨加工後無資料，跳過 replenishment 上傳。")
            logger.info("✅ Firebase（products）上傳成功！")
            return True
        

        logger.info(f"準備上傳 {len(df_repl)} 筆補貨建議資料到 replenishment...")
        fb_mgr.upload_replenishment_data(df_repl)
        df_repl.to_csv("data/processed/products_details_for_replenishment.csv", index=False , encoding='utf-8-sig')
        logger.info("✅ Firebase（products + replenishment）上傳成功！")
        return True
    except Exception as e:
        logger.error(f"❌ Firebase 上傳失敗: {e}", exc_info=True)
        return False

def main():
    """
    【執行層 / 指揮中心】
    """
    start_all = time.perf_counter()
    
    try:
        # 1. 初始化
        db_mgr, service, fb_mgr = initialize_services()

        # 2. 檢查連線
        if not db_mgr.check_connection():
            logger.error("⚠️ 無法連線至 POS 伺服器。")
            return

        logger.info(SEPARATOR)
        logger.info(f"🚀 數據同步任務啟動: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(SEPARATOR)

        # 3. [Extract] 從 SQL 抓取數據
        df_stock = fetch_stock_data(service)

        # 4. [Load] 上傳到 Firebase：同時寫入 products 與 replenishment
        upload_to_firebase(fb_mgr, df_stock)

        # 5. 統計成果
        logger.info("")
        logger.info(SEPARATOR)
        total_time = time.perf_counter() - start_all
        logger.info(f"✨ 任務全數完成！總耗時: {total_time:.2f} 秒")
        logger.info(SEPARATOR)
        
    except Exception as e:
        logger.error(f"❌ 程序執行失敗: {e}", exc_info=True)

if __name__ == "__main__":
    main()