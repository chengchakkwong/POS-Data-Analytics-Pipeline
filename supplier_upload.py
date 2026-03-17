"""
【獨立腳本】從 POS 取得供應商名單（get_supplier_info），上傳至 Firestore collection「supplier」。
可單獨排程或手動執行。
"""
import time
import logging
from db_utils import POSDatabaseManager
from pos_service import POSDataService
from firebase_service import FirebaseManager
import logger_config  # 導入統一的日誌配置

logger = logging.getLogger(__name__)


def main():
    start_all = time.perf_counter()
    try:
        logger.info("初始化資料庫連接...")
        db_mgr = POSDatabaseManager(timeout=3)
        service = POSDataService(db_mgr)

        if not db_mgr.check_connection():
            logger.error("⚠️ 無法連線至 POS 伺服器。")
            return

        logger.info("=" * 40)
        logger.info("🚀 供應商名單上傳任務啟動")
        logger.info("=" * 40)

        # [Extract] 取得供應商名單
        df_supplier = service.get_supplier_info()
        if df_supplier is None or df_supplier.empty:
            logger.warning("⚠️ 無供應商資料，跳過上傳。")
            return

        # [Load] 上傳至 Firestore collection「supplier」
        try:
            fb_mgr = FirebaseManager(key_path='serviceAccountKey.json')
            fb_mgr.upload_supplier_data(df_supplier)
        except Exception as e:
            logger.error(f"❌ Firebase 上傳失敗: {e}", exc_info=True)
            return

        total_time = time.perf_counter() - start_all
        logger.info("")
        logger.info("=" * 40)
        logger.info(f"✨ 供應商上傳完成！總耗時: {total_time:.2f} 秒")
        logger.info("=" * 40)
    except Exception as e:
        logger.error(f"❌ 程序執行失敗: {e}", exc_info=True)


if __name__ == "__main__":
    main()
