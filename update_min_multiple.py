"""
【獨立腳本】從入貨紀錄計算 guessed_min / guessed_multiple，並上傳至 Firestore replenishment。
可單獨排程執行，不需與 POS_Sync_Tool 同跑。
"""
import time
import logging
from db_utils import POSDatabaseManager
from pos_service import POSDataService
from firebase_service import FirebaseManager
from replenishment_service import compute_guessed_min_multiple
import logger_config  # 導入統一的日誌配置

logger = logging.getLogger(__name__)


def main(years=2):
    start_all = time.perf_counter()

    try:
        logger.info("初始化資料庫連接...")
        db_mgr = POSDatabaseManager(timeout=3)
        service = POSDataService(db_mgr)

        if not db_mgr.check_connection():
            logger.error("⚠️ 無法連線至 POS 伺服器。")
            return

        logger.info("=" * 40)
        logger.info(f"🚀 Min/Multiple 更新任務啟動: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 40)

        # [Extract] 入貨紀錄（MoveTypeID=1，最近 years 年）
        df_inbound = None
        try:
            df_inbound = service.get_inbound_movements_for_min_multiple(years=years)
        except Exception as e:
            logger.error(f"❌ 抓取入貨紀錄失敗: {e}", exc_info=True)
            return

        if df_inbound is None or df_inbound.empty:
            logger.warning("⚠️ 無入貨紀錄，跳過計算與上傳。")
            return

        # [Transform] 因數統計法計算 guessed_min / guessed_multiple
        df_min_mult = compute_guessed_min_multiple(df_inbound)
        if df_min_mult.empty:
            logger.warning("⚠️ 計算後無有效 Min/Multiple，跳過上傳。")
            return

        # [Load] 上傳至 Firestore replenishment（僅寫入 guessed_min、guessed_multiple，merge=True + hash 快取）
        try:
            fb_mgr = FirebaseManager(key_path='serviceAccountKey.json')
            logger.info(f"準備上傳 {len(df_min_mult)} 筆 guessed_min/guessed_multiple 到 replenishment...")
            fb_mgr.upload_guessed_min_multiple(df_min_mult)
        except Exception as e:
            logger.error(f"❌ Firebase 上傳失敗: {e}", exc_info=True)
            return

        total_time = time.perf_counter() - start_all
        logger.info("")
        logger.info("=" * 40)
        logger.info(f"✨ Min/Multiple 更新完成！總耗時: {total_time:.2f} 秒")
        logger.info("=" * 40)

    except Exception as e:
        logger.error(f"❌ 程序執行失敗: {e}", exc_info=True)


if __name__ == "__main__":
    main(years=2)
