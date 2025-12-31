import time
import logging
from db_utils import POSDatabaseManager
from pos_service import POSDataService
import logger_config  # 導入統一的日誌配置

# 使用統一的日誌配置
logger = logging.getLogger(__name__)

def main():
    """
    【執行層 / 指揮中心】
    """
    start_all = time.perf_counter()
    
    try:
        # 1. 初始化
        logger.info("初始化資料庫連接和服務...")
        db_mgr = POSDatabaseManager(timeout=3)
        service = POSDataService(db_mgr)

        # 2. 檢查連線
        if not db_mgr.check_connection():
            logger.error("⚠️ 無法連線至 POS 伺服器。")
            return

        logger.info("="*40)
        logger.info(f"🚀 數據同步任務啟動: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*40)

        # 3. 執行任務與命名優化
        
        # [Task A] 更新數據字典 (了解資料表結構用)
        try:
            service.generate_data_dictionary()
        except Exception as e:
            logger.error(f"❌ 生成數據字典失敗: {e}", exc_info=True)

        # [Task B] 抓取商品庫存主檔
        df_stock = None
        try:
            df_stock = service.get_stock_master_data()
        except Exception as e:
            logger.error(f"❌ 抓取商品庫存主檔失敗: {e}", exc_info=True)

        # [Task C] 增量同步銷售歷史
        df_sales = None
        try:
            df_sales = service.sync_daily_sales()
        except Exception as e:
            logger.error(f"❌ 增量同步銷售歷史失敗: {e}", exc_info=True)

        # 4. 統計成果
        logger.info("")
        logger.info("="*40)
        if df_stock is not None and not df_stock.empty:
            logger.info(f"📊 今日庫存清單: {len(df_stock):,} 筆")
        elif df_stock is not None:
            logger.warning("📊 今日庫存清單: 0 筆（無數據）")
            
        if df_sales is not None and not df_sales.empty:
            logger.info(f"📈 累計銷售紀錄: {len(df_sales):,} 筆")
        elif df_sales is not None:
            logger.warning("📈 累計銷售紀錄: 0 筆（無數據）")
            
        total_time = time.perf_counter() - start_all
        logger.info(f"✨ 任務全數完成！總耗時: {total_time:.2f} 秒")
        logger.info("="*40)
        
    except Exception as e:
        logger.error(f"❌ 程序執行失敗: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()