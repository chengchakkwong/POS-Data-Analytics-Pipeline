import time
import logging
from db_utils import POSDatabaseManager
from pos_service import POSDataService
from firebase_service import FirebaseManager # 記得確保這個檔案在旁邊
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
        fb_mgr = FirebaseManager(key_path='serviceAccountKey.json')

        # 2. 檢查連線
        if not db_mgr.check_connection():
            logger.error("⚠️ 無法連線至 POS 伺服器。")
            return

        logger.info("="*40)
        logger.info(f"🚀 數據同步任務啟動: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*40)

        # 3. 執行任務與命名優化

        # [Extract] 從 SQL 抓取數據
        df_stock = None
        try:
            df_stock = service.get_stock_master_data()
        except Exception as e:
            logger.error(f"❌ 抓取商品庫存主檔失敗: {e}", exc_info=True)

        # [Load] 上傳到 Firebase
        # 4. [Load] 上傳到 Firebase
        if df_stock is not None and not df_stock.empty:
            try:
                # 定義目標欄位
                target_cols = ['ProductCode', 'Barcode', 'Name', 'CurrStock', 'RetailPrice', 'Category', 'Supplier']
                
                # 檢查 DataFrame 中實際存在的欄位 (避免因 SQL 沒抓到某些欄位而報錯)
                valid_cols = [col for col in target_cols if col in df_stock.columns]
                
                # 顯示篩選資訊
                if len(valid_cols) < len(target_cols):
                    missing = set(target_cols) - set(valid_cols)
                    logger.warning(f"⚠️ 注意: SQL 查詢結果缺少以下欄位，將略過: {missing}")
                
                # 執行篩選
                df_final = df_stock[valid_cols]
                
                logger.info(f"準備上傳 {len(df_final)} 筆資料到 Firebase...")
                
                # 這行就是關鍵！把 df 直接丟給上傳器
                fb_mgr.upload_stock_data(df_final)
                
                logger.info("✅ Firebase 上傳成功！")
            except Exception as e:
                logger.error(f"❌ Firebase 上傳失敗: {e}", exc_info=True)
        else:
            logger.warning("⚠️ 查無資料，跳過上傳步驟。")



        # 4. 統計成果
        logger.info("")
        logger.info("="*40)            
        total_time = time.perf_counter() - start_all
        logger.info(f"✨ 任務全數完成！總耗時: {total_time:.2f} 秒")
        logger.info("="*40)
        
    except Exception as e:
        logger.error(f"❌ 程序執行失敗: {e}", exc_info=True)

if __name__ == "__main__":
    main()