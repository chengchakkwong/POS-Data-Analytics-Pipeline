import time
import os
from db_utils import POSDatabaseManager
from pos_service import POSDataService

def main():
    """
    【執行層 / 指揮中心】
    """
    start_all = time.perf_counter()
    
    # 1. 初始化
    db_mgr = POSDatabaseManager(timeout=3)
    service = POSDataService(db_mgr)

    # 2. 檢查連線
    if not db_mgr.check_connection():
        print("⚠️ 無法連線至 POS 伺服器。")
        return

    print("\n" + "="*40)
    print(f"🚀 數據同步任務啟動: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*40)

    # 3. 執行任務與命名優化
    
    # [Task A] 更新數據字典 (了解資料表結構用)
    service.generate_data_dictionary()
    
    # [Task B] 抓取商品庫存主檔
    # 推薦命名: df_stock (簡潔) 或 goods_master (強調這是一份主表)
    df_stock = service.get_stock_master_data()
    
    # [Task C] 增量同步銷售歷史
    # 推薦命名: df_sales (簡潔) 或 sales_history (強調這是歷史累積數據)
    df_sales = service.sync_daily_sales()

    # 4. 統計成果
    print("\n" + "="*40)
    if df_stock is not None:
        print(f"📊 今日庫存清單: {len(df_stock):,} 筆")
    if df_sales is not None:
        print(f"📈 累計銷售紀錄: {len(df_sales):,} 筆")
        
    total_time = time.perf_counter() - start_all
    print(f"✨ 任務全數完成！總耗時: {total_time:.2f} 秒")
    print("="*40)

if __name__ == "__main__":
    main()