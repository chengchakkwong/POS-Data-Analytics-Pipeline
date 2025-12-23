import os
import time
import pandas as pd
from datetime import date, datetime, timedelta

class POSDataService:
    """
    【業務邏輯層】
    專門負責處理 POS 的數據邏輯（庫存、銷售、結構）。
    只專注於數據如何提取與加工。
    """
    def __init__(self, db_manager):
        self.db = db_manager
        self.today = date.today().strftime('%Y-%m-%d')
        
    def generate_data_dictionary(self, output_file="table_structure.txt"):
        print("🔍 正在生成數據字典...")
        sql = """
            SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH, IS_NULLABLE
            FROM INFORMATION_SCHEMA.COLUMNS
            ORDER BY TABLE_NAME, ORDINAL_POSITION
        """
        df = self.db.execute_query(sql)
        if df.empty: return

        with open(output_file, "w", encoding="utf-8") as f:
            current_table = ""
            for _, row in df.iterrows():
                if row['TABLE_NAME'] != current_table:
                    current_table = row['TABLE_NAME']
                    f.write(f"\n📦 dbo.{current_table}\n")
                
                max_len = f"({int(row['CHARACTER_MAXIMUM_LENGTH'])})" if pd.notnull(row['CHARACTER_MAXIMUM_LENGTH']) and row['CHARACTER_MAXIMUM_LENGTH'] > 0 else ""
                f.write(f"   ├─ {row['COLUMN_NAME']:<30} {row['DATA_TYPE']:<12} {max_len:<8} NULL={row['IS_NULLABLE']}\n")
        print(f"✅ 結構已存至 {output_file}")
 
    def get_stock_master_data(self):
        """
        提取完整的商品庫存與分類資訊。
        在此階段即進行數據清洗，確保產出的 CSV 是整潔且可直接使用的。
        """
        print("🚀 正在執行全量商品庫存關聯查詢...")
        
        sql = """
            SELECT 
                g.SID AS GoodsID, 
                g.ID AS ProductCode, 
                g.Barcode, 
                g.Name, 
                g.Note, 
                s.CurrStock, 
                g.RetailPrice, 
                g.LastInCost, 
                g.AvgCost,
                d.Name AS Category,          
                t1.Name AS InboundLocation,  
                t2.Name AS Supplier          
            FROM dbo.GoodsInfo g
            LEFT JOIN dbo.GoodsStock s ON g.SID = s.GoodsID AND s.ShopID = 1
            LEFT JOIN dbo.Dept d ON g.DeptID = d.SID
            LEFT JOIN dbo.ProductType1 t1 ON g.ProductType1ID = t1.SID
            LEFT JOIN dbo.ProductType2 t2 ON g.ProductType2ID = t2.SID
        """
        df = self.db.execute_query(sql)
        
        if not df.empty:
            # --- 在源頭清洗數據 (Clean at Source) ---
            # 1. 清洗產品名稱與備註中的換行符與前後空白
            # 這樣產出的 CSV 就不會再有行數錯亂的問題
            text_columns = ['Name', 'Note', 'Category', 'InboundLocation', 'Supplier']
            for col in text_columns:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.replace(r'[\n\r\t]+', ' ', regex=True).str.strip()

            # 確保輸出目錄存在
            if not os.path.exists('data/processed'):
                os.makedirs('data/processed')
            
            output_file = "data/processed/DetailGoodsStockToday.csv"
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"✅ 今日完整庫存清單已更新: {output_file}")
            
        return df
    
    def sync_daily_sales(self, cache_file="data/processed/vw_GoodsDailySales_cache.parquet"):
        """
        增量同步每日銷售數據。
        建議將快取檔案存放在 data/processed/ 文件夾中。
        """
        start_time = time.perf_counter()

        # 判斷快取是否存在
        if os.path.exists(cache_file):
            df_old = pd.read_parquet(cache_file)
            last_date = pd.to_datetime(df_old["rDate"].max())
            # 如果有快取，往前推一天進行增量同步（確保最後一天的數據完整性）
            sync_start = (last_date - timedelta(days=1)).strftime("%Y-%m-%d")
            print(f"📅 發現現有快取，最後日期為 {last_date.date()}，從 {sync_start} 開始增量同步...")
        else:
            df_old = pd.DataFrame()
            # 若無快取，直接設定為 2024-01-01
            sync_start = "2024-01-01"
            print(f"ℹ️ 無現有快取，將從初始設定日期 {sync_start} 開始全量同步...")

        # 這裡只使用 sync_start，不要再重新指定
        print(f"🔄 正在增量同步自 {sync_start} 起的銷售數據...")


        sql = """
            SELECT D.GoodsID, M.rDate, SUM(D.Quantity) AS TotalQty, SUM(D.FinalAmt) AS TotalAmt
            FROM dbo.SalesDetail AS D
            JOIN dbo.SalesMaster AS M ON D.SalesMasterID = M.SID
            WHERE CONVERT(date, CONVERT(varchar(8), M.rDate)) >= :start_date
            GROUP BY D.GoodsID, M.rDate
        """
        df_new = self.db.execute_query(sql, params={"start_date": sync_start})

        if df_new.empty:
            print("✅ 無新數據。")
            return df_old
        
        # 合併新舊數據並去重（以 GoodsID 和 rDate 為準，保留最新的紀錄）
        df_all = pd.concat([df_old, df_new], ignore_index=True).drop_duplicates(subset=["GoodsID", "rDate"], keep="last")
        
        # 儲存快取 
        try:
            df_all.to_parquet(cache_file, compression="snappy")
        except ImportError:
            print("❌ 儲存失敗：環境中缺少 Parquet 引擎 (pyarrow)。")
            print("💡 請在終端機執行：pip install pyarrow")
        except Exception as e:
            # 如果是 Snappy 不支援，嘗試 gzip
            print(f"⚠️ Snappy 壓縮失敗，嘗試 gzip: {e}")
            df_all.to_parquet(cache_file, compression="gzip")
        
        duration = time.perf_counter() - start_time
        print(f"💾 同步完成！共 {len(df_all):,} 筆，耗時 {duration:.2f} 秒")
        return df_all