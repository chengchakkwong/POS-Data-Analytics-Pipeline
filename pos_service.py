import time
import logging
import pandas as pd
from datetime import date, timedelta
from pathlib import Path
import logger_config  # 導入統一的日誌配置

# 使用統一的日誌配置
logger = logging.getLogger(__name__)

class POSDataService:
    """
    【業務邏輯層】
    專門負責處理 POS 的數據邏輯（庫存、銷售、結構）。
    只專注於數據如何提取與加工。
    """
    def __init__(self, db_manager):
        self.db = db_manager
        self.today = date.today().strftime('%Y-%m-%d')
        self.output_dir = Path('data/processed')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_data_dictionary(self, filename="table_structure.txt"):
        """生成數據字典，輸出到 data/processed/"""
        logger.info("🔍 正在生成數據字典...")
        sql = """
            SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH, IS_NULLABLE
            FROM INFORMATION_SCHEMA.COLUMNS
            ORDER BY TABLE_NAME, ORDINAL_POSITION
        """
        df = self.db.execute_query(sql)
        if df.empty:
            return

        output_file = self.output_dir / filename
        with open(output_file, "w", encoding="utf-8") as f:
            current_table = ""
            for _, row in df.iterrows():
                if row['TABLE_NAME'] != current_table:
                    current_table = row['TABLE_NAME']
                    f.write(f"\n📦 dbo.{current_table}\n")
                
                max_len = f"({int(row['CHARACTER_MAXIMUM_LENGTH'])})" if pd.notnull(row['CHARACTER_MAXIMUM_LENGTH']) and row['CHARACTER_MAXIMUM_LENGTH'] > 0 else ""
                f.write(f"   ├─ {row['COLUMN_NAME']:<30} {row['DATA_TYPE']:<12} {max_len:<8} NULL={row['IS_NULLABLE']}\n")
        logger.info(f"✅ 結構已存至 {output_file}")
 
    def get_stock_master_data(self):
        """提取完整的商品庫存與分類資訊，輸出到 data/processed/"""
        logger.info("🚀 正在執行全量商品庫存關聯查詢...")
        
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
            # 清洗文本欄位：移除換行符與前後空白
            text_cols = ['Name', 'Note', 'Category', 'InboundLocation', 'Supplier']
            for col in text_cols:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.replace(r'[\n\r\t]+', ' ', regex=True).str.strip()

            output_file = self.output_dir / "DetailGoodsStockToday.csv"
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            logger.info(f"✅ 今日完整庫存清單已更新: {output_file}")
            
        return df
    

    def sync_daily_sales(self, cache_dirname="vw_GoodsDailySales_partitioned"):
        
        """增量同步每日銷售數據，並依年月分區儲存到資料夾"""
        start_time = time.perf_counter()
        
        # 注意：這裡的 cache_dir 變成了一個「資料夾路徑」，而不是 .parquet 檔案
        cache_dir = self.output_dir / cache_dirname 

        # ---------------------------------------------------------
        # 1. 判斷快取是否存在，找出最後同步日期
        # ---------------------------------------------------------
        if cache_dir.exists():
            # 【優化點 1：Columnar 的威力】
            # 不需要把幾十萬筆資料全部讀進來，我們「只讀取 rDate 這一欄」來找最大日期，瞬間完成！
            df_dates = pd.read_parquet(cache_dir, columns=["rDate"])
            last_date = pd.to_datetime(df_dates["rDate"].max())
            sync_start = (last_date - timedelta(days=1)).strftime("%Y-%m-%d")
            logger.info(f"📅 發現分區快取，最後日期為 {last_date.date()}，從 {sync_start} 增量同步...")
        else:
            sync_start = "2024-01-01"
            logger.info(f"ℹ️ 無現有快取，將從 {sync_start} 開始全量同步並建立分區...")

        logger.info(f"🔄 正在從資料庫獲取數據...")

        sql = """
            SELECT D.GoodsID, M.rDate, SUM(D.Quantity) AS TotalQty, SUM(D.FinalAmt) AS TotalAmt
            FROM dbo.SalesDetail AS D
            JOIN dbo.SalesMaster AS M ON D.SalesMasterID = M.SID
            WHERE CONVERT(date, CONVERT(varchar(8), M.rDate)) >= :start_date
            GROUP BY D.GoodsID, M.rDate
        """
        df_new = self.db.execute_query(sql, params={"start_date": sync_start})

        if df_new.empty:
            logger.info("✅ 無新數據。")
            return pd.DataFrame() # 依照你的需求，看要回傳空表還是怎樣處理
        
        # ---------------------------------------------------------
        # 2. 為新資料建立分區欄位 (year, month)
        # ---------------------------------------------------------
        df_new['rDate'] = pd.to_datetime(df_new['rDate'])
        df_new['year'] = df_new['rDate'].dt.year.astype(str)
        df_new['month'] = df_new['rDate'].dt.month.astype(str).str.zfill(2) # 補零變成 '01', '02'

        # ---------------------------------------------------------
        # 3. 找出這次更新「影響到」哪些分區，並讀取舊資料
        # ---------------------------------------------------------
        # 提取新資料牽涉到的 (年, 月) 組合
        affected_partitions = df_new[['year', 'month']].drop_duplicates()
        
        if cache_dir.exists():
            # 建立 PyArrow 可以看懂的 filter 格式
            # 例如：[[('year','==','2024'), ('month','==','03')], [('year','==','2024'), ('month','==','04')]]
            filters = [
                [('year', '==', row['year']), ('month', '==', row['month'])] 
                for _, row in affected_partitions.iterrows()
            ]
            logger.info(f"📂 只讀取受影響的分區進行合併: {filters}")
            
            # 【優化點 2：條件式讀取】只把「受影響的月份」舊資料讀出來，不用讀取沒變動的歷史月份！
            df_affected_old = pd.read_parquet(cache_dir, filters=filters)
        else:
            df_affected_old = pd.DataFrame()

        # ---------------------------------------------------------
        # 4. 合併新舊數據並去重 (範圍縮小到只有受影響的月份)
        # ---------------------------------------------------------
        df_combined = pd.concat([df_affected_old, df_new], ignore_index=True)
        df_combined = df_combined.drop_duplicates(subset=["GoodsID", "rDate"], keep="last")
        
        # ---------------------------------------------------------
        # 5. 寫回 Parquet (只覆寫受影響的分區)
        # ---------------------------------------------------------
        try:
            # 【優化點 3：自動分區覆寫魔法】
            # existing_data_behavior='delete_matching' 會自動把受影響的那個月份整個砍掉，
            # 替換成我們剛剛合併去重好的 df_combined。沒有變動的舊月份完全不會被動到！
            df_combined.to_parquet(
                cache_dir, 
                engine='pyarrow',
                partition_cols=['year', 'month'], 
                compression="snappy",
                existing_data_behavior='delete_matching' 
            )
        except Exception as e:
            logger.error(f"❌ 寫入分區失敗: {e}", exc_info=True)
            logger.error("💡 請確保有安裝: pip install pyarrow")
        
        duration = time.perf_counter() - start_time
        logger.info(f"💾 同步並分區完成！本次處理 {len(df_combined):,} 筆 (受影響分區)，耗時 {duration:.2f} 秒")
        
        return df_combined



