import pandas as pd
import numpy as np
import os
import time
import warnings
import logging
import multiprocessing
from joblib import Parallel, delayed

# --- 系統配置與效能設定 ---
TEST_MODE = False      # 設為 False 處理全量商品
SAMPLE_SIZE = 50       # (測試模式才有效)
# 💡 動態取得 CPU 核心數，留下 1~2 顆給系統背景運作，確保電腦不卡死
CPU_WORKERS = max(1, multiprocessing.cpu_count() - 2) 
# ------------------------

# 嘗試導入模型 (統一強制使用 CPU 以確保多進程穩定性)
try:
    from neuralprophet import NeuralProphet
    import torch
    HAS_NEURAL_PROPHET = True
except ImportError:
    from prophet import Prophet
    HAS_NEURAL_PROPHET = False

# 忽略日誌干擾
warnings.filterwarnings('ignore')
logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('neuralprophet').setLevel(logging.ERROR)
logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)

# --- 1. 數據分類與預處理 ---

def calculate_category_seasonal_indices(sales_df, stock_df):
    """計算類別層級的季節性係數 (用於輔助數據稀疏的 C/Z 類商品)"""
    sales_df = sales_df.copy()
    sales_df['rDate'] = pd.to_datetime(sales_df['rDate'])
    df_with_cat = sales_df.merge(stock_df[['GoodsID', 'Category']], on='GoodsID', how='left')
    cat_monthly = df_with_cat.groupby(['Category', pd.Grouper(key='rDate', freq='MS')])['TotalQty'].sum().reset_index()
    cat_avg = cat_monthly.groupby('Category')['TotalQty'].mean().rename('Cat_Avg_Qty').reset_index()
    cat_indices = cat_monthly.merge(cat_avg, on='Category')
    # 避免除以 0
    cat_indices['Cat_Avg_Qty'] = cat_indices['Cat_Avg_Qty'].replace(0, 1) 
    cat_indices['Seasonal_Index'] = cat_indices['TotalQty'] / cat_indices['Cat_Avg_Qty']
    cat_indices['Month'] = cat_indices['rDate'].dt.month
    index_map = cat_indices.groupby(['Category', 'Month'])['Seasonal_Index'].mean().to_dict()
    return index_map

def perform_abc_xyz_analysis(stock_df, sales_df):
    """執行 ABC-XYZ 矩陣分析 (價值與波動)"""
    sales_df['rDate'] = pd.to_datetime(sales_df['rDate'])
    last_date = sales_df['rDate'].max()
    start_12m = last_date - pd.DateOffset(months=12)
    
    df_12m = sales_df[sales_df['rDate'] >= start_12m].copy()
    summary_12m = df_12m.groupby('GoodsID').agg({'TotalQty': 'sum', 'TotalAmt': 'sum'}).reset_index()
    
    merged = pd.merge(stock_df, summary_12m, on='GoodsID', how='left').fillna(0)
    # 確保有 LastInCost 欄位，沒有的話設為 0 免得報錯
    if 'LastInCost' not in merged.columns:
        merged['LastInCost'] = 0
    merged['TotalProfit'] = merged['TotalAmt'] - (merged['LastInCost'] * merged['TotalQty'])
    merged = merged.sort_values('TotalProfit', ascending=False)
    
    total_prof = merged['TotalProfit'].sum()
    if total_prof > 0:
        merged['ProfitRatio'] = merged['TotalProfit'].cumsum() / total_prof
    else:
        merged['ProfitRatio'] = 1.0
        
    merged['ABC_Class'] = np.select([(merged['ProfitRatio'] <= 0.7), (merged['ProfitRatio'] <= 0.9)], ['A', 'B'], default='C')

    # XYZ 分析：月均銷量與變異係數
    monthly_matrix = sales_df.groupby(['GoodsID', pd.Grouper(key='rDate', freq='MS')])['TotalQty'].sum().unstack(fill_value=0)
    stats = pd.DataFrame(index=monthly_matrix.index)
    stats['Mean_Monthly'] = monthly_matrix.mean(axis=1)
    stats['CV'] = np.where(stats['Mean_Monthly'] > 0, monthly_matrix.std(axis=1) / stats['Mean_Monthly'], 9.99)
    stats['XYZ_Class'] = np.select([(stats['CV'] <= 0.5), (stats['CV'] <= 1.0)], ['X', 'Y'], default='Z')
    
    first_sale = sales_df.groupby('GoodsID')['rDate'].min().reset_index()
    first_sale['Month_Age'] = (last_date.year - first_sale['rDate'].dt.year) * 12 + (last_date.month - first_sale['rDate'].dt.month)
    
    analysis_df = merged.merge(stats[['CV', 'XYZ_Class', 'Mean_Monthly']], left_on='GoodsID', right_index=True, how='left')
    analysis_df = analysis_df.merge(first_sale[['GoodsID', 'Month_Age']], on='GoodsID', how='left').fillna({
        'Month_Age': 99, 'XYZ_Class': 'Z', 'CV': 9.99, 'Mean_Monthly': 0
    })
    
    # 銷售不到 4 個月標記為新品
    is_new = (analysis_df['Month_Age'] < 4)
    analysis_df.loc[is_new, 'ABC_Class'], analysis_df.loc[is_new, 'XYZ_Class'] = 'New', 'New'
    
    return analysis_df

# --- 2. 預測核心 (純 CPU 穩定極速版) ---

def run_single_sku_forecast(item, item_sales, next_month, cat_index_map):
    """單一 SKU 預測單元 (接收專屬的小 DataFrame)"""
    gid = item['GoodsID']
    # 👇 新增這行：從 stock 表抓取 ProductCode，如果沒有則顯示 Unknown
    p_code = item.get('ProductCode', 'Unknown') 
    
    abc, xyz, cv = item['ABC_Class'], item['XYZ_Class'], item['CV']
    cat, total_qty_year = item.get('Category', 'Unknown'), item.get('TotalQty', 0)
    
    # 如果完全沒有歷史紀錄，建立空表避免報錯
    if item_sales.empty:
        item_sales = pd.DataFrame(columns=['rDate', 'TotalQty'])
    else:
        item_sales = item_sales.sort_values('rDate')
        
    mean_qty = item['Mean_Monthly']
    base_pred = 0

    try:
        # [策略 1] 新品：抓取最近 4 週的平均銷售動能
        if abc == 'New':
            if not item_sales.empty:
                recent = item_sales[item_sales['rDate'] >= (item_sales['rDate'].max() - pd.Timedelta(weeks=4))]
                base_pred = (recent['TotalQty'].sum() / 4) * 4 if not recent.empty else mean_qty
            else:
                base_pred = mean_qty
        
        # [策略 2] X/Y 類：具有規律與季節性，交給 AI 模型捕捉
        elif xyz in ['X', 'Y']: 
            m_df = item_sales.groupby(pd.Grouper(key='rDate', freq='MS'))['TotalQty'].sum().reset_index()
            m_df.columns = ['ds', 'y']
            
            # 模型需要至少一年的資料 (12筆) 才能學會年度季節性
            if len(m_df) >= 12:
                if HAS_NEURAL_PROPHET:
                    m = NeuralProphet(
                        yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
                        accelerator="cpu", epochs=30, batch_size=None,
                        trainer_config={"logger": False, "enable_checkpointing": False, "enable_progress_bar": False}
                    )
                    m.fit(m_df, freq="MS", progress=None)
                    future = m.make_future_dataframe(m_df, periods=1)
                    forecast = m.predict(future)
                    base_pred = max(0, forecast.iloc[-1]['yhat1'])
                else:
                    m = Prophet(yearly_seasonality=True, uncertainty_samples=0)
                    m.fit(m_df)
                    base_pred = max(0, m.predict(m.make_future_dataframe(periods=1, freq='MS')).iloc[-1]['yhat'])
            else:
                # 資料不足 12 筆，退回移動平均
                base_pred = m_df['y'].tail(3).mean() if not m_df.empty else mean_qty
                
        # [策略 3] Z 類：長尾死水商品，取過去半年中位數剔除極端值
        else:
            if not item_sales.empty:
                base_pred = item_sales.groupby(pd.Grouper(key='rDate', freq='MS'))['TotalQty'].sum().tail(6).median()
            else:
                base_pred = mean_qty
                
    except Exception as e:
        base_pred = mean_qty

    # --- 防爆門限與補貨邏輯 (Reality Check) ---
    
    # 1. 預測上限防禦 (非新品預測值不可超過月均量的 3 倍)
    if abc != 'New' and mean_qty > 0:
        base_pred = min(base_pred, mean_qty * 3)
        
    # 2. A/B 類主力品保底防禦 (預測出 0 時，強制回退平均值)
    if base_pred <= 0 and abc in ['A', 'B'] and mean_qty > 0:
        base_pred = mean_qty

    # 3. 類別季節性加成 (僅針對新品或 Z 類冷門品)
    boost = min(2.0, cat_index_map.get((cat, next_month), 1.0)) if (abc == 'New' or xyz == 'Z') else 1.0
    final_demand = base_pred * max(1.0, boost)
    
    # 4. 安全庫存係數 (根據 CV 變異係數動態調整)
    safety_ratio = min(0.5 if abc == 'A' else 0.3, cv * 0.5)
    target_stock = final_demand + (final_demand * safety_ratio)
    
    # 5. 最終庫存天花板
    if abc != 'New' and mean_qty > 0:
        target_stock_limit = max(mean_qty * 4, 2)
        target_stock = min(target_stock, target_stock_limit)
        
    if abc != 'New' and total_qty_year > 0:
        target_stock = min(target_stock, total_qty_year)

    return {
        # 👇 這裡把 'GoodsID': gid 改成 'ProductCode': p_code
        'ProductCode': p_code, 
        # 💡 小提醒：如果你的商品名稱欄位叫 GoodsName1，這裡也可以順便改成 item.get('GoodsName1', 'Unknown')
        'Name': item.get('GoodsName1', item.get('Name', 'Unknown')), 
        'ABC_XYZ': f"{abc}{xyz}",
        'Base_Demand': round(base_pred, 2), 'Final_Demand': round(final_demand, 2), 
        'Target_Stock': round(target_stock, 2), 'CurrStock': item.get('CurrStock', 0)
    }

# --- 3. 主程序執行邏輯 ---

def main():
    input_stock = "data/processed/DetailGoodsStockToday.csv"
    input_sales = "data/processed/vw_GoodsDailySales_partitioned"
    output_path = "data/insights/final_inventory_plan.csv"
    
    print(f"🚀 啟動優化版智慧補貨系統 (運算引擎: CPU 多核心模式)")
    print(f"🧵 配置平行進程數: {CPU_WORKERS} / {multiprocessing.cpu_count()}")
        
    if TEST_MODE:
        print(f"⚠️ 警告：目前處於測試模式，僅處理前 {SAMPLE_SIZE} 個商品。")
    else:
        print(f"🔥 全量模式啟動！正在分析全店數據...")
    
    # 確保資料夾存在，否則創建
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        df_stock = pd.read_csv(input_stock)
        # 根據副檔名或路徑決定讀取方式
        if input_sales.endswith('.parquet') or os.path.isdir(input_sales):
            df_sales = pd.read_parquet(input_sales)
        else:
            df_sales = pd.read_csv(input_sales)
    except FileNotFoundError:
        print("❌ 錯誤: 找不到數據源。請確認 data/processed/ 路線下有正確檔案。")
        return

    print("📊 執行 ABC-XYZ 分類與類別加成分析...")
    analysis_df = perform_abc_xyz_analysis(df_stock, df_sales)
    cat_index_map = calculate_category_seasonal_indices(df_sales, df_stock)
    
    # 篩選出有價值的商品進行預測 (排除常年不動的 C 類)
    target_skus = analysis_df[analysis_df['ABC_Class'].isin(['A', 'B', 'New'])]
    
    if TEST_MODE:
        target_skus = target_skus.head(SAMPLE_SIZE)
        
    next_month = (pd.to_datetime(df_sales['rDate']).max() + pd.DateOffset(months=1)).month
    
    print("📦 正在解決記憶體問題：預先打包各 SKU 的專屬歷史資料...")
    # [核心優化] 避免序列化大表，提前按 SKU 切碎資料
    sales_by_sku = {gid: df for gid, df in df_sales.groupby('GoodsID')}

    print(f"🔮 開始高效並行預測 {len(target_skus)} 個商品...")
    start_time = time.time()
    
    try:
        # 啟動 loky 平行運算池
        results = Parallel(n_jobs=CPU_WORKERS, backend="loky")(
            delayed(run_single_sku_forecast)(
                row, 
                sales_by_sku.get(row['GoodsID'], pd.DataFrame()), # 只拿自己 SKU 的資料
                next_month, 
                cat_index_map
            ) 
            for _, row in target_skus.iterrows()
        )
        
        forecast_df = pd.DataFrame(results)
        # 建議進貨量 = 目標庫存 - 現有庫存 (不能為負數)
        forecast_df['Suggested_Order'] = (forecast_df['Target_Stock'] - forecast_df['CurrStock']).clip(lower=0)
        
        # 匯出 CSV，按建議進貨量降冪排列
        forecast_df.sort_values(by='Suggested_Order', ascending=False).to_csv(output_path, index=False, encoding='utf-8-sig')
        
        end_time = time.time()
        print(f"✅ 系統分析完成！總耗時: {round(end_time - start_time, 2)} 秒。")
        print(f"📂 報告已匯出至: {output_path}")
            
    except Exception as e:
        print(f"❌ 執行平行運算時發生中斷: {e}")

if __name__ == "__main__":
    main()