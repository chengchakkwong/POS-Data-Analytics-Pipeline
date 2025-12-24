import pandas as pd
import numpy as np
import os
import time
import warnings
import logging
from joblib import Parallel, delayed

# --- 系統配置與效能設定 ---
TEST_MODE = False      # 正式開始！設為 False 以處理全量 7000 個商品
SAMPLE_SIZE = 50       # (測試模式才有效)
# 💡 效能提示：4070 Ti (12GB) 的黃金平衡點通常在 3-4。
# 超過 5 個 worker 會導致 CPU 序列化數據過慢，且 GPU 上下文切換頻繁，反而降低總體速度。
GPU_WORKERS = 4        
# ------------------------

# 嘗試導入 NeuralProphet 以支援 GPU 加速
try:
    from neuralprophet import NeuralProphet
    import torch
    HAS_NEURAL_PROPHET = True
    # 偵測 NVIDIA GPU
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if DEVICE == "cuda":
        # 4070 Ti 有 12GB VRAM，優化顯存分配
        torch.cuda.empty_cache()
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
except ImportError:
    from prophet import Prophet
    HAS_NEURAL_PROPHET = False
    DEVICE = "cpu"
    vram_total = 0

# 忽略日誌干擾
warnings.filterwarnings('ignore')
logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('neuralprophet').setLevel(logging.ERROR)

# --- 1. 數據分類與預處理 ---

def calculate_category_seasonal_indices(sales_df, stock_df):
    """計算類別層級的季節性係數 (用於輔助數據稀疏商品)"""
    sales_df = sales_df.copy()
    sales_df['rDate'] = pd.to_datetime(sales_df['rDate'])
    df_with_cat = sales_df.merge(stock_df[['GoodsID', 'Category']], on='GoodsID', how='left')
    cat_monthly = df_with_cat.groupby(['Category', pd.Grouper(key='rDate', freq='MS')])['TotalQty'].sum().reset_index()
    cat_avg = cat_monthly.groupby('Category')['TotalQty'].mean().rename('Cat_Avg_Qty').reset_index()
    cat_indices = cat_monthly.merge(cat_avg, on='Category')
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
    merged['TotalProfit'] = merged['TotalAmt'] - (merged['LastInCost'] * merged['TotalQty'])
    merged = merged.sort_values('TotalProfit', ascending=False)
    
    total_prof = merged['TotalProfit'].sum()
    if total_prof > 0:
        merged['ProfitRatio'] = merged['TotalProfit'].cumsum() / total_prof
    else:
        merged['ProfitRatio'] = 1.0
        
    merged['ABC_Class'] = np.select([(merged['ProfitRatio'] <= 0.7), (merged['ProfitRatio'] <= 0.9)], ['A', 'B'], default='C')

    # XYZ 分析：確保 Mean_Monthly 是基於完整的時間範圍 (避免 Mean 過高)
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
    
    is_new = (analysis_df['Month_Age'] < 4)
    analysis_df.loc[is_new, 'ABC_Class'], analysis_df.loc[is_new, 'XYZ_Class'] = 'New', 'New'
    
    return analysis_df

# --- 2. GPU 加速預測核心 ---

def run_single_sku_forecast(item, sales_df, next_month, cat_index_map):
    """單一 SKU 預測單元：利用 4070 Ti 的 CUDA 進行 NeuralProphet 運算"""
    gid, abc, xyz, cv, cat, total_qty_year = item['GoodsID'], item['ABC_Class'], item['XYZ_Class'], item['CV'], item['Category'], item['TotalQty']
    item_sales = sales_df[sales_df['GoodsID'] == gid].sort_values('rDate')
    mean_qty = item['Mean_Monthly']
    
    # 建立完整的時間序列，填補 0 銷量月份 (這對 AI 預測極其重要)
    full_date_range = pd.date_range(start=sales_df['rDate'].min(), end=sales_df['rDate'].max(), freq='MS')
    
    base_pred = 0
    used_gpu = False
    try:
        if abc == 'New':
            recent = item_sales[item_sales['rDate'] >= (sales_df['rDate'].max() - pd.Timedelta(weeks=4))]
            base_pred = (recent['TotalQty'].sum() / 4) * 4 if not recent.empty else mean_qty
        
        elif xyz == 'Y':
            # 填補 0 值：確保模型看到淡季
            m_df = item_sales.groupby(pd.Grouper(key='rDate', freq='MS'))['TotalQty'].sum().reindex(full_date_range, fill_value=0).reset_index()
            m_df.columns = ['ds', 'y']
            
            if len(m_df) >= 12:
                if HAS_NEURAL_PROPHET:
                    m = NeuralProphet(
                        yearly_seasonality=True,
                        weekly_seasonality=False,
                        daily_seasonality=False,
                        accelerator=DEVICE if DEVICE == "cuda" else None,
                        epochs=30, 
                        batch_size=None, # 自動調整批次，避免數據量太小出錯
                        trainer_config={"logger": False, "enable_checkpointing": False, "enable_progress_bar": False}
                    )
                    m.fit(m_df, freq="MS", progress=None)
                    future = m.make_future_dataframe(m_df, periods=1)
                    forecast = m.predict(future)
                    base_pred = max(0, forecast.iloc[-1]['yhat1'])
                    used_gpu = (DEVICE == "cuda")
                    if used_gpu: torch.cuda.empty_cache()
                else:
                    m = Prophet(yearly_seasonality=True, uncertainty_samples=0)
                    m.fit(m_df)
                    base_pred = max(0, m.predict(m.make_future_dataframe(periods=1, freq='MS')).iloc[-1]['yhat'])
            else:
                base_pred = m_df['y'].tail(3).mean()
        
        elif xyz == 'X':
            base_pred = item_sales.groupby(pd.Grouper(key='rDate', freq='MS'))['TotalQty'].sum().tail(3).mean()
        
        else:
            base_pred = item_sales.groupby(pd.Grouper(key='rDate', freq='MS'))['TotalQty'].sum().tail(6).median()
    except:
        base_pred = mean_qty

    # --- 防爆門限與補貨邏輯 ---
    
    # [優化 1] 基礎預測不應超過月平均的 3 倍 (針對非新品)
    if abc != 'New' and mean_qty > 0:
        base_pred = min(base_pred, mean_qty * 3)
    
    # 如果預測出來還是 0 但它是 A 類穩定品，強制回退到平均值 (雙重保險)
    if base_pred <= 0 and abc in ['A', 'B'] and mean_qty > 0:
        base_pred = mean_qty

    boost = min(2.0, cat_index_map.get((cat, next_month), 1.0)) if (abc == 'New' or xyz == 'Z') else 1.0
    final_demand = base_pred * max(1.0, boost)
    
    # 安全庫存防禦
    safety_ratio = min(0.5 if abc == 'A' else 0.3, cv * 0.5)
    target_stock = final_demand + (final_demand * safety_ratio)
    
    # [優化 2] 強力現實檢查 (Reality Check)
    if abc != 'New' and mean_qty > 0:
        target_stock_limit = max(mean_qty * 4, 2)
        target_stock = min(target_stock, target_stock_limit)
        
    if abc != 'New' and total_qty_year > 0:
        target_stock = min(target_stock, total_qty_year)

    return {
        'GoodsID': gid, 'Name': item['Name'], 'ABC_XYZ': f"{abc}{xyz}",
        'Base_Demand': round(base_pred, 2),
        'Final_Demand': round(final_demand, 2), 'Target_Stock': round(target_stock, 2),
        'CurrStock': item['CurrStock'], 'Used_GPU': used_gpu
    }

# --- 3. 主程序執行邏輯 ---

def main():
    input_stock = "data/processed/DetailGoodsStockToday.csv"
    input_sales = "data/processed/vw_GoodsDailySales_cache.parquet"
    output_path = "data/insights/final_inventory_plan.csv"
    
    print(f"🚀 啟動優化版智慧補貨系統 (裝置: {DEVICE.upper()})")
    if DEVICE == "cuda":
        print(f"💎 裝置硬體: NVIDIA RTX 4070 Ti (總顯存: {vram_total:.2f} GB)")
        print(f"🧵 最佳並行進程數: {GPU_WORKERS}")
        
    if TEST_MODE:
        print(f"⚠️ 警告：目前處於測試模式，僅處理前 {SAMPLE_SIZE} 個商品。")
    else:
        print(f"🔥 全量模式啟動！正在分析全店數據...")
    
    if not os.path.exists(input_stock) or not os.path.exists(input_sales):
        print("❌ 錯誤: 找不到數據源。")
        return

    df_stock, df_sales = pd.read_csv(input_stock), pd.read_parquet(input_sales)
    
    print("📊 執行 ABC-XYZ 分類與類別加成分析...")
    analysis_df = perform_abc_xyz_analysis(df_stock, df_sales)
    cat_index_map = calculate_category_seasonal_indices(df_sales, df_stock)
    
    # 篩選需要預測的商品
    target_skus = analysis_df[analysis_df['ABC_Class'].isin(['A', 'B', 'New'])]
    
    if TEST_MODE:
        target_skus = target_skus.head(SAMPLE_SIZE)
        
    next_month = (pd.to_datetime(df_sales['rDate']).max() + pd.DateOffset(months=1)).month
    
    print(f"🔮 開始高效並行預測 {len(target_skus)} 個商品...")
    start_time = time.time()
    
    # 並行進程數設定
    n_workers = GPU_WORKERS if DEVICE == "cuda" else -1
    
    try:
        results = Parallel(n_jobs=n_workers, backend="loky")(
            delayed(run_single_sku_forecast)(row, df_sales, next_month, cat_index_map) 
            for _, row in target_skus.iterrows()
        )
        
        forecast_df = pd.DataFrame(results)
        forecast_df['Suggested_Order'] = (forecast_df['Target_Stock'] - forecast_df['CurrStock']).clip(lower=0)
        forecast_df.sort_values(by='Suggested_Order', ascending=False).to_csv(output_path, index=False, encoding='utf-8-sig')
        
        gpu_count = forecast_df['Used_GPU'].sum()
        end_time = time.time()
        
        print(f"✅ 分析完成！總耗時: {round(end_time - start_time, 2)} 秒。")
        print(f"📊 統計：共 {len(forecast_df)} 項，GPU 加速 {gpu_count} 項。")
            
    except Exception as e:
        print(f"❌ 執行中斷: {e}")

if __name__ == "__main__":
    main()