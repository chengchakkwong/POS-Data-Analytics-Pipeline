import pandas as pd
import numpy as np
import os
import time
import warnings
import logging
import multiprocessing
from joblib import Parallel, delayed
from tqdm.auto import tqdm
import contextlib
import joblib

# --- 終極靜音設定：封殺所有底層碎碎念 ---
logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('neuralprophet').setLevel(logging.ERROR)
logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)

cmdstanpy_logger = logging.getLogger('cmdstanpy')
cmdstanpy_logger.setLevel(logging.ERROR)
cmdstanpy_logger.disabled = True
# ----------------------------------------

# --- 系統配置與效能設定 ---
TEST_MODE = False      # 設為 False 處理全量商品
SAMPLE_SIZE = 50       # (測試模式才有效)
CPU_WORKERS = max(1, multiprocessing.cpu_count() - 2) 
# ------------------------

try:
    from neuralprophet import NeuralProphet
    import torch
    HAS_NEURAL_PROPHET = True
except ImportError:
    from prophet import Prophet
    HAS_NEURAL_PROPHET = False

warnings.filterwarnings('ignore')

# --- 1. 輔助計算函數 ---

def calculate_category_seasonal_indices(sales_df, stock_df):
    """計算類別層級的季節性係數 (這部分仍需保留以作需求加成)"""
    sales_df = sales_df.copy()
    sales_df['rDate'] = pd.to_datetime(sales_df['rDate'])
    df_with_cat = sales_df.merge(stock_df[['GoodsID', 'Category']], on='GoodsID', how='left')
    cat_monthly = df_with_cat.groupby(['Category', pd.Grouper(key='rDate', freq='MS')])['TotalQty'].sum().reset_index()
    cat_avg = cat_monthly.groupby('Category')['TotalQty'].mean().rename('Cat_Avg_Qty').reset_index()
    cat_indices = cat_monthly.merge(cat_avg, on='Category')
    cat_indices['Cat_Avg_Qty'] = cat_indices['Cat_Avg_Qty'].replace(0, 1) 
    cat_indices['Seasonal_Index'] = cat_indices['TotalQty'] / cat_indices['Cat_Avg_Qty']
    cat_indices['Month'] = cat_indices['rDate'].dt.month
    index_map = cat_indices.groupby(['Category', 'Month'])['Seasonal_Index'].mean().to_dict()
    return index_map

# --- 2. 預測核心 (純 CPU 穩定極速版) ---

def run_single_sku_forecast(item, item_sales, next_month, cat_index_map):
    import warnings
    import logging
    import sys
    import os
    from contextlib import contextmanager

    warnings.filterwarnings('ignore')
    logging.getLogger('NP').setLevel(logging.ERROR)
    logging.getLogger('NP.df_utils').setLevel(logging.ERROR)
    logging.getLogger('py.warnings').setLevel(logging.ERROR)

    @contextmanager
    def suppress_output():
        with open(os.devnull, 'w') as devnull:
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = devnull, devnull
            try:
                yield
            finally:
                sys.stdout, sys.stderr = old_stdout, old_stderr

    gid = item['GoodsID']
    p_code = item.get('ProductCode', 'Unknown') 
    
    # 直接從上一棒的 CSV 讀取標籤與預算出的均值
    abc, xyz, cv = item['ABC_Class'], item['XYZ_Class'], item['CV']
    cat = item.get('Category', 'Unknown')
    mean_qty = item.get('Mean_Monthly_Qty', 0)
    
    if item_sales.empty:
        item_sales = pd.DataFrame(columns=['rDate', 'TotalQty'])
    else:
        item_sales = item_sales.sort_values('rDate')
        
    base_pred = 0

    try:
        # [策略 1] 新品精準預測 (Run-rate 預估)
        if abc == 'New':
            if not item_sales.empty:
                first_sale_date = item_sales['rDate'].min()
                last_record_date = item_sales['rDate'].max()
                active_days = (last_record_date - first_sale_date).days
                active_days = max(active_days, 1) 
                
                total_sold = item_sales['TotalQty'].sum()
                daily_run_rate = total_sold / active_days
                base_pred = daily_run_rate * 30
                
                if active_days < 7:
                    base_pred = min(base_pred, total_sold * 3)
            else:
                base_pred = mean_qty
        
        # [策略 2] X/Y 類：交給 AI 模型捕捉趨勢與季節性
        elif xyz in ['X', 'Y']: 
            m_df = item_sales.groupby(pd.Grouper(key='rDate', freq='MS'))['TotalQty'].sum().reset_index()
            m_df.columns = ['ds', 'y']
            
            if len(m_df) >= 12:
                with suppress_output():
                    if HAS_NEURAL_PROPHET:
                        m = NeuralProphet(
                            yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
                            learning_rate=0.1, accelerator="cpu", epochs=30, batch_size=None,
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
                base_pred = m_df['y'].tail(3).mean() if not m_df.empty else mean_qty
                
        # [策略 3] Z 類：最高防禦 + 剔除極端大單
        else:
            if not item_sales.empty:
                m_sales = item_sales.groupby(pd.Grouper(key='rDate', freq='MS'))['TotalQty'].sum()
                historical_max = m_sales.max()
                last_6m = m_sales.tail(6)
                
                if len(last_6m) >= 2:
                    last_6m_max = last_6m.max()
                    # 發現異常客製化大單，降階取第二大值
                    if last_6m_max == historical_max and last_6m_max > 0:
                        base_pred = last_6m.nlargest(2).iloc[1] 
                    else:
                        base_pred = last_6m_max
                elif len(last_6m) == 1:
                    base_pred = last_6m.iloc[0]
                else:
                    base_pred = mean_qty
            else:
                base_pred = mean_qty
                
    except Exception as e:
        base_pred = mean_qty

    # --- 防爆門限與 目標庫存計算 (移除下單量邏輯) ---
    if abc != 'New' and mean_qty > 0:
        base_pred = min(base_pred, mean_qty * 3)
        
    if base_pred <= 0 and abc in ['A', 'B'] and mean_qty > 0:
        base_pred = mean_qty

    # 季節性加成
    boost = min(2.0, cat_index_map.get((cat, next_month), 1.0)) if (abc == 'New' or xyz == 'Z') else 1.0
    final_demand = base_pred * max(1.0, boost)
    
    # 抓取安全庫存比例
    safety_ratio = min(0.5 if abc == 'A' else 0.3, cv * 0.5)
    target_stock = final_demand + (final_demand * safety_ratio)
    
    # 目標庫存上限防爆
    if abc != 'New' and mean_qty > 0:
        target_stock_limit = max(mean_qty * 4, 2)
        target_stock = min(target_stock, target_stock_limit)

    return {
        'ProductCode': p_code, 
        'Name': item.get('Name', 'Unknown'), 
        'ABC_XYZ': f"{abc}{xyz}",
        'Strategy': item.get('Strategy', 'Unknown'), # 加入戰略標籤
        'CurrStock': item.get('CurrStock', 0),       # 保留目前庫存供參考
        'Base_Demand': round(base_pred, 2), 
        'Final_Demand': round(final_demand, 2), 
        'Target_Stock': round(target_stock, 2)
        # 🗑️ 已徹底移除 Suggested_Order 與 Final_Order
    }

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback

# --- 3. 主程序執行邏輯 ---

def main():
    input_labels = "data/insights/abc_xyz_analysis.csv"
    input_stock = "data/processed/products_details_for_replenishment.csv"
    input_sales = "data/processed/vw_GoodsDailySales_partitioned"
    output_path = "data/insights/target_stock_plan.csv" # 🎯 檔名改成 目標庫存計畫
    
    print(f"🚀 啟動 AI 智慧預測引擎 (目標庫存規劃模式)")
    print(f"🧵 配置平行進程數: {CPU_WORKERS} / {multiprocessing.cpu_count()}")
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        # 讀取標籤
        if not os.path.exists(input_labels):
            print("❌ 錯誤: 找不到 abc_xyz_analysis.csv！請確認是否已經先執行過引擎一。")
            return
        df_labels = pd.read_csv(input_labels)
        
        # 讀取今日庫存
        df_stock = pd.read_csv(input_stock)
        
        # 讀取歷史銷售
        if input_sales.endswith('.parquet') or os.path.isdir(input_sales):
            df_sales = pd.read_parquet(input_sales)
        else:
            df_sales = pd.read_csv(input_sales)
            
    except Exception as e:
        print(f"❌ 讀取資料失敗: {e}")
        return

    print("🔄 正在同步今日最新庫存與商品標籤...")
    analysis_df = pd.merge(
        df_labels[['GoodsID', 'ABC_Class', 'XYZ_Class', 'CV', 'Mean_Monthly_Qty', 'Strategy']], 
        df_stock, 
        on='GoodsID', 
        how='inner'
    )
    
    next_month = (pd.to_datetime(df_sales['rDate']).max() + pd.DateOffset(months=1)).month
    cat_index_map = calculate_category_seasonal_indices(df_sales, df_stock)
    
    target_skus = analysis_df[analysis_df['ABC_Class'].isin(['A', 'B', 'New'])]
    
    if TEST_MODE:
        print(f"⚠️ 警告：目前處於測試模式，僅處理前 {SAMPLE_SIZE} 個重點商品。")
        target_skus = target_skus.head(SAMPLE_SIZE)
    else:
        print(f"🔥 全量模式啟動！準備預測 {len(target_skus)} 個重點商品...")
    
    print("📦 預先打包各 SKU 的專屬歷史資料...")
    sales_by_sku = {gid: df for gid, df in df_sales.groupby('GoodsID')}

    start_time = time.time()
    
    try:
        # 執行多進程預測
        with tqdm_joblib(tqdm(desc="🔮 AI 預測進度", total=len(target_skus))):
            results = Parallel(n_jobs=CPU_WORKERS, backend="loky")(
                delayed(run_single_sku_forecast)(
                    row, 
                    sales_by_sku.get(row['GoodsID'], pd.DataFrame()), 
                    next_month, 
                    cat_index_map
                ) 
                for _, row in target_skus.iterrows()
            )
        
        forecast_df = pd.DataFrame(results)

        # 🌟 分流 2：C 類長尾商品極簡規劃 (包含 8028 防爆邏輯) 🌟
        c_class_skus = analysis_df[analysis_df['ABC_Class'] == 'C']
        
        if not c_class_skus.empty and not TEST_MODE:
            print(f"🧹 正在光速處理 {len(c_class_skus)} 個 C 類長尾商品...")
            c_results = []
            
            for _, row in c_class_skus.iterrows():
                mean_qty = row.get('Mean_Monthly_Qty', 0)
                # 防呆：Mean_Monthly_Qty 可能為 NaN
                if pd.isna(mean_qty):
                    mean_qty = 0
                curr_stock = row.get('CurrStock', 0)
                
                # C 類規則（不看特殊原因）
                # 1) mean_qty == 0 時直接回傳 0（避免把非賣商品誤判成要進貨）
                if mean_qty <= 0:
                    target_stock = 0
                else:
                    # 2) 有 FirstOrderQty 時：若 FirstOrderQty 過大（> mean_qty*12），壓回 mean_qty*1.2
                    first_order_qty = row.get('FirstOrderQty', 0)
                    if pd.notna(first_order_qty) and first_order_qty > 0:
                        if first_order_qty > (mean_qty * 12):
                            target_stock = mean_qty * 1.2
                        else:
                            target_stock = first_order_qty
                    else:
                        # 3) 沒有 FirstOrderQty：用 mean_qty*1.2 當作保守目標
                        target_stock = mean_qty * 1.2
                    
                c_results.append({
                    'ProductCode': row.get('ProductCode', 'Unknown'), 
                    'Name': row.get('Name', 'Unknown'), 
                    'ABC_XYZ': f"C{row['XYZ_Class']}",
                    'Strategy': row.get('Strategy', 'Unknown'),
                    'CurrStock': curr_stock,
                    'Base_Demand': round(mean_qty, 2), 
                    'Final_Demand': round(mean_qty, 2), 
                    'Target_Stock': round(target_stock, 2)
                    # 🗑️ 已徹底移除 Suggested_Order 與 Final_Order
                })
                
            c_forecast_df = pd.DataFrame(c_results)
            # 將 A/B 類的 AI 預測結果與 C 類結果合併
            forecast_df = pd.concat([forecast_df, c_forecast_df], ignore_index=True)

        # 匯出 CSV (改以 目標庫存 降冪排序)
        forecast_df.sort_values(by='Target_Stock', ascending=False).to_csv(output_path, index=False, encoding='utf-8-sig')
        
        end_time = time.time()
        print(f"\n✅ 今日目標庫存計畫產生完成！總耗時: {round(end_time - start_time, 2)} 秒。")
        print(f"📂 報告已匯出至: {output_path}")
            
    except Exception as e:
        print(f"\n❌ 執行預測時發生中斷: {e}")

if __name__ == "__main__":
    main()