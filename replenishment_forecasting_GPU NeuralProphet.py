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
# 1. 關閉 Prophet 與神經網路的常規警告
logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('neuralprophet').setLevel(logging.ERROR)
logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)

# 2. 徹底封殺底層 C++ 引擎 (cmdstanpy) 的狂洗版 INFO
cmdstanpy_logger = logging.getLogger('cmdstanpy')
cmdstanpy_logger.setLevel(logging.ERROR)
cmdstanpy_logger.disabled = True
# ----------------------------------------

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
    if 'LastInCost' not in merged.columns:
        merged['LastInCost'] = 0
    merged['TotalProfit'] = merged['TotalAmt'] - (merged['LastInCost'] * merged['TotalQty'])
    
    # 🌟 [核心升級 1] 時間公平邏輯 (月均毛利) 🌟
    first_sale_12m = df_12m.groupby('GoodsID')['rDate'].min().reset_index()
    first_sale_12m.columns = ['GoodsID', 'First_Sale_Date']
    merged = pd.merge(merged, first_sale_12m, on='GoodsID', how='left')
    
    # 🛡️ [防爆機制] 算出存活月數，若無紀錄視為12個月，若少於1個月強制進位成1個月
    merged['Months_Active'] = ((last_date - merged['First_Sale_Date']).dt.days / 30)
    merged['Months_Active'] = merged['Months_Active'].fillna(12).clip(lower=1) 
    
    merged['Monthly_Avg_Profit'] = merged['TotalProfit'] / merged['Months_Active']
    merged = merged.sort_values('Monthly_Avg_Profit', ascending=False)
    
    total_prof = merged['Monthly_Avg_Profit'].sum()
    if total_prof > 0:
        merged['ProfitRatio'] = merged['Monthly_Avg_Profit'].cumsum() / total_prof
    else:
        merged['ProfitRatio'] = 1.0
        
    merged['ABC_Class'] = np.select([(merged['ProfitRatio'] <= 0.7), (merged['ProfitRatio'] <= 0.9)], ['A', 'B'], default='C')

    # XYZ 分析
    monthly_matrix = sales_df.groupby(['GoodsID', pd.Grouper(key='rDate', freq='MS')])['TotalQty'].sum().unstack(fill_value=0)
    stats = pd.DataFrame(index=monthly_matrix.index)
    stats['Mean_Monthly'] = monthly_matrix.mean(axis=1)
    # 修正：確保長度匹配
    cv_values = np.where(stats['Mean_Monthly'] > 0, monthly_matrix.std(axis=1) / stats['Mean_Monthly'], 9.99)
    stats['CV'] = cv_values
    stats['XYZ_Class'] = np.select([(stats['CV'] <= 0.5), (stats['CV'] <= 1.0)], ['X', 'Y'], default='Z')
    
    first_sale = sales_df.groupby('GoodsID')['rDate'].min().reset_index()
    first_sale['Month_Age'] = (last_date.year - first_sale['rDate'].dt.year) * 12 + (last_date.month - first_sale['rDate'].dt.month)
    
    analysis_df = merged.merge(stats[['CV', 'XYZ_Class', 'Mean_Monthly']], left_on='GoodsID', right_index=True, how='left')
    analysis_df = analysis_df.merge(first_sale[['GoodsID', 'Month_Age']], on='GoodsID', how='left').fillna({
        'Month_Age': 99, 'XYZ_Class': 'Z', 'CV': 9.99, 'Mean_Monthly': 0
    })
    
    is_new = (analysis_df['Month_Age'] < 4)
    analysis_df.loc[is_new, 'ABC_Class'] = 'New'
    analysis_df.loc[is_new, 'XYZ_Class'] = 'New'
    
    return analysis_df

# --- 2. 預測核心 (純 CPU 穩定極速版) ---

def run_single_sku_forecast(item, item_sales, next_month, cat_index_map):
    """單一 SKU 預測單元 (接收專屬的小 DataFrame)"""
    
    # 👇 [新增 1] 導入底層系統控制套件
    import warnings
    import logging
    import sys
    import os
    from contextlib import contextmanager

    # 封殺標準警告
    warnings.filterwarnings('ignore')
    logging.getLogger('NP').setLevel(logging.ERROR)
    logging.getLogger('NP.df_utils').setLevel(logging.ERROR)
    logging.getLogger('py.warnings').setLevel(logging.ERROR)

    # 👇 [新增 2] 建立「黑洞靜音魔法」，強制吞噬所有不受控的 tqdm 進度條
    @contextmanager
    def suppress_output():
        with open(os.devnull, 'w') as devnull:
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = devnull, devnull
            try:
                yield
            finally:
                sys.stdout, sys.stderr = old_stdout, old_stderr
    # -----------------------------------------------------------

    gid = item['GoodsID']
    p_code = item.get('ProductCode', 'Unknown') 
    
    abc, xyz, cv = item['ABC_Class'], item['XYZ_Class'], item['CV']
    cat, total_qty_year = item.get('Category', 'Unknown'), item.get('TotalQty', 0)
    
    if item_sales.empty:
        item_sales = pd.DataFrame(columns=['rDate', 'TotalQty'])
    else:
        item_sales = item_sales.sort_values('rDate')
        
    mean_qty = item['Mean_Monthly']
    base_pred = 0

    try:
        # [策略 1] 新品精準預測
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
        
        # [策略 2] X/Y 類：交給 AI 模型捕捉
        elif xyz in ['X', 'Y']: 
            m_df = item_sales.groupby(pd.Grouper(key='rDate', freq='MS'))['TotalQty'].sum().reset_index()
            m_df.columns = ['ds', 'y']
            
            if len(m_df) >= 12:
                # 👇 [新增 3] 把 AI 訓練過程關進黑洞裡！
                with suppress_output():
                    if HAS_NEURAL_PROPHET:
                        m = NeuralProphet(
                            yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
                            learning_rate=0.1, # 💡 [新增 4] 直接指定學習率，跳過尋找 LR 的過程，運算速度爆增！
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
                base_pred = m_df['y'].tail(3).mean() if not m_df.empty else mean_qty
                
        # [策略 3] Z 類：長尾死水商品
        else:
            if not item_sales.empty:
                base_pred = item_sales.groupby(pd.Grouper(key='rDate', freq='MS'))['TotalQty'].sum().tail(6).median()
            else:
                base_pred = mean_qty
                
    except Exception as e:
        base_pred = mean_qty

    # --- 防爆門限與補貨邏輯 ---
    if abc != 'New' and mean_qty > 0:
        base_pred = min(base_pred, mean_qty * 3)
        
    if base_pred <= 0 and abc in ['A', 'B'] and mean_qty > 0:
        base_pred = mean_qty

    boost = min(2.0, cat_index_map.get((cat, next_month), 1.0)) if (abc == 'New' or xyz == 'Z') else 1.0
    final_demand = base_pred * max(1.0, boost)
    
    safety_ratio = min(0.5 if abc == 'A' else 0.3, cv * 0.5)
    target_stock = final_demand + (final_demand * safety_ratio)
    
    if abc != 'New' and mean_qty > 0:
        target_stock_limit = max(mean_qty * 4, 2)
        target_stock = min(target_stock, target_stock_limit)
        
    if abc != 'New' and total_qty_year > 0:
        target_stock = min(target_stock, total_qty_year)

    curr_stock = item.get('CurrStock', 0)
    suggested_order = max(0, target_stock - curr_stock)
    moq = item.get('MinOrderQty', 0)
    
    if suggested_order > (moq / 2):
        final_order = max(suggested_order, moq) 
    else:
        final_order = 0

    return {
        'ProductCode': p_code, 
        'Name': item.get('GoodsName1', item.get('Name', 'Unknown')), 
        'ABC_XYZ': f"{abc}{xyz}",
        'MinOrderQty': moq,
        'CurrStock': curr_stock,
        'Base_Demand': round(base_pred, 2), 
        'Final_Demand': round(final_demand, 2), 
        'Target_Stock': round(target_stock, 2), 
        'Suggested_Order': round(suggested_order, 2),
        'Final_Order': round(final_order, 2) 
    }

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """為 joblib 加上單行動態 tqdm 進度條的黑魔法"""
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
    input_stock = "data/processed/DetailGoodsStockToday.csv"
    input_sales = "data/processed/vw_GoodsDailySales_partitioned"
    output_path = "data/insights/final_inventory_plan.csv"
    
    print(f"🚀 啟動優化版智慧補貨系統 (運算引擎: CPU 多核心模式)")
    print(f"🧵 配置平行進程數: {CPU_WORKERS} / {multiprocessing.cpu_count()}")
        
    if TEST_MODE:
        print(f"⚠️ 警告：目前處於測試模式，僅處理前 {SAMPLE_SIZE} 個商品。")
    else:
        print(f"🔥 全量模式啟動！正在分析全店數據...")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        df_stock = pd.read_csv(input_stock)
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
    
    target_skus = analysis_df[analysis_df['ABC_Class'].isin(['A', 'B', 'New'])]
    
    if TEST_MODE:
        target_skus = target_skus.head(SAMPLE_SIZE)
        
    next_month = (pd.to_datetime(df_sales['rDate']).max() + pd.DateOffset(months=1)).month
    
    print("📦 正在解決記憶體問題：預先打包各 SKU 的專屬歷史資料...")
    sales_by_sku = {gid: df for gid, df in df_sales.groupby('GoodsID')}

    print(f"🔮 開始高效並行預測 {len(target_skus)} 個商品...")
    start_time = time.time()
    
    try:
        # 💡 使用 with tqdm_joblib 來包裝你的平行運算
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
        
        # 匯出 CSV，按 Final_Order (最終下單量) 降冪排列
        forecast_df.sort_values(by='Final_Order', ascending=False).to_csv(output_path, index=False, encoding='utf-8-sig')
        
        end_time = time.time()
        print(f"\n✅ 系統分析完成！總耗時: {round(end_time - start_time, 2)} 秒。")
        print(f"📂 報告已匯出至: {output_path}")
            
    except Exception as e:
        print(f"\n❌ 執行平行運算時發生中斷: {e}")

if __name__ == "__main__":
    main()