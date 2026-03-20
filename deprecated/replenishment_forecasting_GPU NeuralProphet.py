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

# --- 1. 數據分類與預處理 ---

def calculate_category_seasonal_indices(sales_df, stock_df):
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

def perform_abc_xyz_analysis(stock_df, sales_df):
    sales_df['rDate'] = pd.to_datetime(sales_df['rDate'])
    last_date = sales_df['rDate'].max()
    start_12m = last_date - pd.DateOffset(months=12)
    
    df_12m = sales_df[sales_df['rDate'] >= start_12m].copy()
    summary_12m = df_12m.groupby('GoodsID').agg({'TotalQty': 'sum', 'TotalAmt': 'sum'}).reset_index()
    
    merged = pd.merge(stock_df, summary_12m, on='GoodsID', how='left').fillna(0)
    if 'LastInCost' not in merged.columns:
        merged['LastInCost'] = 0
    merged['TotalProfit'] = merged['TotalAmt'] - (merged['LastInCost'] * merged['TotalQty'])
    
    first_sale_12m = df_12m.groupby('GoodsID')['rDate'].min().reset_index()
    first_sale_12m.columns = ['GoodsID', 'First_Sale_Date']
    merged = pd.merge(merged, first_sale_12m, on='GoodsID', how='left')
    
    merged['Months_Active'] = ((last_date - merged['First_Sale_Date']).dt.days / 30)
    merged['Months_Active'] = merged['Months_Active'].fillna(12).clip(lower=1) 
    merged['Monthly_Avg_Profit'] = merged['TotalProfit'] / merged['Months_Active']

    monthly_matrix = sales_df.groupby(['GoodsID', pd.Grouper(key='rDate', freq='MS')])['TotalQty'].sum().unstack(fill_value=0)
    stats = pd.DataFrame(index=monthly_matrix.index)
    stats['Mean_Monthly'] = monthly_matrix.mean(axis=1)
    cv_values = np.where(stats['Mean_Monthly'] > 0, monthly_matrix.std(axis=1) / stats['Mean_Monthly'], 9.99)
    stats['CV'] = cv_values
    stats['XYZ_Class'] = np.select([(stats['CV'] <= 0.5), (stats['CV'] <= 1.0)], ['X', 'Y'], default='Z')
    
    first_sale = sales_df.groupby('GoodsID')['rDate'].min().reset_index()
    first_sale['Month_Age'] = (last_date.year - first_sale['rDate'].dt.year) * 12 + (last_date.month - first_sale['rDate'].dt.month)
    
    analysis_df = merged.merge(stats[['CV', 'XYZ_Class', 'Mean_Monthly']], left_on='GoodsID', right_index=True, how='left')
    analysis_df = analysis_df.merge(first_sale[['GoodsID', 'Month_Age']], on='GoodsID', how='left').fillna({
        'Month_Age': 99, 'XYZ_Class': 'Z', 'CV': 9.99, 'Mean_Monthly': 0
    })
    
    # [保留你的完美邏輯] 先把新品抽離，再只對成熟商品做 ABC 70/20/10 切分
    analysis_df['ProfitRatio'] = 1.0
    analysis_df['ABC_Class'] = 'C'
    is_new = (analysis_df['Month_Age'] < 4)
    mature_mask = ~is_new

    mature_df = analysis_df.loc[mature_mask].copy().sort_values('Monthly_Avg_Profit', ascending=False)
    total_prof_mature = mature_df['Monthly_Avg_Profit'].sum()
    if total_prof_mature > 0 and not mature_df.empty:
        mature_df['ProfitRatio'] = mature_df['Monthly_Avg_Profit'].cumsum() / total_prof_mature
    else:
        mature_df['ProfitRatio'] = 1.0

    mature_df['ABC_Class'] = np.select(
        [(mature_df['ProfitRatio'] <= 0.7), (mature_df['ProfitRatio'] <= 0.9)],
        ['A', 'B'],
        default='C'
    )
    analysis_df.loc[mature_df.index, 'ProfitRatio'] = mature_df['ProfitRatio']
    analysis_df.loc[mature_df.index, 'ABC_Class'] = mature_df['ABC_Class']

    analysis_df.loc[is_new, 'ABC_Class'] = 'New'
    analysis_df.loc[is_new, 'XYZ_Class'] = 'New'
    
    return analysis_df

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
                
        # 🌟 [策略 3 新增] Z 類：長尾死水商品 (最高防禦 + 剔除極端值 B2B 大單) 🌟
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
    input_stock = "data/processed/products_details_for_replenishment.csv"
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

    print(f"🔮 開始高效並行預測 {len(target_skus)} 個重點商品...")
    start_time = time.time()
    
    try:
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

        # 🌟 [策略 4 新增] C 類長尾商品極簡補貨 (引入 FirstOrderQty) 🌟
        c_class_skus = analysis_df[analysis_df['ABC_Class'] == 'C']
        
        if not c_class_skus.empty and not TEST_MODE:
            print(f"🧹 正在快速處理 {len(c_class_skus)} 個 C 類長尾商品...")
            c_results = []
            
            for _, row in c_class_skus.iterrows():
                mean_qty = row['Mean_Monthly']
                curr_stock = row.get('CurrStock', 0)
                moq = row.get('MinOrderQty', 0)
                
                # 優先使用 FirstOrderQty，若無則用月均銷量 1.5 倍
                first_order_qty = row.get('FirstOrderQty', 0)
                if pd.notna(first_order_qty) and first_order_qty > 0:
                    target_stock = first_order_qty
                else:
                    target_stock = mean_qty * 1.5
                    
                suggested_order = max(0, target_stock - curr_stock)
                final_order = max(suggested_order, moq) if suggested_order > 0 else 0
                
                c_results.append({
                    'ProductCode': row.get('ProductCode', 'Unknown'), 
                    'Name': row.get('GoodsName1', row.get('Name', 'Unknown')), 
                    'ABC_XYZ': f"C{row['XYZ_Class']}",
                    'MinOrderQty': moq,
                    'CurrStock': curr_stock,
                    'Base_Demand': round(mean_qty, 2), # C 類以均值展示
                    'Final_Demand': round(mean_qty, 2), 
                    'Target_Stock': round(target_stock, 2), 
                    'Suggested_Order': round(suggested_order, 2),
                    'Final_Order': round(final_order, 2) 
                })
                
            c_forecast_df = pd.DataFrame(c_results)
            # 將 A/B 類的 AI 預測結果與 C 類結果合併
            forecast_df = pd.concat([forecast_df, c_forecast_df], ignore_index=True)

        
        # 匯出 CSV
        forecast_df.sort_values(by='Final_Order', ascending=False).to_csv(output_path, index=False, encoding='utf-8-sig')
        
        end_time = time.time()
        print(f"\n✅ 系統分析完成！總耗時: {round(end_time - start_time, 2)} 秒。")
        print(f"📂 報告已匯出至: {output_path}")
            
    except Exception as e:
        print(f"\n❌ 執行時發生中斷: {e}")

if __name__ == "__main__":
    main()