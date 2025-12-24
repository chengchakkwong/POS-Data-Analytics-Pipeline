import pandas as pd
import numpy as np
from prophet import Prophet
from joblib import Parallel, delayed
import os
import time
import warnings

# 忽略 Prophet 產生的冗餘訊息
warnings.filterwarnings('ignore')
import logging
logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)

# --- 1. 數據分類與預處理模組 ---

def calculate_category_seasonal_indices(sales_df, stock_df):
    """
    計算類別層級的季節性係數 (Category Boosting)
    用於輔助單品數據不穩定的商品 (AZ/New)
    """
    sales_df = sales_df.copy()
    sales_df['rDate'] = pd.to_datetime(sales_df['rDate'])
    
    # 關聯類別資訊
    df_with_cat = sales_df.merge(stock_df[['GoodsID', 'Category']], on='GoodsID', how='left')
    
    # 按類別與月份彙總
    cat_monthly = df_with_cat.groupby(['Category', pd.Grouper(key='rDate', freq='MS')])['TotalQty'].sum().reset_index()
    
    # 計算每個類別的長期平均月銷量 (基準值)
    cat_avg = cat_monthly.groupby('Category')['TotalQty'].mean().rename('Cat_Avg_Qty').reset_index()
    
    # 計算季節係數 = 當月銷量 / 平均月銷量
    cat_indices = cat_monthly.merge(cat_avg, on='Category')
    cat_indices['Seasonal_Index'] = cat_indices['TotalQty'] / cat_indices['Cat_Avg_Qty']
    cat_indices['Month'] = cat_indices['rDate'].dt.month
    
    # 建立查詢表：{ (類別, 月份): 係數 }
    index_map = cat_indices.groupby(['Category', 'Month'])['Seasonal_Index'].mean().to_dict()
    return index_map

def perform_abc_xyz_analysis(stock_df, sales_df):
    """
    執行 ABC-XYZ 矩陣分析與新品識別
    ABC 基於最近 12 個月利潤
    XYZ 基於所有 23 個月波動
    """
    sales_df['rDate'] = pd.to_datetime(sales_df['rDate'])
    last_date = sales_df['rDate'].max()
    start_12m = last_date - pd.DateOffset(months=12)
    
    # --- ABC 分析 (最近 12 個月) ---
    df_12m = sales_df[sales_df['rDate'] >= start_12m].copy()
    summary_12m = df_12m.groupby('GoodsID').agg({
        'TotalQty': 'sum', 
        'TotalAmt': 'sum'
    }).reset_index()
    
    # 合併庫存主檔計算利潤
    merged = pd.merge(stock_df, summary_12m, on='GoodsID', how='left').fillna(0)
    # 使用你之前定義的簡單成本邏輯或 AdjustedCost
    merged['TotalProfit'] = merged['TotalAmt'] - (merged['LastInCost'] * merged['TotalQty'])
    
    # 排序與計算累計佔比
    merged = merged.sort_values('TotalProfit', ascending=False)
    total_prof = merged['TotalProfit'].sum()
    if total_prof > 0:
        merged['ProfitRatio'] = merged['TotalProfit'].cumsum() / total_prof
    else:
        merged['ProfitRatio'] = 1.0
        
    conditions_abc = [(merged['ProfitRatio'] <= 0.7), (merged['ProfitRatio'] <= 0.9)]
    merged['ABC_Class'] = np.select(conditions_abc, ['A', 'B'], default='C')

    # --- XYZ 分析 (所有 23 個月) ---
    monthly_matrix = sales_df.groupby(['GoodsID', pd.Grouper(key='rDate', freq='MS')])['TotalQty'].sum().unstack(fill_value=0)
    stats = pd.DataFrame(index=monthly_matrix.index)
    stats['Mean_Monthly'] = monthly_matrix.mean(axis=1)
    stats['CV'] = np.where(stats['Mean_Monthly'] > 0, monthly_matrix.std(axis=1) / stats['Mean_Monthly'], 9.99)
    
    conditions_xyz = [(stats['CV'] <= 0.5), (stats['CV'] <= 1.0)]
    stats['XYZ_Class'] = np.select(conditions_xyz, ['X', 'Y'], default='Z')
    
    # --- 新品識別 ---
    first_sale = sales_df.groupby('GoodsID')['rDate'].min().reset_index()
    first_sale['Month_Age'] = (last_date.year - first_sale['rDate'].dt.year) * 12 + (last_date.month - first_sale['rDate'].dt.month)
    
    # 彙整分類結果
    analysis_df = merged.merge(stats[['CV', 'XYZ_Class', 'Mean_Monthly']], left_on='GoodsID', right_index=True, how='left')
    analysis_df = analysis_df.merge(first_sale[['GoodsID', 'Month_Age']], on='GoodsID', how='left').fillna({
        'Month_Age': 99, 'XYZ_Class': 'Z', 'CV': 9.99, 'Mean_Monthly': 0
    })
    
    # 強制標記上架 < 4 個月的為 New
    is_new = (analysis_df['Month_Age'] < 4)
    analysis_df.loc[is_new, 'ABC_Class'] = 'New'
    analysis_df.loc[is_new, 'XYZ_Class'] = 'New'
    
    return analysis_df

# --- 2. 智慧補貨預測核心 (並行運算單元) ---

def run_single_sku_forecast(item, sales_df, next_month, cat_index_map):
    """
    單一 SKU 預測 worker：包含策略分流、類別加成與防爆限制
    """
    gid = item['GoodsID']
    abc = item['ABC_Class']
    xyz = item['XYZ_Class']
    cv = item['CV']
    cat = item['Category']
    total_qty_year = item['TotalQty']
    
    # 提取該商品銷售紀錄
    item_sales = sales_df[sales_df['GoodsID'] == gid].sort_values('rDate')
    
    # --- [步驟 1] 基礎預測策略 (Base Demand) ---
    base_pred = 0
    try:
        if abc == 'New':
            # 新品採用最近 4 週 Run-rate
            recent_data = item_sales[item_sales['rDate'] >= (item_sales['rDate'].max() - pd.Timedelta(weeks=4))]
            base_pred = (recent_data['TotalQty'].sum() / 4) * 4 if not recent_data.empty else item['Mean_Monthly']
        
        elif xyz == 'Y':
            # 季節性商品採用 Prophet
            m_df = item_sales.groupby(pd.Grouper(key='rDate', freq='MS'))['TotalQty'].sum().reset_index()
            m_df.columns = ['ds', 'y']
            if len(m_df) >= 12:
                m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False, uncertainty_samples=0)
                m.fit(m_df)
                future = m.make_future_dataframe(periods=1, freq='MS')
                base_pred = max(0, m.predict(future).iloc[-1]['yhat'])
            else:
                base_pred = m_df['y'].tail(3).mean()
        
        elif xyz == 'X':
            # 穩定型採用最近 3 個月平均
            base_pred = item_sales.groupby(pd.Grouper(key='rDate', freq='MS'))['TotalQty'].sum().tail(3).mean()
        
        else:
            # 隨機型採用最近 6 個月中位數 (抗離群值)
            m_df = item_sales.groupby(pd.Grouper(key='rDate', freq='MS'))['TotalQty'].sum().tail(6)
            base_pred = m_df.median() if not m_df.empty else 0
    except:
        base_pred = item['Mean_Monthly']

    # --- [步驟 2] 類別加成與防爆 (Category Boosting & Boost Cap) ---
    raw_boost = cat_index_map.get((cat, next_month), 1.0)
    # 防爆 1：加成倍數最高限制 2.0 倍
    boost_factor = min(2.0, raw_boost) if (abc == 'New' or xyz == 'Z') else 1.0
    final_demand = base_pred * max(1.0, boost_factor)

    # --- [步驟 3] 安全庫存防禦牆 (Safety Buffer & Safety Cap) ---
    # 防爆 2：安全庫存係數 A類上限 0.5, 其餘 0.3
    safety_ratio_limit = 0.5 if abc == 'A' else 0.3
    safety_ratio = min(safety_ratio_limit, cv * 0.5)
    safety_buffer = final_demand * safety_ratio
    
    target_stock = final_demand + safety_buffer
    
    # --- [步驟 4] 現實檢查 (Reality Check) ---
    # 防爆 3：目標庫存不應超過該商品去年總銷量的一半 (針對非新品)
    if abc != 'New' and total_qty_year > 0 and target_stock > total_qty_year:
        target_stock = total_qty_year * 0.5

    return {
        'GoodsID': gid,
        'Name': item['Name'],
        'Category': cat,
        'ABC_XYZ': f"{abc}{xyz}",
        'Base_Demand': round(base_pred, 2),
        'Cat_Boost': round(boost_factor, 2),
        'Final_Demand': round(final_demand, 2),
        'Safety_Buffer': round(safety_buffer, 2),
        'Target_Stock': round(target_stock, 2),
        'CurrStock': item['CurrStock']
    }

# --- 3. 主程序執行邏輯 ---

def main():
    # 設定路徑
    input_stock = "data/processed/DetailGoodsStockToday.csv"
    input_sales = "data/processed/vw_GoodsDailySales_cache.parquet"
    output_path = "data/insights/final_inventory_plan.csv"
    
    print("🚀 啟動零售智慧分析系統...")
    
    if not os.path.exists(input_stock) or not os.path.exists(input_sales):
        print("❌ 錯誤: 找不到輸入檔案，請確認路徑。")
        return

    # 載入數據
    df_stock = pd.read_csv(input_stock)
    df_sales = pd.read_parquet(input_sales)
    
    # 1. 執行 ABC-XYZ 分類 (向量化運算，極快)
    print("📊 正在計算 ABC-XYZ 矩陣與識別新產品...")
    analysis_df = perform_abc_xyz_analysis(df_stock, df_sales)
    
    # 2. 計算類別季節係數
    print("⚙️ 正在分析類別季節規律...")
    cat_index_map = calculate_category_seasonal_indices(df_sales, df_stock)
    
    # 3. 並行執行補貨預測 (針對 A, B 類與 New 類進行重點預測)
    print("🔮 啟動多核心並行預測引擎 (處理 A/B/New 商品)...")
    target_skus = analysis_df[analysis_df['ABC_Class'].isin(['A', 'B', 'New'])]
    
    last_date = pd.to_datetime(df_sales['rDate']).max()
    next_month = (last_date + pd.DateOffset(months=1)).month
    
    start_time = time.time()
    
    # 使用所有可用 CPU 核心並行運算
    results = Parallel(n_jobs=-1)(
        delayed(run_single_sku_forecast)(row, df_sales, next_month, cat_index_map) 
        for _, row in target_skus.iterrows()
    )
    
    forecast_df = pd.DataFrame(results)
    
    # 4. 計算建議採購量
    forecast_df['Suggested_Order'] = (forecast_df['Target_Stock'] - forecast_df['CurrStock']).clip(lower=0)
    
    # 5. 整合與排序輸出
    forecast_df = forecast_df.sort_values(by=['ABC_XYZ', 'Suggested_Order'], ascending=[True, False])
    forecast_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    end_time = time.time()
    print(f"✅ 分析完成！耗時: {round(end_time - start_time, 2)} 秒")
    print(f"📊 最終計畫已儲存至: {output_path}")

    # 輸出簡易統計
    print("\n--- 預測統計摘要 ---")
    print(f"總處理品項數: {len(target_skus)}")
    print(f"建議採購總項數: {len(forecast_df[forecast_df['Suggested_Order'] > 0])}")
    print(f"平均單品加成係數: {round(forecast_df['Cat_Boost'].mean(), 2)}")

if __name__ == "__main__":
    main()