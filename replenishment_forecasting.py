import pandas as pd
import numpy as np
from prophet import Prophet
import os

def calculate_category_seasonal_indices(sales_df, stock_df):
    """
    計算類別層級的季節性係數，用於輔助單品預測
    """
    sales_df = sales_df.copy()
    sales_df['rDate'] = pd.to_datetime(sales_df['rDate'])
    
    # 合併類別資訊
    df_with_cat = sales_df.merge(stock_df[['GoodsID', 'Category']], on='GoodsID', how='left')
    
    # 彙整到 類別 + 月份
    cat_monthly = df_with_cat.groupby(['Category', pd.Grouper(key='rDate', freq='MS')])['TotalQty'].sum().reset_index()
    
    # 計算每個類別的長期平均月銷量
    cat_avg = cat_monthly.groupby('Category')['TotalQty'].mean().rename('Cat_Avg_Qty').reset_index()
    
    # 計算每個月的係數 (Seasonal Index)
    cat_indices = cat_monthly.merge(cat_avg, on='Category')
    cat_indices['Seasonal_Index'] = cat_indices['TotalQty'] / cat_indices['Cat_Avg_Qty']
    cat_indices['Month'] = cat_indices['rDate'].dt.month
    
    # 建立查詢表: {Category: {Month: Index}}
    index_map = cat_indices.groupby(['Category', 'Month'])['Seasonal_Index'].mean().to_dict()
    return index_map

def analyze_abc_xyz(stock_df, sales_df):
    """
    執行 ABC-XYZ 分類與新品標記邏輯
    """
    sales_df['rDate'] = pd.to_datetime(sales_df['rDate'])
    last_date = sales_df['rDate'].max()
    start_12m = last_date - pd.DateOffset(months=12)
    
    # 1. ABC 分析 (基於最近 12 個月利潤)
    df_12m = sales_df[sales_df['rDate'] >= start_12m].copy()
    summary_12m = df_12m.groupby('GoodsID').agg({'TotalQty': 'sum', 'TotalAmt': 'sum'}).reset_index()
    
    # 這裡簡化合併與成本邏輯 (延用你之前的向量化架構)
    merged = pd.merge(summary_12m, stock_df, on='GoodsID', how='outer').fillna(0)
    merged['TotalProfit'] = merged['TotalAmt'] - (merged['LastInCost'] * merged['TotalQty'])
    
    # 排序標記 ABC
    merged = merged.sort_values('TotalProfit', ascending=False)
    merged['CumProfit'] = merged['TotalProfit'].cumsum()
    total_prof = merged['TotalProfit'].sum()
    merged['ProfitRatio'] = merged['CumProfit'] / total_prof if total_prof > 0 else 0
    
    conditions = [(merged['ProfitRatio'] <= 0.7), (merged['ProfitRatio'] <= 0.9)]
    merged['ABC_Class'] = np.select(conditions, ['A', 'B'], default='C')

    # 2. XYZ 分析 (基於 23 個月波動)
    monthly_matrix = sales_df.groupby(['GoodsID', pd.Grouper(key='rDate', freq='MS')])['TotalQty'].sum().unstack(fill_value=0)
    stats = pd.DataFrame(index=monthly_matrix.index)
    stats['Mean'] = monthly_matrix.mean(axis=1)
    stats['CV'] = np.where(stats['Mean'] > 0, monthly_matrix.std(axis=1) / stats['Mean'], 9.99)
    
    xyz_cond = [(stats['CV'] <= 0.5), (stats['CV'] <= 1.0)]
    stats['XYZ_Class'] = np.select(xyz_cond, ['X', 'Y'], default='Z')
    
    # 3. 新品標記
    first_sale = sales_df.groupby('GoodsID')['rDate'].min().reset_index()
    first_sale['Age'] = (last_date.year - first_sale['rDate'].dt.year) * 12 + (last_date.month - first_sale['rDate'].dt.month)
    
    # 合併所有結果
    final_df = merged.merge(stats[['CV', 'XYZ_Class', 'Mean']], left_on='GoodsID', right_index=True, how='left')
    final_df = final_df.merge(first_sale[['GoodsID', 'Age']], on='GoodsID', how='left').fillna({'Age': 99, 'XYZ_Class': 'Z', 'CV': 9.99})
    
    is_new = (final_df['Age'] < 4)
    final_df.loc[is_new, 'ABC_Class'] = 'New'
    final_df.loc[is_new, 'XYZ_Class'] = 'New'
    
    return final_df

def predict_demand_robust(skus_to_predict, sales_df, stock_df):
    """
    整合類別加成與防爆邏輯的預測引擎
    """
    cat_index_map = calculate_category_seasonal_indices(sales_df, stock_df)
    last_date = pd.to_datetime(sales_df['rDate']).max()
    next_month = (last_date + pd.DateOffset(months=1)).month
    
    predictions = []
    
    for _, item in skus_to_predict.iterrows():
        gid = item['GoodsID']
        abc_xyz = f"{item['ABC_Class']}{item['XYZ_Class']}"
        total_qty_year = item['TotalQty']
        
        item_sales = sales_df[sales_df['GoodsID'] == gid].sort_values('rDate')
        
        # --- [邏輯 1] 基礎預測 (Base Demand) ---
        if 'Y' in abc_xyz:
            # 季節性產品用 Prophet
            try:
                m_df = item_sales.groupby(pd.Grouper(key='rDate', freq='MS'))['TotalQty'].sum().reset_index()
                m_df.columns = ['ds', 'y']
                m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
                m.fit(m_df)
                future = m.make_future_dataframe(periods=1, freq='MS')
                base_pred = max(0, m.predict(future).iloc[-1]['yhat'])
            except:
                base_pred = item['Mean']
        elif 'X' in abc_xyz:
            base_pred = item_sales.groupby(pd.Grouper(key='rDate', freq='MS'))['TotalQty'].sum().tail(3).mean()
        else:
            # Z 類或數據不足，採用中位數預防極端波動
            m_df = item_sales.groupby(pd.Grouper(key='rDate', freq='MS'))['TotalQty'].sum().tail(6)
            base_pred = m_df.median() if not m_df.empty else 0

        # --- [邏輯 2] 類別加成與天花板 (Boost & Cap) ---
        raw_boost = cat_index_map.get((item['Category'], next_month), 1.0)
        # 防爆：加成係數最高不超過 2.0 倍
        boost_factor = min(2.0, raw_boost) if (item['ABC_Class'] == 'New' or item['XYZ_Class'] == 'Z') else 1.0
        final_demand = base_pred * max(1.0, boost_factor)

        # --- [邏輯 3] 安全庫存防禦牆 (Safety Cap) ---
        # 防爆：安全係數 A 類上限 0.5，其餘 0.3
        if item['ABC_Class'] == 'A':
            safety_ratio = min(0.5, item['CV'] * 0.5)
        else:
            safety_ratio = min(0.3, item['CV'] * 0.5)
            
        safety_buffer = final_demand * safety_ratio
        target_stock = final_demand + safety_buffer
        
        # --- [邏輯 4] 現實檢查 (Reality Check) ---
        # 如果目標庫存高於過去一年的總銷量，強制修正 (針對非新品)
        if item['ABC_Class'] != 'New' and target_stock > total_qty_year and total_qty_year > 0:
            target_stock = total_qty_year * 0.5

        predictions.append({
            'GoodsID': gid,
            'Name': item['Name'],
            'ABC_XYZ': abc_xyz,
            'Base_Demand': round(base_pred, 2),
            'Cat_Boost': round(boost_factor, 2),
            'Final_Demand': round(final_demand, 2),
            'Safety_Buffer': round(safety_buffer, 2),
            'Target_Stock': round(target_stock, 2)
        })
        
    return pd.DataFrame(predictions)

if __name__ == "__main__":
    print("🚀 啟動零售智慧分析與補貨系統...")
    
    # 讀取資料
    df_stock = pd.read_csv("data/processed/DetailGoodsStockToday.csv")
    df_sales = pd.read_parquet("data/processed/vw_GoodsDailySales_cache.parquet")
    
    # 1. 執行分類 (ABC-XYZ)
    print("📊 正在進行 ABC-XYZ 分類與新品識別...")
    analysis_df = analyze_abc_xyz(df_stock, df_sales)
    
    # 2. 執行預測 (僅針對 A/B 類與新品，節省算力)
    print("🔮 正在執行防爆預測與類別加成計算...")
    target_skus = analysis_df[analysis_df['Supplier'].isin(['海王地毯'])]
    forecast_df = predict_demand_robust(target_skus, df_sales, df_stock)
    
    # 3. 合併庫存計算補貨
    final_output = forecast_df.merge(analysis_df[['GoodsID', 'CurrStock']], on='GoodsID')
    final_output['Suggested_Order'] = (final_output['Target_Stock'] - final_output['CurrStock']).clip(lower=0)
    
    # 4. 存檔
    final_output.to_csv("data/insights/final_inventory_plan.csv", index=False, encoding='utf-8-sig')
    print(f"✅ 補貨計畫已完成，結果儲存至 data/insights/final_inventory_plan.csv")