import pandas as pd
import numpy as np
from prophet import Prophet
import os

def calculate_category_seasonal_indices(sales_df, stock_df):
    """
    計算每個類別的季節性加成係數 (以月為單位)
    """
    # 確保日期格式
    sales_df['rDate'] = pd.to_datetime(sales_df['rDate'])
    
    # 合併類別資訊
    df_with_cat = sales_df.merge(stock_df[['GoodsID', 'Category']], on='GoodsID', how='left')
    
    # 1. 彙總到 類別 + 月份 的總銷量
    cat_monthly = df_with_cat.groupby(['Category', pd.Grouper(key='rDate', freq='MS')])['TotalQty'].sum().reset_index()
    
    # 2. 計算每個類別的長期平均月銷量 (Benchmark)
    cat_avg = cat_monthly.groupby('Category')['TotalQty'].mean().rename('Cat_Avg_Qty').reset_index()
    
    # 3. 計算每個月相對於平均的係數 (Seasonal Index)
    cat_indices = cat_monthly.merge(cat_avg, on='Category')
    cat_indices['Seasonal_Index'] = cat_indices['TotalQty'] / cat_indices['Cat_Avg_Qty']
    
    # 提取月份資訊供後續比對
    cat_indices['Month'] = cat_indices['rDate'].dt.month
    
    # 整理成方便查詢的字典格式: {Category: {Month: Index}}
    index_map = cat_indices.groupby(['Category', 'Month'])['Seasonal_Index'].mean().to_dict()
    
    return index_map

def predict_demand_logic(skus_to_predict, sales_df, stock_df, forecast_months=1):
    """
    具備「類別加成」與「新品保護」的綜合預測邏輯
    """
    # 預先計算類別季節係數
    cat_index_map = calculate_category_seasonal_indices(sales_df, stock_df)
    
    # 判斷下個月是幾月 (用於提取類別係數)
    last_date = pd.to_datetime(sales_df['rDate']).max()
    next_month = (last_date + pd.DateOffset(months=1)).month
    
    predictions = []
    
    for _, item in skus_to_predict.iterrows():
        gid = item['GoodsID']
        cat = item['Category']
        abc_class = item['ABC_Class']
        xyz_class = item['XYZ_Class']
        cv = item['CV']
        
        # 取得該 SKU 的歷史數據
        item_sales = sales_df[sales_df['GoodsID'] == gid].sort_values('rDate')
        
        # --- 第一階段：基礎預測量 (Base Prediction) ---
        base_pred = 0
        
        if abc_class == 'New':
            # 新品策略：最近 4 週的週平均 * 4 (Run-rate)
            recent_4_weeks = item_sales[item_sales['rDate'] >= (last_date - pd.Timedelta(weeks=4))]
            base_pred = (recent_4_weeks['TotalQty'].sum() / 4) * 4 if not recent_4_weeks.empty else item['Mean_Monthly_Qty']
            
        elif xyz_class == 'Y':
            # 季節規律策略：使用 Prophet
            try:
                m_df = item_sales.groupby(pd.Grouper(key='rDate', freq='MS'))['TotalQty'].sum().reset_index()
                m_df.columns = ['ds', 'y']
                if len(m_df) >= 12:
                    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
                    m.fit(m_df)
                    future = m.make_future_dataframe(periods=forecast_months, freq='MS')
                    forecast = m.predict(future)
                    base_pred = max(0, forecast.iloc[-1]['yhat'])
                else:
                    base_pred = m_df['y'].tail(3).mean()
            except:
                base_pred = item['Mean_Monthly_Qty']
                
        elif xyz_class == 'X':
            # 穩定型策略：最近 3 個月平均
            m_df = item_sales.groupby(pd.Grouper(key='rDate', freq='MS'))['TotalQty'].sum().tail(3)
            base_pred = m_df.mean() if not m_df.empty else item['Mean_Monthly_Qty']
            
        else: # Z 類 (隨機)
            # 隨機型策略：最近 6 個月中位數 (抗擾動)
            m_df = item_sales.groupby(pd.Grouper(key='rDate', freq='MS'))['TotalQty'].sum().tail(6)
            base_pred = m_df.median() if not m_df.empty else 0

        # --- 第二階段：類別加成 (Category Boosting) ---
        # 取得該類別在下個月的季節係數
        boost_factor = cat_index_map.get((cat, next_month), 1.0)
        
        # 邏輯：如果是 AZ, BZ 或 New 商品，因為個體規律不明顯，強制參考類別的大趨勢
        if (abc_class == 'New') or (xyz_class == 'Z' and abc_class in ['A', 'B']):
            final_pred = base_pred * max(1.0, boost_factor) # 若係數小於 1 (淡季) 則維持現狀
        else:
            # AX, AY 商品已經有自己的規律，不重複加成
            final_pred = base_pred

        # --- 第三階段：安全庫存 (Safety Stock Buffer) ---
        # 針對「新品」設置 30% 的保護天花板，防止 CV 誤導補貨過多
        if abc_class == 'New':
            safety_ratio = min(0.3, cv * 0.5)
        else:
            safety_ratio = cv * 0.5
            
        safety_buffer = final_pred * safety_ratio
        
        predictions.append({
            'GoodsID': gid,
            'Name': item['Name'],
            'Category': cat,
            'ABC_XYZ': f"{abc_class}{xyz_class}",
            'Base_Demand': round(base_pred, 2),
            'Cat_Boost': round(boost_factor, 2),
            'Final_Demand': round(final_pred, 2),
            'Safety_Buffer': round(safety_buffer, 2),
            'Target_Stock': round(final_pred + safety_buffer, 2)
        })
        
    return pd.DataFrame(predictions)

if __name__ == "__main__":
    # 路徑設定
    path_analysis = "data/insights/abc_xyz_analysis.csv"
    path_sales = "data/processed/vw_GoodsDailySales_cache.parquet"
    path_stock = "data/processed/DetailGoodsStockToday.csv"
    output_path = "data/insights/replenishment_forecast_v2.csv"
    
    print("🚀 啟動『類別加成』智慧補貨引擎...")
    
    try:
        if os.path.exists(path_analysis):
            analysis_df = pd.read_csv(path_analysis)
            sales_df = pd.read_parquet(path_sales)
            stock_df = pd.read_csv(path_stock)
            
            # 挑選需要預測的對象 (例如 A 類, B 類與新品)
            target_skus = analysis_df[analysis_df['ABC_Class'].isin(['A', 'B', 'New'])]
            
            print(f"📊 正在為 {len(target_skus)} 個核心商品計算補貨量...")
            
            # 執行預測
            forecast_results = predict_demand_logic(target_skus, sales_df, stock_df)
            
            # 合併庫存計算實際補貨量
            final_df = forecast_results.merge(analysis_df[['GoodsID', 'CurrStock']], on='GoodsID')
            final_df['Suggested_Order'] = (final_df['Target_Stock'] - final_df['CurrStock']).clip(lower=0)
            
            # 排序與儲存
            final_df = final_df.sort_values(by='Suggested_Order', ascending=False)
            final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            
            print(f"✅ 補貨計畫完成！結果已儲存至: {output_path}")
            
            # 輸出幾個範例檢查
            print("\n--- 預測範例 (前 5 筆建議採購) ---")
            print(final_df[['Name', 'ABC_XYZ', 'Base_Demand', 'Cat_Boost', 'Suggested_Order']].head())
            
    except Exception as e:
        print(f"❌ 預測失敗: {e}")