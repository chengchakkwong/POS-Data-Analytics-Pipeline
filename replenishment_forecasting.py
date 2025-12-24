import pandas as pd
import numpy as np
from prophet import Prophet
import os

def predict_demand_logic(skus_to_predict, sales_df, forecast_months=1):
    """
    根據不同的 ABC-XYZ 標籤應用不同的預測邏輯
    """
    predictions = []
    
    # 為了計算方便，先將銷售轉換為月度格式
    sales_df['rDate'] = pd.to_datetime(sales_df['rDate'])
    
    for _, item in skus_to_predict.iterrows():
        gid = item['GoodsID']
        strategy = item['Strategy']
        abc_xyz = f"{item['ABC_Class']}{item['XYZ_Class']}"
        
        # 取得該 SKU 的歷史數據
        item_sales = sales_df[sales_df['GoodsID'] == gid].sort_values('rDate')
        
        # --- 策略 A: 針對 AY, BY (季節性模型 Prophet) ---
        if abc_xyz in ['AY', 'BY']:
            try:
                # 準備 Prophet 格式
                m_df = item_sales.groupby(pd.Grouper(key='rDate', freq='MS'))['TotalQty'].sum().reset_index()
                m_df.columns = ['ds', 'y']
                
                # 至少需要 12 個月才能跑季節性
                if len(m_df) >= 12:
                    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
                    m.fit(m_df)
                    future = m.make_future_dataframe(periods=forecast_months, freq='MS')
                    forecast = m.predict(future)
                    pred_val = max(0, forecast.iloc[-1]['yhat'])
                else:
                    # 數據不足則降級到移動平均
                    pred_val = m_df['y'].tail(3).mean()
            except:
                pred_val = 0
                
        # --- 策略 B: 針對 AX, BX, CX (穩定型：移動平均) ---
        elif 'X' in abc_xyz:
            # 取最近三個月的平均
            m_df = item_sales.groupby(pd.Grouper(key='rDate', freq='MS'))['TotalQty'].sum().tail(3)
            pred_val = m_df.mean() if not m_df.empty else 0
            
        # --- 策略 C: 針對 New (新品：Run-rate 加上 10% 增長) ---
        elif item['ABC_Class'] == 'New':
            # 取最近 4 週的週平均再乘以 4
            recent_4_weeks = item_sales[item_sales['rDate'] >= (item_sales['rDate'].max() - pd.Timedelta(weeks=4))]
            weekly_avg = recent_4_weeks['TotalQty'].sum() / 4
            pred_val = weekly_avg * 4 * 1.1 # 預期新品成長 10%
            
        # --- 策略 D: 針對 Z (隨機型：中位數預測) ---
        else:
            m_df = item_sales.groupby(pd.Grouper(key='rDate', freq='MS'))['TotalQty'].sum().tail(6)
            pred_val = m_df.median() if not m_df.empty else 0

        # --- 計算安全庫存 (Safety Stock) ---
        # 公式：Z-score * Std * sqrt(LeadTime) 
        # 這裡簡單化：利用 CV 值，CV 越高，安全庫存加成越高

        # --- 修改前的安全庫存邏輯 ---
        # safety_buffer = pred_val * (item['CV'] * 0.5)# CV 越大，buffer 越大

        # --- 修改後的優化邏輯 ---
        if item['ABC_Class'] == 'New':
            # 新品的 CV 不可靠，強制將安全庫存係數限制在 0.3 (即 30% 緩衝)
            safety_coeff = min(0.3, item['CV'] * 0.5)
        else:
            # 老商品維持原樣
            safety_coeff = item['CV'] * 0.5

        safety_buffer = pred_val * safety_coeff
        predictions.append({
            'GoodsID': gid,
            'Name': item['Name'],
            'Predicted_Demand': round(pred_val, 2),
            'Safety_Stock_Buffer': round(safety_buffer, 2),
            'Target_Inventory': round(pred_val + safety_buffer, 2)
        })
        
    return pd.DataFrame(predictions)

if __name__ == "__main__":
    # 讀取上一階段的分類結果
    input_analysis = "data/insights/abc_xyz_analysis.csv"
    input_sales = "data/processed/vw_GoodsDailySales_cache.parquet"
    
    if os.path.exists(input_analysis):
        analysis_res = pd.read_csv(input_analysis)
        sales_data = pd.read_parquet(input_sales)
        
        # 為了演示，我們只針對前 50 個重要商品 (A 類) 進行預測
        top_skus = analysis_res[analysis_res['ABC_Class'].isin(['A'])].head(50)
        
        print(f"🚀 開始為 {len(top_skus)} 個核心商品產出預測...")
        result = predict_demand_logic(top_skus, sales_data)
        
        # 合併現有庫存計算補貨量
        # 建議補貨量 = Target_Inventory - CurrStock
        final_report = pd.merge(result, analysis_res[['GoodsID', 'CurrStock']], on='GoodsID')
        final_report['Suggested_Order'] = (final_report['Target_Inventory'] - final_report['CurrStock']).clip(lower=0)
        
        print(final_report[['Name', 'Predicted_Demand', 'Target_Inventory', 'CurrStock', 'Suggested_Order']].head(10))
        
        # 儲存結果
        final_report.to_csv("data/insights/replenishment_forecast.csv", index=False, encoding='utf-8-sig')