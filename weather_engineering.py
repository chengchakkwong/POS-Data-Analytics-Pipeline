import pandas as pd
import requests
import io
import os
from datetime import datetime

class WeatherFeatureBuilder:
    """
    專門用於處理香港天文台 (HKO) 開放數據 API 的氣溫數據。
    
    業務場景：針對位於元朗的家品零售店，鎖定 'YLP' (元朗公園) 氣象站數據。
    元朗區通常日夜溫差較大，對季節性家品（如暖風機、保暖墊）銷售有顯著影響。
    """

    # 固定鎖定元朗公園
    TARGET_STATION = 'YLP' 

    @staticmethod
    def get_daily_temperature(start_year=2023, end_year=2024, save_dir='data/external_factors'):
        """
        從 HKO API 獲取元朗公園 (YLP) 的每日平均氣溫數據。

        Args:
            start_year (int): 篩選開始年份
            end_year (int): 篩選結束年份
            save_dir (str): 儲存路徑

        Returns:
            pd.DataFrame: 包含 date, mean_temp, temp_ma_7, temp_diff 的數據
        """
        
        station_code = WeatherFeatureBuilder.TARGET_STATION
        
        # 1. 構建 API URL
        # HKO API 直接返回 CSV 格式，包含該站點的所有歷史數據
        url = f"https://data.weather.gov.hk/weatherAPI/opendata/opendata.php?dataType=CLMTEMP&rformat=csv&station={station_code}"
        
        print(f"正在下載元朗店周邊氣溫數據 (站點: {station_code})...")
        
        try:
            # 2. 下載數據
            response = requests.get(url, timeout=10)
            response.raise_for_status() # 檢查連線是否成功
            
            # 3. 讀取 CSV
            csv_content = response.content.decode('utf-8')
            
            # 使用 skiprows=2 跳過前兩行 metadata
            df = pd.read_csv(io.StringIO(csv_content), skiprows=2)
            
            # --- 數據清理 ---
            
            df.columns = ['year', 'month', 'day', 'mean_temp', 'completeness']
            
            # 過濾掉 CSV 尾部的備註文字 (Footer)
            df['year'] = pd.to_numeric(df['year'], errors='coerce')
            df = df.dropna(subset=['year']) 
            
            # 確保年、月、日是整數
            df['year'] = df['year'].astype(int)
            df['month'] = pd.to_numeric(df['month'], errors='coerce').fillna(1).astype(int)
            df['day'] = pd.to_numeric(df['day'], errors='coerce').fillna(1).astype(int)
            
            # 4. 建立日期欄位
            df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
            
            # 5. 處理缺失值
            df['mean_temp'] = pd.to_numeric(df['mean_temp'], errors='coerce')
            df['mean_temp'] = df['mean_temp'].ffill()
            
            # --- 特徵工程 (順序優化) ---
            # [關鍵修改] 
            # 我們在篩選年份 "之前" 先計算特徵。
            # 這樣 start_year 的 1月1日 就能用到去年的 12月 數據來計算 7天平均，
            # 避免了前幾天出現 NaN (空值) 的情況。
            
            # [特徵 A] 7天移動平均
            df['temp_ma_7'] = df['mean_temp'].rolling(window=7).mean() 
            
            # [特徵 B] 溫差 (今日 - 昨日)
            df['temp_diff'] = df['mean_temp'].diff() 

            # 6. 篩選時間範圍 (現在可以放心地切了)
            mask = (df['year'] >= start_year) & (df['year'] <= end_year)
            
            # 選取需要的欄位
            df_final = df.loc[mask, ['date', 'mean_temp', 'temp_ma_7', 'temp_diff']].copy()
            
            # 再次檢查 NaN (理論上不應該有了，除非 start_year 是該站點歷史第一年)
            df_final.fillna(method='bfill', inplace=True) 

            # 8. 儲存
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'yuen_long_weather_{start_year}_{end_year}.csv')
            df_final.to_csv(save_path, index=False)
            
            print(f"元朗天氣特徵處理完成！已儲存至: {save_path}")
            print(f"數據覆蓋範圍: {df_final['date'].min().date()} 至 {df_final['date'].max().date()}")
            
            return df_final

        except Exception as e:
            print(f"獲取天氣數據時發生錯誤: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

# --- 測試代碼 ---
if __name__ == "__main__":
    # 測試抓取數據
    df_weather = WeatherFeatureBuilder.get_daily_temperature(
        start_year=2023, 
        end_year=2024
    )
    
    if not df_weather.empty:
        print("\n--- 元朗氣溫數據預覽 (檢查 1月1日是否有值) ---")
        print(df_weather.head(10))
        
        print("\n--- 驟降溫日子檢查 (temp_diff < -3) ---")
        cold_snaps = df_weather[df_weather['temp_diff'] < -3]
        if not cold_snaps.empty:
            print(cold_snaps.head())
    else:
        print("DataFrame 為空，請檢查錯誤訊息。")
