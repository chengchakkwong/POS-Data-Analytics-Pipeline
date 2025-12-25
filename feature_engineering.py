import pandas as pd
import holidays
import numpy as np
import os
from datetime import datetime, timedelta

class ExternalFeatureBuilder:
    """
    用於構建零售預測所需的外部特徵（假期、日曆效應）。
    針對 LightGBM 模型優化。
    """

    @staticmethod
    def _get_holiday_category(name):
        """
        輔助函數：將假期名稱分類。
        邏輯：根據關鍵字將假期分為 'Traditional' (傳統節日) 或 'Public' (一般公眾假期)。
        這有助於區分送禮型節日 vs 休閒型節日。
        """
        if pd.isna(name):
            return 'None'
        
        # 關鍵字映射 (可根據實際業務需求調整)
        traditional_keywords = [
            'Lunar New Year', 'Chinese New Year', 'Mid-Autumn', 
            'Dragon Boat', 'Chung Yeung', 'Qingming', 'Easter', 'Christmas'
        ]
        
        # 如果名稱包含上述關鍵字，視為傳統/重要節日 (通常伴隨消費高峰)
        for keyword in traditional_keywords:
            if keyword.lower() in name.lower():
                return 'Traditional_Gift'
        
        # 其他視為一般公眾假期 (Labor Day, SAR Day, National Day etc.)
        return 'Public_Holiday'

    @staticmethod
    def get_hk_holidays(start_year: int = 2023, end_year: int = None, save_dir: str = 'data/external_factors') -> pd.DataFrame:
        """
        生成包含完整日曆特徵的 DataFrame。
        
        優化：
        自動抓取 start_year-1 和 end_year+1 的數據作為緩衝，
        確保年初和年末的 days_until 特徵計算準確，不會出現 999。
        """
        
        # 1. 設定時間範圍
        if end_year is None:
            end_year = datetime.now().year + 1

        # [關鍵修改] 擴大抓取範圍：前後各多抓一年
        # 用途：確保 start_year 年初能算到去年的 CNY，end_year 年末能算到明年的 CNY
        buffer_start_year = start_year - 1
        buffer_end_year = end_year + 1
        
        years = list(range(buffer_start_year, buffer_end_year + 1))
        hk_holidays = holidays.HK(years=years) 

        # 2. 建立包含緩衝區的完整日曆
        full_date_range = pd.date_range(start=f'{buffer_start_year}-01-01', end=f'{buffer_end_year}-12-31')
        df = pd.DataFrame(full_date_range, columns=['date'])
        
        # 3. 映射假期資料
        holiday_data = pd.DataFrame(list(hk_holidays.items()), columns=['date', 'holiday_name'])
        holiday_data['date'] = pd.to_datetime(holiday_data['date'])
        
        # 合併
        df = df.merge(holiday_data, on='date', how='left')
        
        # 4. 基礎特徵
        df['is_holiday'] = df['holiday_name'].notna().astype(int)
        df['day_of_week'] = df['date'].dt.dayofweek 
        
        # [Feature: Working_Day]
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['working_day'] = ((df['is_weekend'] == 0) & (df['is_holiday'] == 0)).astype(int)

        # [Feature: Holiday_Name_Category]
        df['holiday_category'] = df['holiday_name'].apply(ExternalFeatureBuilder._get_holiday_category)

        # [Feature: Is_Long_Weekend]
        df['is_long_weekend'] = 0
        long_weekend_mask = (df['is_holiday'] == 1) & (df['day_of_week'].isin([0, 4]))
        df.loc[long_weekend_mask, 'is_long_weekend'] = 1

        # --- 時間距離特徵計算 (在有 Buffer 的情況下進行) ---
        
        holiday_dates_series = df.loc[df['is_holiday'] == 1, 'date']
        
        # [Feature: Days_Until / Since Any Holiday]
        # 泛用假期的 "Since" 還是保留，因為 "上一個假期" 和 "下一個假期" 可能是完全不同的節日 (e.g. 剛過聖誕，下一個是元旦)
        # 這跟 CNY 的單一循環不同，所以保留泛用的 Since
        df['next_holiday_date'] = holiday_dates_series.reindex(df.index).bfill()
        df['days_until_next_holiday'] = (df['next_holiday_date'] - df['date']).dt.days
        
        df['last_holiday_date'] = holiday_dates_series.reindex(df.index).ffill()
        df['days_since_last_holiday'] = (df['date'] - df['last_holiday_date']).dt.days

        # --- [Feature: CNY Specific] ---
        # 僅保留 days_until_cny，因為它與 days_since_cny 高度負相關 (Sum approx 365)
        # 對零售而言，"距離過年還有多久" (辦年貨) 是最強訊號
        cny_mask = df['holiday_name'].str.contains('Lunar New Year|Chinese New Year', case=False, na=False)
        cny_dates_series = df.loc[cny_mask, 'date']

        if not cny_dates_series.empty:
            df['next_cny_date'] = cny_dates_series.reindex(df.index).bfill()
            df['days_until_cny'] = (df['next_cny_date'] - df['date']).dt.days
        else:
            df['days_until_cny'] = 999

        # 5. [關鍵修改] 裁切回使用者需要的年份範圍
        # 計算完成後，把 buffer 年份切掉，只保留 start_year 到 end_year
        # 這樣邊界的數值就會是正確的計算結果，而不是 999
        mask = (df['date'].dt.year >= start_year) & (df['date'].dt.year <= end_year)
        df_final = df[mask].copy()

        # 再次檢查填充 (以防萬一 buffer 還是不夠，雖然一年通常足夠)
        cols_to_fill = ['days_until_next_holiday', 'days_since_last_holiday', 'days_until_cny']
        for col in cols_to_fill:
            if col in df_final.columns:
                df_final[col] = df_final[col].fillna(999).astype(int)

        # 6. 整理欄位
        cols_to_keep = [
            'date', 
            'day_of_week', 
            'is_weekend', 
            'working_day',
            'is_holiday', 
            'holiday_name', 
            'holiday_category', 
            'is_long_weekend',
            'days_until_next_holiday', 
            'days_since_last_holiday',
            'days_until_cny'  # 已移除 redundant 的 days_since_cny
        ]
        df_final = df_final[cols_to_keep]

        # 7. 儲存
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, f'hk_date_features_{start_year}_{end_year}.csv')
        df_final.to_csv(file_path, index=False, encoding='utf-8-sig')
        
        print(f"特徵工程完成。已生成 {start_year} 至 {end_year} 的數據 (使用了前後一年作為緩衝)。")
        print(f"檔案儲存於: {file_path}")
        
        return df_final

# --- 測試代碼 ---
if __name__ == "__main__":
    # 測試：雖然只請求 2024，但因為有緩衝機制，
    # 2024-01-01 的 days_until_cny 應該能算出數值，而不是 999
    df = ExternalFeatureBuilder.get_hk_holidays(start_year=2024, end_year=2025)
    
    print("\n--- 數據預覽 ---")
    # 檢查 days_until_cny 的變化
    print(df[['date', 'is_holiday', 'days_until_cny']].iloc[0:10]) # 年初
    print(df[['date', 'is_holiday', 'days_until_cny']].iloc[-10:]) # 年末