import pandas as pd
import numpy as np
import os

def analyze_profit_abc(
    stock_df: pd.DataFrame, 
    sales_df: pd.DataFrame, 
    conservative_cost_ratio: float = 0.80
) -> pd.DataFrame:
    """
    商品 ABC 利潤分析 (針對最近 12 個月)
    """
    # 1. 彙總銷售資料 (加入頻率觀察)
    sales_summary = sales_df.groupby('GoodsID').agg({
        'TotalQty': 'sum', 
        'TotalAmt': 'sum',
        'rDate': 'nunique'  # 計算這商品有多少天產生過銷售
    }).rename(columns={'rDate': 'SalesDays'}).reset_index()

    # 2. 合併資料 (Outer Join)
    merged_df = pd.merge(sales_summary, stock_df, on='GoodsID', how='outer')

    # 3. 數值與文字欄位填補
    numeric_cols = ['TotalQty', 'TotalAmt', 'CurrStock', 'RetailPrice', 'LastInCost', 'AvgCost']
    merged_df[numeric_cols] = merged_df[numeric_cols].fillna(0)

    # 處理被刪除商品的名稱與代碼
    mask_missing_info = merged_df['ProductCode'].isna()
    if mask_missing_info.any():
        deleted_labels = "Deleted (ID: " + merged_df['GoodsID'].astype(str).str.split('.').str[0] + ")"
        merged_df['ProductCode'] = merged_df['ProductCode'].fillna(deleted_labels)
        merged_df['Name'] = merged_df['Name'].fillna(deleted_labels)
    
    merged_df['Barcode'] = merged_df['Barcode'].fillna("DELETED")
    
    text_cols = ['Category', 'Supplier', 'Note', 'InboundLocation']
    for col in text_cols:
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].fillna("DELETED")

    # 4. 標記「雜項 / 泛用」類商品 (排除膠袋等)
    generic_conditions = (
        (merged_df['ProductCode'].astype(str) == '202320232023') |
        (merged_df['Name'].str.contains('五金家品|Deleted|膠袋徵費|塑膠袋|購物袋', na=False))
    )
    merged_df['Is_Generic'] = np.where(generic_conditions, 'Yes', 'No')

    # 5. 向量化計算調整後成本
    unit_price = np.where(merged_df['TotalQty'] > 0, merged_df['TotalAmt'] / merged_df['TotalQty'], 0)
    cost_missing = (merged_df['LastInCost'] <= 0)
    
    cost_suspicious = np.zeros(len(merged_df), dtype=bool)
    valid_cost_mask = merged_df['LastInCost'] > 0
    cost_suspicious[valid_cost_mask] = (unit_price[valid_cost_mask] / merged_df.loc[valid_cost_mask, 'LastInCost']) > 9

    merged_df['AdjustedCost'] = np.where(
        cost_missing | cost_suspicious,
        unit_price * conservative_cost_ratio,
        merged_df['LastInCost']
    )

    # 6. 計算利潤
    merged_df['TotalCost'] = merged_df['AdjustedCost'] * merged_df['TotalQty']
    merged_df['TotalProfit'] = merged_df['TotalAmt'] - merged_df['TotalCost']

    # 7. ABC 分類計算 (只針對非 Generic)
    is_calc = merged_df['Is_Generic'] == 'No'
    df_calc = merged_df[is_calc].copy()
    df_excl = merged_df[~is_calc].copy()

    # 8. 套用 ABC 分級標籤 (加入負利潤保護)
    df_calc = df_calc.sort_values(by='TotalProfit', ascending=False)
    positive_profit_mask = df_calc['TotalProfit'] > 0

    # 只對正利潤跑累計
    df_calc.loc[positive_profit_mask, 'CumulativeProfit'] = df_calc.loc[positive_profit_mask, 'TotalProfit'].cumsum()
    total_pos_prof = df_calc.loc[positive_profit_mask, 'TotalProfit'].sum()
    if total_pos_prof != 0:
        df_calc.loc[positive_profit_mask, 'ProfitCumulativeRatio'] = df_calc['CumulativeProfit'] / total_pos_prof
    else:
        df_calc['ProfitCumulativeRatio'] = 0

    conditions = [
        (df_calc['ProfitCumulativeRatio'] <= 0.7) & positive_profit_mask,
        (df_calc['ProfitCumulativeRatio'] <= 0.9) & positive_profit_mask,
        (df_calc['TotalProfit'] <= 0)
    ]
    choices = ['A', 'B', 'C']
    df_calc['ABC_Class'] = np.select(conditions, choices, default='C')

    # 9. 處理排除項目
    df_excl['CumulativeProfit'] = np.nan
    df_excl['ProfitCumulativeRatio'] = np.nan
    df_excl['ABC_Class'] = 'Excluded'

    final_df = pd.concat([df_calc, df_excl], ignore_index=True)
    return final_df

def analyze_abc_xyz(abc_df, sales_df):
    """
    進行 XYZ 波動分析，並加入「新品(New SKU)」識別邏輯
    """
    # 1. 時間格式化
    sales_df['rDate'] = pd.to_datetime(sales_df['rDate'])
    today = sales_df['rDate'].max()
    
    # --- 新增：新品識別邏輯 ---
    # 找出每個商品的「首賣日」
    first_sale = sales_df.groupby('GoodsID')['rDate'].min().reset_index()
    first_sale.columns = ['GoodsID', 'FirstSaleDate']
    
    # 計算商品年資 (月)
    first_sale['Month_Age'] = (today.year - first_sale['FirstSaleDate'].dt.year) * 12 + \
                               (today.month - first_sale['FirstSaleDate'].dt.month)
    # -----------------------

    # 2. 建立「月份銷售矩陣」 (看每個月波動)
    monthly_matrix = sales_df.groupby([
        'GoodsID', 
        pd.Grouper(key='rDate', freq='MS')
    ])['TotalQty'].sum().unstack(fill_value=0)

    # 3. 計算統計指標 (CV = 標準差 / 平均值)
    stats = pd.DataFrame(index=monthly_matrix.index)
    stats['Mean_Monthly_Qty'] = monthly_matrix.mean(axis=1)
    stats['Std_Monthly_Qty'] = monthly_matrix.std(axis=1)
    
    stats['CV'] = np.where(
        stats['Mean_Monthly_Qty'] > 0, 
        stats['Std_Monthly_Qty'] / stats['Mean_Monthly_Qty'], 
        9.99
    )

    # 4. 定義 XYZ 標籤
    conditions = [
        (stats['CV'] <= 0.5),
        (stats['CV'] <= 1.0),
        (stats['CV'] > 1.0)
    ]
    choices = ['X', 'Y', 'Z']
    stats['XYZ_Class'] = np.select(conditions, choices, default='Z')

    # 5. 合併 ABC、XYZ 與 新品標籤
    final_df = pd.merge(abc_df, stats[['CV', 'XYZ_Class', 'Mean_Monthly_Qty']], on='GoodsID', how='left')
    final_df = pd.merge(final_df, first_sale, on='GoodsID', how='left')
    
    # 補足缺失值
    final_df['XYZ_Class'] = final_df['XYZ_Class'].fillna('Z')
    final_df['CV'] = final_df['CV'].fillna(9.99)
    final_df['Month_Age'] = final_df['Month_Age'].fillna(99) # 沒銷售記錄的視為老商品/未知

    # --- 強制覆蓋新品標籤 ---
    # 定義新品門檻 (例如上架少於 4 個月)
    is_new = (final_df['Month_Age'] < 4) & (final_df['ABC_Class'] != 'Excluded')
    final_df.loc[is_new, 'ABC_Class'] = 'New'
    final_df.loc[is_new, 'XYZ_Class'] = 'New'

    # 6. 加上「智慧預測建議策略」
    def get_strategy(row):
        if row['ABC_Class'] == 'New':
            return '新品觀察 (手動控貨)'
        
        combo = f"{row['ABC_Class']}{row['XYZ_Class']}"
        strategy_map = {
            'AX': '自動補貨 (高頻穩定)',
            'AY': 'AI 季節性預測 (重點對象)',
            'AZ': '高安全庫存 (利潤高但難抓)',
            'BX': '定期補貨',
            'BY': '季節性補貨',
            'BZ': '觀望/依訂單進貨',
            'CX': '基本品 (低庫存管理)',
            'CY': '季節品 (減少庫存)',
            'CZ': '不建議預測 (考慮汰換)',
            'ExcludedExcluded': '排除對象'
        }
        return strategy_map.get(combo, '其他')

    final_df['Strategy'] = final_df.apply(get_strategy, axis=1)
    final_df['displayname'] = final_df['Name'] + " | " + final_df['ProductCode']
    
    # 最終排序：新品排最前，接著 A -> B -> C
    final_df['SortOrder'] = final_df['ABC_Class'].replace({'New': 0, 'A': 1, 'B': 2, 'C': 3, 'Excluded': 4})
    final_df = final_df.sort_values(by=['SortOrder', 'TotalProfit'], ascending=[True, False]).drop(columns=['SortOrder'])

    return final_df

if __name__ == "__main__":
    input_stock = "data/processed/DetailGoodsStockToday.csv"
    input_sales = "data/processed/vw_GoodsDailySales_cache.parquet"
    output_path = "data/insights/abc_xyz_analysis.csv"
    
    print("🔍 開始進行智慧庫存分析 (ABC-XYZ + 新品識別)...")
    
    try:
        if not os.path.exists(input_stock) or not os.path.exists(input_sales):
            print("❌ 錯誤: 找不到輸入檔案，請確認路徑。")
        else:
            df_stock = pd.read_csv(input_stock)
            df_sales = pd.read_parquet(input_sales)
            
            # 1. 篩選最近 12 個月的數據做 ABC (價值)
            df_sales['rDate'] = pd.to_datetime(df_sales['rDate'])
            last_date = df_sales['rDate'].max()
            start_date_abc = last_date - pd.DateOffset(months=12)
            df_sales_recent12 = df_sales[df_sales['rDate'] >= start_date_abc].copy()
            
            # 2. 執行分析
            # ABC 使用近 12 個月確定價值地位
            abc_df = analyze_profit_abc(df_stock, df_sales_recent12)
            
            # XYZ 使用全部 23 個月確定波動規律 (內部會判斷 Month_Age)
            abc_xyz_df = analyze_abc_xyz(abc_df, df_sales)
            
            # 3. 儲存結果
            abc_xyz_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"✅ 分析完成！共處理 {len(abc_xyz_df)} 個 SKU。")
            print(f"📊 結果儲存至: {output_path}")
            
            # 簡單統計
            new_count = len(abc_xyz_df[abc_xyz_df['ABC_Class'] == 'New'])
            print(f"💡 偵測到 {new_count} 個新上架產品，已標記為 'New' 避開誤判。")
            
    except Exception as e:
        print(f"❌ 分析失敗: {e}")