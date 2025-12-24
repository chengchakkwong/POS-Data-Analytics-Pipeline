import pandas as pd
import numpy as np
import os

def analyze_profit_abc(
    stock_df: pd.DataFrame, 
    sales_df: pd.DataFrame, 
    conservative_cost_ratio: float = 0.80
) -> pd.DataFrame:
    """
    優化後的商品 ABC 利潤分析 (向量化版本)
    """
    # 1. 彙總銷售資料
    sales_summary = sales_df.groupby('GoodsID').agg({
        'TotalQty': 'sum', 
        'TotalAmt': 'sum'
    }).reset_index()

    # 2. 合併資料 (Outer Join)
    merged_df = pd.merge(sales_summary, stock_df, on='GoodsID', how='outer')

    # 3. 數值與文字欄位填補 (優化速度)
    numeric_cols = ['TotalQty', 'TotalAmt', 'CurrStock', 'RetailPrice', 'LastInCost', 'AvgCost']
    merged_df[numeric_cols] = merged_df[numeric_cols].fillna(0)

    # 處理被刪除商品的名稱與代碼 (避免使用 apply)
    mask_missing_info = merged_df['ProductCode'].isna()
    if mask_missing_info.any():
        deleted_labels = "Deleted (ID: " + merged_df['GoodsID'].astype(str).str.split('.').str[0] + ")"
        merged_df['ProductCode'] = merged_df['ProductCode'].fillna(deleted_labels)
        merged_df['Name'] = merged_df['Name'].fillna(deleted_labels)
    
    merged_df['Barcode'] = merged_df['Barcode'].fillna("DELETED")
    
    # 填補其餘文字欄位
    text_cols = ['Category', 'Supplier', 'Note', 'InboundLocation']
    for col in text_cols:
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].fillna("DELETED")

    # 4. 標記「雜項 / 泛用」類商品
    generic_conditions = (
        (merged_df['ProductCode'].astype(str) == '202320232023') |
        (merged_df['Name'].str.contains('五金家品|Deleted|膠袋徵費', na=False))
    )
    merged_df['Is_Generic'] = np.where(generic_conditions, 'Yes', 'No')

    # 5. 向量化計算調整後成本 (取代 apply)
    # 計算成交單價
    unit_price = np.where(merged_df['TotalQty'] > 0, merged_df['TotalAmt'] / merged_df['TotalQty'], 0)
    
    # 判斷成本是否異常 (成本缺失 或 售價/成本 > 9)
    cost_missing = (merged_df['LastInCost'] <= 0)
    
    # 避免除以 0 的警告，先設為 True
    cost_suspicious = np.zeros(len(merged_df), dtype=bool)
    valid_cost_mask = merged_df['LastInCost'] > 0
    cost_suspicious[valid_cost_mask] = (unit_price[valid_cost_mask] / merged_df.loc[valid_cost_mask, 'LastInCost']) > 9

    # 計算 AdjustedCost
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

    # 排序並計算累計利潤
    df_calc = df_calc.sort_values(by='TotalProfit', ascending=False)
    df_calc['CumulativeProfit'] = df_calc['TotalProfit'].cumsum()
    
    total_prof_sum = df_calc['TotalProfit'].sum()
    if total_prof_sum != 0:
        df_calc['ProfitCumulativeRatio'] = df_calc['CumulativeProfit'] / total_prof_sum
    else:
        df_calc['ProfitCumulativeRatio'] = 0

    # 8. 套用 ABC 分級標籤 (向量化)
    conditions = [
        (df_calc['ProfitCumulativeRatio'] <= 0.7),
        (df_calc['ProfitCumulativeRatio'] <= 0.9),
        (df_calc['ProfitCumulativeRatio'] > 0.9)
    ]
    choices = ['A', 'B', 'C']
    df_calc['ABC_Class'] = np.select(conditions, choices, default='else')

    # 9. 處理排除項目的欄位
    df_excl['CumulativeProfit'] = np.nan
    df_excl['ProfitCumulativeRatio'] = np.nan
    df_excl['ABC_Class'] = 'Excluded'

    # 10. 合併回最終結果
    final_df = pd.concat([df_calc, df_excl], ignore_index=True)
    final_df['displayname'] = final_df['Name'] + " | " + final_df['ProductCode']
    
    # 依利潤排序 (A商品在前)
    final_df = final_df.sort_values(by=['ABC_Class', 'TotalProfit'], ascending=[True, False])

    return final_df

if __name__ == "__main__":
    # 設定路徑
    input_stock = "data/processed/DetailGoodsStockToday.csv"
    input_sales = "data/processed/vw_GoodsDailySales_cache.parquet"
    output_path = "data/processed/ABC_Analysis_Insights2.csv"

    try:
        if not os.path.exists(input_stock) or not os.path.exists(input_sales):
            print("❌ 錯誤: 找不到輸入檔案，請確認路徑。")
        else:
            df_s = pd.read_csv(input_stock)
            df_h = pd.read_parquet(input_sales)
            
            df_insights = analyze_profit_abc(df_s, df_h)
            
            df_insights.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"✅ 商業洞察分析完成！儲存至: {output_path}")
            
    except Exception as e:
        print(f"❌ 分析失敗: {e}")