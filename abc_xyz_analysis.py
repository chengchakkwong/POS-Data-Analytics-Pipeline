import pandas as pd
import numpy as np
import os
from logger_config import get_logger
from pathlib import Path
from firebase_service import FirebaseManager

logger = get_logger(__name__)

def analyze_profit_abc(
    stock_df: pd.DataFrame, 
    sales_df: pd.DataFrame, 
    conservative_cost_ratio: float = 0.80,
    month_age_map: dict | None = None,
) -> pd.DataFrame:
    """
    商品 ABC 利潤分析（依傳入的 sales_df 期間計算，加入時間公平與新品隔離邏輯）
    """
    sales_df['rDate'] = pd.to_datetime(sales_df['rDate'])

    # 1. 彙總銷售資料 (加入首賣日以計算年資)
    sales_summary = sales_df.groupby('GoodsID').agg({
        'TotalQty': 'sum', 
        'TotalAmt': 'sum',
        'rDate': ['nunique', 'min']  # 同時抓取銷售天數與首賣日
    })
    sales_summary.columns = ['TotalQty', 'TotalAmt', 'SalesDays', 'FirstSaleDate']
    sales_summary = sales_summary.reset_index()

    # 計算存活月數 (Month_Age)，最少算 1 個月避免除以零
    # 若 month_age_map 提供，代表這個商品的 Month_Age 由「整個歷史」決定（避免誤把舊商品當新品）。
    if month_age_map is None:
        today = sales_df['rDate'].max()
        sales_summary['Month_Age'] = (
            (today.year - sales_summary['FirstSaleDate'].dt.year) * 12
            + (today.month - sales_summary['FirstSaleDate'].dt.month)
        )
        sales_summary['Month_Age'] = sales_summary['Month_Age'].clip(lower=1)
    else:
        # 先放空白，後面 merge 之後會用 month_age_map 覆寫
        sales_summary['Month_Age'] = np.nan

    # 2. 合併資料 (Outer Join)
    merged_df = pd.merge(sales_summary, stock_df, on='GoodsID', how='outer')

    # 若提供 month_age_map，則 Month_Age 由整個歷史首次上架日決定
    if month_age_map is not None:
        merged_df['Month_Age'] = merged_df['GoodsID'].map(month_age_map)

    # 3. 數值與文字欄位填補
    numeric_cols = ['TotalQty', 'TotalAmt', 'CurrStock', 'RetailPrice', 'LastInCost', 'AvgCost']
    merged_df[numeric_cols] = merged_df[numeric_cols].fillna(0)
    merged_df['Month_Age'] = merged_df['Month_Age'].fillna(99).clip(lower=1)  # 沒銷售的當作老商品

    # 處理被刪除商品的名稱與代碼
    mask_missing_info = merged_df['ProductCode'].isna()
    if mask_missing_info.any():
        deleted_labels = "Deleted (ID: " + merged_df['GoodsID'].astype(str).str.split('.').str[0] + ")"
        merged_df.loc[mask_missing_info, 'ProductCode'] = deleted_labels
        merged_df.loc[mask_missing_info, 'Name'] = deleted_labels
    
    merged_df['Barcode'] = merged_df['Barcode'].fillna("DELETED")
    
    text_cols = ['Category', 'Supplier', 'Note', 'InboundLocation']
    for col in text_cols:
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].fillna("DELETED")

    # 4. 標記「雜項 / 泛用」類商品 (排除膠袋等)
    generic_conditions = (
        (merged_df['ProductCode'].astype(str) == '202320232023') |
        (merged_df['Name'].str.contains('五金家品|Deleted|膠袋徵費|塑膠袋', na=False))
    )
    merged_df['Is_Generic'] = np.where(generic_conditions, 'Yes', 'No')

    # 5. 向量化計算調整後成本 (超棒的防呆邏輯保留)
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

    # 6. 計算總利潤 與 🌟月均利潤 (時間公平)🌟
    merged_df['TotalCost'] = merged_df['AdjustedCost'] * merged_df['TotalQty']
    merged_df['TotalProfit'] = merged_df['TotalAmt'] - merged_df['TotalCost']
    merged_df['Monthly_Avg_Profit'] = merged_df['TotalProfit'] / merged_df['Month_Age']

    # 7. 分流計算：排除泛用品
    is_calc = merged_df['Is_Generic'] == 'No'
    df_calc = merged_df[is_calc].copy()
    df_excl = merged_df[~is_calc].copy()

    # 8. 🌟 完美的 ABC 切分邏輯 (先隔離新品，再算 70/20/10) 🌟
    df_calc['ProfitCumulativeRatio'] = 1.0
    df_calc['ABC_Class'] = 'C'
    
    is_new = df_calc['Month_Age'] < 4
    mature_mask = ~is_new

    # 只拿成熟、且利潤為正的商品來排 ABC
    mature_df = df_calc[mature_mask & (df_calc['Monthly_Avg_Profit'] > 0)].copy()
    mature_df = mature_df.sort_values(by='Monthly_Avg_Profit', ascending=False)
    
    total_prof_mature = mature_df['Monthly_Avg_Profit'].sum()
    if total_prof_mature > 0:
        mature_df['CumulativeProfit'] = mature_df['Monthly_Avg_Profit'].cumsum()
        mature_df['ProfitCumulativeRatio'] = mature_df['CumulativeProfit'] / total_prof_mature
    else:
        mature_df['ProfitCumulativeRatio'] = 1.0

    mature_df['ABC_Class'] = np.select(
        [(mature_df['ProfitCumulativeRatio'] <= 0.7), (mature_df['ProfitCumulativeRatio'] <= 0.9)],
        ['A', 'B'],
        default='C'
    )
    
    # 寫回計算結果
    df_calc.loc[mature_df.index, 'ProfitCumulativeRatio'] = mature_df['ProfitCumulativeRatio']
    df_calc.loc[mature_df.index, 'ABC_Class'] = mature_df['ABC_Class']
    
    # 新品直接標記，利潤為負的老品保持為預設的 C
    df_calc.loc[is_new, 'ABC_Class'] = 'New'

    # 9. 處理排除項目
    df_excl['CumulativeProfit'] = np.nan
    df_excl['ProfitCumulativeRatio'] = np.nan
    df_excl['ABC_Class'] = 'Excluded'

    final_df = pd.concat([df_calc, df_excl], ignore_index=True)
    return final_df

def analyze_abc_xyz(abc_df, sales_df):
    """
    進行 XYZ 波動分析
    """
    sales_df['rDate'] = pd.to_datetime(sales_df['rDate'])
    
    # 建立「月份銷售矩陣」
    monthly_matrix = sales_df.groupby([
        'GoodsID', 
        pd.Grouper(key='rDate', freq='MS')
    ])['TotalQty'].sum().unstack(fill_value=0)

    # 計算統計指標 (CV = 標準差 / 平均值)
    stats = pd.DataFrame(index=monthly_matrix.index)
    stats['Mean_Monthly_Qty'] = monthly_matrix.mean(axis=1)
    stats['Std_Monthly_Qty'] = monthly_matrix.std(axis=1)
    
    stats['CV'] = np.where(
        stats['Mean_Monthly_Qty'] > 0, 
        stats['Std_Monthly_Qty'] / stats['Mean_Monthly_Qty'], 
        9.99
    )

    # 定義 XYZ 標籤
    conditions = [
        (stats['CV'] <= 0.5),
        (stats['CV'] <= 1.0),
        (stats['CV'] > 1.0)
    ]
    stats['XYZ_Class'] = np.select(conditions, ['X', 'Y', 'Z'], default='Z')

    # 合併 ABC 與 XYZ
    final_df = pd.merge(abc_df, stats[['CV', 'XYZ_Class', 'Mean_Monthly_Qty']], on='GoodsID', how='left')
    
    # 補足缺失值
    final_df['XYZ_Class'] = final_df['XYZ_Class'].fillna('Z')
    final_df['CV'] = final_df['CV'].fillna(9.99)

    # 確保新品在 XYZ 軸也標記為 New (因為 ABC_Class 已經處理好新品了)
    is_new_abc = final_df['ABC_Class'] == 'New'
    final_df.loc[is_new_abc, 'XYZ_Class'] = 'New'

    # 加上「智慧預測建議策略」
    def get_strategy(row):
        if row['ABC_Class'] == 'New':
            return '新品觀察 (手動控貨)'
        if row['ABC_Class'] == 'Excluded':
            return '排除對象'
            
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
            'CZ': '不建議預測 (考慮汰換)'
        }
        return strategy_map.get(combo, '其他')

    final_df['Strategy'] = final_df.apply(get_strategy, axis=1)
    final_df['displayname'] = final_df['Name'] + " | " + final_df['ProductCode']
    
    # 最終排序：新品排最前，接著 A -> B -> C -> Excluded
    final_df['SortOrder'] = final_df['ABC_Class'].replace({'New': 0, 'A': 1, 'B': 2, 'C': 3, 'Excluded': 4})
    # 使用月均利潤作為第二排序條件
    final_df = final_df.sort_values(by=['SortOrder', 'Monthly_Avg_Profit'], ascending=[True, False]).drop(columns=['SortOrder'])

    return final_df

def upload_classification_to_firebase(classification_df: pd.DataFrame) -> None:
    try:
        firebase = FirebaseManager()
        firebase.upload_classification_df(classification_df)
        logger.info("✅ 分類結果已上傳至 Firebase")
    except Exception as e:
        logger.error(f"❌ 上傳分類結果失敗: {e}")

if __name__ == "__main__":
    input_stock = "data/processed/DetailGoodsStockToday.csv"
    input_sales = "data/processed/vw_GoodsDailySales_partitioned"
    output_path = "data/insights/abc_xyz_analysis.csv"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    logger.info("🔍 開始進行智慧庫存分析 (ABC-XYZ + 新品識別)...")
    
    try:
        if not os.path.exists(input_stock) or not os.path.exists(input_sales):
            logger.error("❌ 錯誤: 找不到輸入檔案，請確認路徑。")
        else:
            df_stock = pd.read_csv(input_stock)
            df_sales = pd.read_parquet(input_sales)
            
            # 1. 篩選最近 12 個月的數據做 ABC (價值)
            df_sales['rDate'] = pd.to_datetime(df_sales['rDate'])
            last_date = df_sales['rDate'].max()
            start_date_abc = last_date - pd.DateOffset(months=12)
            df_sales_recent12 = df_sales[df_sales['rDate'] >= start_date_abc].copy()

            # 2. Month_Age 用「整個歷史」的首次上架日來決定
            #    目的：New（新品）定義為「商品整個歷史首次上架」距今是否 < 4 個月。
            first_sale_full = (
                df_sales.groupby('GoodsID')['rDate']
                .min()
                .reset_index()
                .rename(columns={'rDate': 'FirstSaleDate'})
            )
            month_age = (
                (last_date.year - first_sale_full['FirstSaleDate'].dt.year) * 12
                + (last_date.month - first_sale_full['FirstSaleDate'].dt.month)
            ).clip(lower=1)
            month_age_map = dict(zip(first_sale_full['GoodsID'], month_age))
            
            # 2. 執行分析
            abc_df = analyze_profit_abc(df_stock, df_sales_recent12, month_age_map=month_age_map)
            abc_xyz_df = analyze_abc_xyz(abc_df, df_sales)
            
            # 3. 儲存結果
            abc_xyz_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            logger.info(f"✅ 分析完成！共處理 {len(abc_xyz_df)} 個 SKU。")
            logger.info(f"📊 結果儲存至: {output_path}")
            
            new_count = len(abc_xyz_df[abc_xyz_df['ABC_Class'] == 'New'])
            logger.info(f"💡 偵測到 {new_count} 個新上架產品，已標記為 'New' 避開誤判。")

            # 4. 上傳分類結果到 Firebase
            upload_classification_to_firebase(abc_xyz_df)
            
    except Exception:
        logger.exception("❌ 分析失敗")