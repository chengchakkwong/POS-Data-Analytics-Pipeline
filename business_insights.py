import pandas as pd
import os
import time

def preprocess_data(df_stock, df_sales, conservative_cost_ratio=0.80):
    """
    數據預處理：合併資料、標記雜項並修正虛假利潤。
    """
    # 合併數據
    df = pd.merge(
        df_sales, 
        df_stock[['GoodsID', 'ProductCode', 'Barcode', 'Name', 'Category', 'Supplier', 'RetailPrice', 'LastInCost', 'CurrStock']], 
        on='GoodsID', 
        how='left'
    )

    # 標記雜項 (Is_Generic)
    df['Is_Generic_Flag'] = (
        (df['ProductCode'].astype(str) == '202320232023') |
        (df['Name'].str.contains('五金家品', na=False))
    )
    df['Is_Generic'] = df['Is_Generic_Flag'].map({True: 'Yes', False: 'No'})

    # 修正成本邏輯
    def get_adjusted_cost(row):
        base_cost = row['LastInCost']
        total_amt = row['TotalAmt']
        total_qty = row['TotalQty']
        
        # 計算實際成交單價
        unit_price = total_amt / total_qty if total_qty > 0 else 0
        
        # 判定條件 1：成本為 0 或缺失 (針對雜項)
        is_cost_missing = (base_cost <= 0 or pd.isna(base_cost))
        
        # 判定條件 2：異常高毛利 (售價/成本 比率 > 9)
        # 說明：如果售價是成本的 9 倍以上，通常是入庫單位錯誤，視為「成本不對」
        is_cost_suspicious = False
        if base_cost > 0 and unit_price > 0:
            if (unit_price / base_cost) > 9:
                is_cost_suspicious = True

        # 如果符合以上任一條件，且屬於雜項或特定需要調整的對象
        if (is_cost_missing or is_cost_suspicious):
            return unit_price * conservative_cost_ratio
        
        return base_cost

    df['AdjustedCost'] = df.apply(get_adjusted_cost, axis=1)
    df['TotalCost'] = df['AdjustedCost'] * df['TotalQty']
    df['TotalProfit'] = df['TotalAmt'] - df['TotalCost']
    
    return df

def analyze_supplier_performance(df):
    """
    [分析 1] 供應商表現排行
    """
    return df.groupby('Supplier').agg({
        'TotalAmt': 'sum',
        'TotalProfit': 'sum',
        'TotalQty': 'sum',
    }).sort_values(by='TotalProfit', ascending=False)

def analyze_abc_classification(df):
    """
    [分析 2] 產品 ABC 分級 (僅針對非雜項)
    """
    product_profit = (
        df[df['Is_Generic'] == 'No']
        .groupby(['ProductCode', 'Name'], as_index=False)
        .agg({'TotalProfit': 'sum', 'TotalQty': 'sum'})
        .sort_values(by='TotalProfit', ascending=False)
    )

    if product_profit.empty:
        return pd.DataFrame()

    product_profit['CumSumProfit'] = product_profit['TotalProfit'].cumsum()
    total_profit_sum = product_profit['TotalProfit'].sum()
    product_profit['ProfitPercent'] = product_profit['CumSumProfit'] / total_profit_sum

    def abc_classifier(percent):
        if percent <= 0.7: return 'A'
        elif percent <= 0.9: return 'B'
        else: return 'C'

    product_profit['ABC_Class'] = product_profit['ProfitPercent'].apply(abc_classifier)
    return product_profit

def analyze_inventory_health(df_stock, df_sales, df_merged):
    """
    [分析 3] 庫存健康度與補貨預警
    """
    total_days = (pd.to_datetime(df_sales['rDate']).max() - pd.to_datetime(df_sales['rDate']).min()).days + 1
    avg_daily_sales = df_merged.groupby('GoodsID')['TotalQty'].sum() / total_days
    
    df_inventory = df_stock[['GoodsID', 'Name', 'CurrStock', 'Category']].copy()
    df_inventory['AvgDailySales'] = df_inventory['GoodsID'].map(avg_daily_sales).fillna(0)
    df_inventory['DaysOfInventory'] = df_inventory['CurrStock'] / df_inventory['AvgDailySales']
    
    df_inventory['Status'] = 'Normal'
    df_inventory.loc[(df_inventory['DaysOfInventory'] < 7) & (df_inventory['AvgDailySales'] > 0), 'Status'] = 'Low Stock'
    df_inventory.loc[df_inventory['AvgDailySales'] == 0, 'Status'] = 'Dead Stock'
    
    return df_inventory

def generate_business_insights(df_stock, df_sales):
    """
    【主程序】統籌所有分析函式並輸出結果。
    """
    print(f"🚀 [{time.strftime('%H:%M:%S')}] 啟動模組化商業分析程序...")

    if df_stock is None or df_sales is None:
        print("❌ 錯誤：數據輸入為空。")
        return None

    # 1. 預處理
    df_master = preprocess_data(df_stock, df_sales)

    # 2. 執行各項分析
    df_supplier = analyze_supplier_performance(df_master)
    df_abc = analyze_abc_classification(df_master)
    df_inventory = analyze_inventory_health(df_stock, df_sales, df_master)

    # 3. 儲存結果
    if not os.path.exists('data/insights'):
        os.makedirs('data/insights')
    
    df_master.drop(columns=['Is_Generic_Flag']).to_csv('data/insights/bi_master_sales_report.csv', index=False, encoding='utf-8-sig')
    df_supplier.to_csv('data/insights/supplier_performance.csv', encoding='utf-8-sig')
    df_abc.to_csv('data/insights/product_abc_analysis.csv', index=False, encoding='utf-8-sig')
    df_inventory.to_csv('data/insights/inventory_health_report.csv', index=False, encoding='utf-8-sig')

    print("-" * 40)
    print(f"✅ 模組化分析完成！已產出以下報表：")
    print(f"1. bi_master_sales_report.csv (總表)")
    print(f"2. supplier_performance.csv (供應商)")
    print(f"3. product_abc_analysis.csv (ABC 分級)")
    print(f"4. inventory_health_report.csv (庫存健康)")
    print("-" * 40)
    
    return df_master

if __name__ == "__main__":
    try:
        df_s = pd.read_csv("data/processed/DetailGoodsStockToday.csv")
        df_h = pd.read_parquet("data/processed/vw_GoodsDailySales_cache.parquet")
        generate_business_insights(df_s, df_h)
    except Exception as e:
        print(f"分析失敗: {e}")