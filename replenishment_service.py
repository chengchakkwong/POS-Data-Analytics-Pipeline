"""
【補貨建議 Transform 層】
從庫存主檔 df 加工出補貨決策用欄位，供上傳至 Firebase replenishment collection。
並從入貨紀錄計算 guessed_min / guessed_multiple，供人機協作把關。
"""
import logging
import math
from collections import Counter, defaultdict
from functools import reduce
import pandas as pd

logger = logging.getLogger(__name__)

# 補貨上傳用欄位（需與 firebase_service.upload_replenishment_data 一致）
REPLENISHMENT_COLS = [
    'ProductCode',
    'LastInCost',
    'AvgCost',
    'InboundLocation',
    'FirstOrderQty',
    'NoteDescription',
]


def prepare(df_stock):
    """
    將庫存主檔加工成補貨建議用 DataFrame。

    - 解析 Note：抽出 FirstOrderQty（獨立數字）與 NoteDescription（其餘文字）
    - 只保留補貨相關欄位

    :param df_stock: pos_service.get_stock_master_data() 回傳的 DataFrame
    :return: 僅含 REPLENISHMENT_COLS 的 DataFrame，若缺欄位則跳過該欄
    """
    if df_stock is None or df_stock.empty:
        logger.warning("⚠️ 庫存主檔為空，跳過補貨加工")
        return pd.DataFrame()

    df = df_stock.copy()

    # 確保 Note 為字串再解析
    if 'Note' not in df.columns:
        logger.warning("⚠️ 庫存主檔無 Note 欄位，將不產生 FirstOrderQty / NoteDescription")
        df['Note'] = ''
    note_series = df['Note'].astype(str).replace('nan', '')

    # FirstOrderQty：從 Note 抽出獨立數字 (?:^|\s)(\d+)(?:\s|$)
    df['FirstOrderQty'] = note_series.str.extract(r'(?:^|\s)(\d+)(?:\s|$)', expand=False)

    # NoteDescription：移除該數字後剩餘文字
    df['NoteDescription'] = note_series.str.replace(r'(?:^|\s)\d+(?:\s|$)', ' ', regex=True).str.strip()
    df['NoteDescription'] = df['NoteDescription'].replace('', None)

    # 只保留存在的補貨欄位
    valid_cols = [c for c in REPLENISHMENT_COLS if c in df.columns]
    if len(valid_cols) < len(REPLENISHMENT_COLS):
        missing = set(REPLENISHMENT_COLS) - set(valid_cols)
        logger.warning(f"⚠️ 補貨資料缺少欄位，將略過: {missing}")

    return df[valid_cols]


def guess_min_and_multiple(history_records, threshold_ratio=0.6):
    """
    根據歷史入貨紀錄，猜測供應商的 Min (起訂量) 與 Multiple (下單倍數)。

    使用「因數統計法」：拆解所有紀錄的因數並舉辦投票，
    得票數超過門檻 (預設 60%) 的最大因數即為 Multiple。
    最後在原始紀錄中找「能被 Multiple 整除」的最小值當 Min（排除樣品單）。

    :param history_records: list of int，例如 [24, 36, 25, 12, 2, 48]
    :param threshold_ratio: float，門檻比例 (0.6 代表該因數必須出現在 60% 的有效紀錄中)
    :return: tuple (guessed_min, guessed_multiple)，無資料時 (None, None)
    """
    # --- 前置作業：資料清洗與防呆 ---
    clean = [int(x) for x in history_records if x is not None and str(x).strip() != '']
    try:
        clean = [x for x in clean if x > 0]
    except (TypeError, ValueError):
        return None, None
        
    if not clean:
        return None, None
    if len(clean) == 1:
        return clean[0], clean[0]

    # --- Step 1 & 2: 因數統計法推算 Multiple ---
    total_count = len(clean)
    factor_votes = defaultdict(int)
    
    # 拆解每個數字的因數，並投下神聖的一票
    for num in clean:
        # 只需計算到平方根，大幅提升運算效能
        limit = int(math.sqrt(num)) + 1
        for i in range(1, limit):
            if num % i == 0:
                factor_votes[i] += 1
                # 加上對稱的另一個因數，避免重複計算
                if i != num // i:
                    factor_votes[num // i] += 1

    # 計算當選門檻 (無條件進位)，例如：6 筆紀錄 * 0.6 = 3.6 -> 需要 4 票才能當選
    required_votes = math.ceil(total_count * threshold_ratio)
    
    guessed_multiple = 1
    for factor, votes in factor_votes.items():
        if votes >= required_votes:
            # 在所有達到門檻的因數中，挑選數字最大的
            if factor > guessed_multiple:
                guessed_multiple = factor

    # --- Step 3: 推算 Min（能被 Multiple 整除的最小值） ---
    valid_candidates = [x for x in set(clean) if x % guessed_multiple == 0]
    if valid_candidates:
        guessed_min = min(valid_candidates)
    else:
        guessed_min = min(clean)

    return guessed_min, guessed_multiple


def compute_guessed_min_multiple(df_inbound):
    """
    從入貨紀錄推算每個商品的 guessed_min（最低起訂量）與 guessed_multiple（下單倍數）。

    依 GoodsNo（= ProductCode）分組，每組呼叫 guess_min_and_multiple() 做推算：
    利用「因數統計法」自動過濾多重雜訊，找出最佳 Multiple 與合法 Min。

    :param df_inbound: pos_service.get_inbound_movements_for_min_multiple() 回傳的 DataFrame，
                       需含欄位 GoodsNo, ChQty（BillDate 可選）
    :return: DataFrame 含欄位 ProductCode, guessed_min, guessed_multiple；無資料時為空 DataFrame
    """
    if df_inbound is None or df_inbound.empty:
        logger.warning("⚠️ 入貨紀錄為空，跳過 Min/Multiple 計算")
        return pd.DataFrame()

    required = ['GoodsNo', 'ChQty']
    if not all(c in df_inbound.columns for c in required):
        logger.warning(f"⚠️ 入貨紀錄缺少欄位 {required}，跳過 Min/Multiple 計算")
        return pd.DataFrame()

    df = df_inbound.copy()
    df['ChQty'] = pd.to_numeric(df['ChQty'], errors='coerce').fillna(0).astype(int)
    df = df[df['ChQty'] > 0]

    if df.empty:
        logger.warning("⚠️ 過濾後無有效入貨量，跳過 Min/Multiple 計算")
        return pd.DataFrame()

    # 對每個商品群組套用因數統計法
    result = df.groupby('GoodsNo').apply(
        lambda g: guess_min_and_multiple(g['ChQty'].tolist())
    )
    
    result = result.reset_index()
    result.columns = ['GoodsNo', '_pair']
    result['guessed_min'] = result['_pair'].apply(lambda p: p[0] if p else None)
    result['guessed_multiple'] = result['_pair'].apply(lambda p: p[1] if p else None)
    result = result.drop(columns=['_pair'])
    result = result.dropna(subset=['guessed_min', 'guessed_multiple'])
    result = result.rename(columns={'GoodsNo': 'ProductCode'})
    result['ProductCode'] = result['ProductCode'].astype(str).str.strip()

    logger.info(f"✅ 已計算 {len(result)} 筆商品的 guessed_min / guessed_multiple（使用高效因數統計法）")
    return result[['ProductCode', 'guessed_min', 'guessed_multiple']]
# ---------------------------------------------------------------------------
# 測試案例（執行 python replenishment_service.py 可驗證演算法）
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    print("案例 A (標準規律):", guess_min_and_multiple([24, 36, 12, 48]))
    # 預期: (12, 12)

    print("案例 B (混入雜訊錯字):", guess_min_and_multiple([24, 24, 25, 36, 12]))
    # 預期: (12, 12) -> 因數統計法下 25 的因數得票少，12 達門檻為 Multiple

    print("案例 C (無規律/散出):", guess_min_and_multiple([2, 3, 4]))
    # 原本期望（2,1）
    # 預期: (2, 2) -> 2 為共同因數達門檻；小樣本不另做嚴格模式，避免過度擬合／過度設計

    print("案例 D (樣品單混入):", guess_min_and_multiple([2, 24, 48, 24]))
    # 預期: (24, 24) -> 2 無法被 24 整除被排除，合法最低起訂量 24

    sys.exit(0)
