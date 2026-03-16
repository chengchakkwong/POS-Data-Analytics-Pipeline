"""
【補貨建議 Transform 層】
從庫存主檔 df 加工出補貨決策用欄位，供上傳至 Firebase replenishment collection。
並從入貨紀錄計算 guessed_min / guessed_multiple，供人機協作把關。
"""
import logging
import math
from collections import Counter
from functools import reduce
import pandas as pd
import math
from collections import defaultdict

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


def guess_min_and_multiple(history_records):
    """
    根據歷史入貨紀錄，猜測供應商的 Min (起訂量) 與 Multiple (下單倍數)。

    三步驟：先去雜訊（若有出現≥2 次的數，則嘗試排除「只出現 1 次」中移除後能讓剩餘 GCD 最大的雜訊），
    再求 GCD 得 Multiple，最後在原始紀錄中找「能被 Multiple 整除」的最小值當 Min（排除樣品單）。

    :param history_records: list of int，例如 [24, 36, 25, 12, 2, 48]
    :return: tuple (guessed_min, guessed_multiple)，無資料時 (None, None)
    """
    clean = [int(x) for x in history_records if x is not None and str(x).strip() != '']
    try:
        clean = [x for x in clean if x > 0]
    except (TypeError, ValueError):
        return None, None
    if not clean:
        return None, None
    if len(clean) == 1:
        return clean[0], clean[0]

    # --- Step 1: 資料清洗（找出可靠的數字，必要時排除單次出現的雜訊）---
    counts = Counter(clean)
    has_frequent = any(cnt >= 2 for cnt in counts.values())
    single_occurrence = [num for num, cnt in counts.items() if cnt == 1]
    if not single_occurrence or not has_frequent:
        # 沒有單次出現的數，或沒有任何數出現≥2 次（如 [2,3,4]）：全部參與 GCD
        reliable_nums = list(set(clean))
    else:
        # 有出現≥2 次的數時：嘗試排除某一個「只出現 1 次」的數，看誰能讓剩餘數字的 GCD 最大
        best_gcd = 0
        best_removal = None
        for s in single_occurrence:
            without_s = list(clean)
            without_s.remove(s)
            if without_s:
                g = reduce(math.gcd, without_s)
                if g > best_gcd:
                    best_gcd = g
                    best_removal = s
        if best_removal is not None and best_gcd >= 1:
            without_noise = [x for x in clean if x != best_removal]
            reliable_nums = list(set(without_noise))
        else:
            reliable_nums = list(set(clean))
    reliable_nums.sort()

    # --- Step 2: 推算 Multiple（求 GCD）---
    guessed_multiple = reduce(math.gcd, reliable_nums)
    if guessed_multiple <= 0:
        guessed_multiple = 1

    # --- Step 3: 推算 Min（能被 Multiple 整除的最小值）---
    valid_candidates = [x for x in set(clean) if x % guessed_multiple == 0]
    if valid_candidates:
        guessed_min = min(valid_candidates)
    else:
        guessed_min = min(clean)

    return guessed_min, guessed_multiple


def compute_guessed_min_multiple(df_inbound):
    """
    從入貨紀錄推算每個商品的 guessed_min（最低起訂量）與 guessed_multiple（下單倍數）。

    依 GoodsNo（= ProductCode）分組，每組呼叫 guess_min_and_multiple() 做三步驟：
    去雜訊 → GCD 得 Multiple → 找合法 Min。

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

    logger.info(f"✅ 已計算 {len(result)} 筆商品的 guessed_min / guessed_multiple（三步驟：去雜訊→GCD→合法 Min）")
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
    # 預期: (12, 12) -> 排除只出現 1 次的 25 後，[24,36,12] 的 GCD=12

    print("案例 C (無規律/散出):", guess_min_and_multiple([2, 3, 4]))
    # 預期: (2, 1) -> GCD=1，可買單個，歷史最少 2

    print("案例 D (樣品單混入):", guess_min_and_multiple([2, 24, 48, 24]))
    # 預期: (24, 24) -> 2 無法被 24 整除被排除，合法最低起訂量 24

    sys.exit(0)
