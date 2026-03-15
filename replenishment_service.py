"""
【補貨建議 Transform 層】
從庫存主檔 df 加工出補貨決策用欄位，供上傳至 Firebase replenishment collection。
"""
import logging
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
