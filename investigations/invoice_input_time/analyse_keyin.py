# import pos_pipeline.config
# from pos_pipeline.database.connection import execute_query
# import pandas as pd

# SESSION_GAP_SEC = 10 * 60  # 10 分鐘

# sql = """
# SELECT
#     SID,
#     recTime,
#     ProductType2Name1 AS supplier,
#     UserName,
#     BillDate
# FROM dbo.GoodsStockMovement
# WHERE MoveTypeID = 1
#   AND recTime IS NOT NULL
#   AND LTRIM(RTRIM(recTime)) <> ''
#   AND recTime >= '20250901000000'
#   AND recTime <  '20260901000000'
# ORDER BY SID
# """

# df = execute_query(sql)

import pandas as pd

from pos_pipeline.config import PROCESSED_DIR


SESSION_GAP_SEC = 10 * 60

INPUT_PATH = (
    PROCESSED_DIR
    / "GoodsStockMovement_20250901_20260831.csv"
)

df = pd.read_csv(
    INPUT_PATH,
    
    usecols=[
    "SID",
    "recTime",
    "ProductType2Name1",
    "UserName",
    "BillDate",
    "GoodsNo",
    "Barcode",
    "ChQty",
    ],
    dtype={
        "recTime": "string",
        "BillDate": "string",
        "UserName": "string",
    },
)

df = df.rename(
    columns={"ProductType2Name1": "supplier"}
)


df["rec_dt"] = pd.to_datetime(df["recTime"].astype(str).str.strip(), format="%Y%m%d%H%M%S", errors="coerce")
df = df.dropna(subset=["rec_dt"]).sort_values("rec_dt").reset_index(drop=True)

df["gap_sec"] = df["rec_dt"].diff().dt.total_seconds()
df["new_session"] = df["gap_sec"].isna() | (df["gap_sec"] > SESSION_GAP_SEC)
df["session_id"] = df["new_session"].cumsum()

# session 內第一筆沒有有效「輸入間隔」
work = df.loc[~df["new_session"], ["gap_sec", "supplier", "rec_dt"]].copy()

print("rows used:", len(df))
print("sessions:", int(df["session_id"].nunique()))
print("intervals:", len(work))
print("\n--- 每筆間隔（秒）---")
print(work["gap_sec"].describe(percentiles=[0.5, 0.75, 0.9]).to_string())

print("\n--- 供應商：筆數 / 間隔總秒數 / 中位間隔 ---")
by_sup = (
    work.groupby("supplier", dropna=False)
    .agg(intervals=("gap_sec", "size"), total_sec=("gap_sec", "sum"), median_sec=("gap_sec", "median"))
    .sort_values("total_sec", ascending=False)
    .head(15)
)
print(by_sup.to_string())
print("\n--- 間隔品質檢查 ---")
print("0 秒:", int((work["gap_sec"] == 0).sum()))
print("1–2 秒:", int(work["gap_sec"].between(1, 2).sum()))
print("超過 1 分鐘:", int((work["gap_sec"] > 60).sum()))
print("超過 5 分鐘:", int((work["gap_sec"] > 300).sum()))

print("\n--- 操作員筆數 ---")
print(df["UserName"].fillna("(null)").value_counts().to_string())

work2 = work[work["gap_sec"] > 0]
print("\n--- 排除 0 秒後 ---")
print("intervals:", len(work2))
print(work2["gap_sec"].describe(percentiles=[0.5, 0.75, 0.9]).to_string())

print("\n--- 0 秒間隔：樣本 ---")

zero_idx = work.index[work["gap_sec"] == 0]

# 每一個 0 秒間隔，對應「當前這筆」；上一筆是 index-1
sample_rows = []
for i in zero_idx[:20]:  # 先看前 20 組
    prev = df.loc[i - 1]
    cur = df.loc[i]
    sample_rows.append(
        {
            "prev_SID": prev["SID"],
            "cur_SID": cur["SID"],
            "recTime": cur["recTime"],
            "prev_Goods": prev.get("GoodsNo", ""),
            "cur_Goods": cur.get("GoodsNo", ""),
            "prev_supplier": prev["supplier"],
            "cur_supplier": cur["supplier"],
            "SID_diff": int(cur["SID"]) - int(prev["SID"]),
        }
    )

sample_df = pd.DataFrame(sample_rows)
print(sample_df.to_string(index=False))

print("\n--- 0 秒：同一秒連打幾筆 ---")
same_second_size = (
    df.groupby("recTime", dropna=False)
    .size()
    .value_counts()
    .sort_index()
)
print(same_second_size.head(15).to_string())

work2 = work[work["gap_sec"] > 0]
print("\n--- 排除 0 秒後 ---")
print("intervals:", len(work2))
print(work2["gap_sec"].describe(percentiles=[0.5, 0.75, 0.9]).to_string())
print("total hours:", round(work2["gap_sec"].sum() / 3600, 1))


work2 = work[work["gap_sec"] > 0].copy()
work2["month"] = work2["rec_dt"].dt.to_period("M")

by_month = (
    work2.groupby("month")
    .agg(
        intervals=("gap_sec", "size"),
        median_sec=("gap_sec", "median"),
        mean_sec=("gap_sec", "mean"),
        total_hours=("gap_sec", lambda s: s.sum() / 3600),
    )
)
print("\n--- 按月 ---")
print(by_month.round(2).to_string())
print("\n12 個月合計 hours:", round(by_month["total_hours"].sum(), 1))
print("平均每月 hours:", round(by_month["total_hours"].mean(), 1))


by_sup = (
    work2.groupby("supplier", dropna=False)
    .agg(
        intervals=("gap_sec", "size"),
        median_sec=("gap_sec", "median"),
        total_hours=("gap_sec", lambda s: s.sum() / 3600),
    )
    .sort_values("total_hours", ascending=False)
    .head(15)
)
print("\n--- 供應商（排除 0 秒）---")
print(by_sup.round(2).to_string())


sess = (
    df.groupby("session_id")
    .agg(
        lines=("SID", "size"),
        suppliers=("supplier", "nunique"),
        duration_min=("rec_dt", lambda s: (s.max() - s.min()).total_seconds() / 60),
    )
)

print("\n--- session 長相 ---")
print("sessions:", len(sess))
print("只有 1 家供應商的 session 比例:", round((sess["suppliers"] == 1).mean(), 3))
print("\n每段行數:")
print(sess["lines"].describe(percentiles=[0.5, 0.75, 0.9]).to_string())
print("\n每段幾家供應商:")
print(sess["suppliers"].value_counts().sort_index().head(10).to_string())
print("\n每段時長（分鐘）:")
print(sess["duration_min"].describe(percentiles=[0.5, 0.75, 0.9]).to_string())

df = df.sort_values(["session_id", "rec_dt", "SID"]).copy()
df["supplier_change"] = (
    df.groupby("session_id")["supplier"].shift() != df["supplier"]
)
df["doc_id"] = df["supplier_change"].cumsum()

docs = (
    df.groupby("doc_id")
    .agg(
        session_id=("session_id", "first"),
        supplier=("supplier", "first"),
        lines=("SID", "size"),
        duration_min=("rec_dt", lambda s: (s.max() - s.min()).total_seconds() / 60),
    )
)

print("\n--- 供應商連續區塊（約＝一張單）---")
print("documents:", len(docs))
print("平均每月張數:", round(len(docs) / 12, 1))
print("\n每張行數:")
print(docs["lines"].describe(percentiles=[0.5, 0.75, 0.9]).to_string())
print("\n每張時長（分鐘）:")
print(docs["duration_min"].describe(percentiles=[0.5, 0.75, 0.9]).to_string())
print("\n時長合計 hours:", round(docs["duration_min"].sum() / 60, 1))

def bucket(lines: int) -> str:
    if lines <= 3:
        return "短單 1-3行"
    if lines <= 10:
        return "中單 4-10行"
    return "長單 11行以上"

docs["bucket"] = docs["lines"].map(bucket)

by_len = (
    docs.groupby("bucket")
    .agg(
        張數=("lines", "size"),
        行數合計=("lines", "sum"),
        時長小時=("duration_min", lambda s: s.sum() / 60),
    )
)

# 讓三類按短→長排
order = ["短單 1-3行", "中單 4-10行", "長單 11行以上"]
by_len = by_len.reindex(order)

print("\n--- 依單據長度 ---")
print(by_len.round(2).to_string())
print("\n長單佔時長比例:", round(by_len.loc["長單 11行以上", "時長小時"] / by_len["時長小時"].sum(), 3))

long_docs = docs[docs["lines"] >= 11]

by_sup_long = (
    long_docs.groupby("supplier")
    .agg(
        張數=("lines", "size"),
        行數=("lines", "sum"),
        時長小時=("duration_min", lambda s: s.sum() / 60),
    )
    .sort_values("時長小時", ascending=False)
    .head(10)
)

print("\n--- 長單供應商 Top 10 ---")
print(by_sup_long.round(2).to_string())
print("\n長單總張數:", len(long_docs))
print("Top 3 佔長單時長:", round(by_sup_long.head(3)["時長小時"].sum() / long_docs["duration_min"].sum() * 60, 3))