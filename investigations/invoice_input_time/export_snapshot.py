"""Export 12 complete months of inbound stock movements."""

from pathlib import Path

from pos_pipeline.config import PROCESSED_DIR
from pos_pipeline.database.connection import execute_query


START_TIME = "20250901000000"
END_TIME = "20260901000000"

OUTPUT_PATH = (
    PROCESSED_DIR
    / "GoodsStockMovement_20250901_20260831.csv"
)

sql = """
SELECT
    *,
    CONVERT(varchar(18), TS, 1) AS TS_hex
FROM dbo.GoodsStockMovement
WHERE MoveTypeID = 1
  AND recTime >= :start_time
  AND recTime < :end_time
ORDER BY SID
"""

print("正在下載 GoodsStockMovement...")

df = execute_query(
    sql,
    params={
        "start_time": START_TIME,
        "end_time": END_TIME,
    },
)

# TS 是 SQL Server rowversion 二進位值。
# 已經轉成可保存的 TS_hex，因此移除原始二進位欄位。
if "TS" in df.columns:
    df = df.drop(columns=["TS"])

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig",
)

print("\n--- 匯出結果 ---")
print("rows:", len(df))
print("columns:", len(df.columns))

if not df.empty:
    print("first SID:", df["SID"].min())
    print("last SID:", df["SID"].max())
    print("first recTime:", df["recTime"].min())
    print("last recTime:", df["recTime"].max())
    print("duplicate SID:", int(df["SID"].duplicated().sum()))

file_size_mb = Path(OUTPUT_PATH).stat().st_size / 1024 / 1024

print("saved:", OUTPUT_PATH)
print("size MB:", round(file_size_mb, 2))