# DevLog 開發日誌

記錄開發過程、決策、嘗試與待辦。

---

## 格式說明

- 隨手記、時間序即可。
- 可寫：今天做了什麼、為什麼這樣改、踩了什麼雷、下次要做的優化。

---

## 2025-03-13 大幅提升 `sync_daily_sales_partitioned` 效能

### 今天做了什麼

在 `pos_service.py` 的 `sync_daily_sales_partitioned` 做了三處關鍵優化，讓增量同步從「每次都讀寫全量」變成「只動到有變動的月份」，大幅減少 I/O 與記憶體。

### 改了什麼（三個優化點）

1. **優化點 1：只讀一欄找最後日期（Columnar 讀取）**  
   判斷快取最後同步日時，不再 `pd.read_parquet(cache_dir)` 把幾十萬筆全部讀進來，改為只讀 `columns=["rDate"]`，用 Parquet 的欄位式儲存只掃一欄，瞬間得到 `max(rDate)`，再推算 `sync_start`。

2. **優化點 2：只讀受影響的分區**  
   新資料進來後先算 `affected_partitions`（牽涉到的 year/month），用 `pd.read_parquet(cache_dir, filters=filters)` 只讀這些分區的舊資料來合併，沒變動的歷史月份完全不讀。

3. **優化點 3：只覆寫受影響的分區**  
   寫回時使用 `existing_data_behavior='delete_matching'`，PyArrow 會只刪除/覆寫「這次有變動」的 year/month 分區，其餘分區原封不動，避免「讀全量 → 合併 → 寫全量」的沉重寫入。

### 為什麼這樣改

- 資料量變大後，每次全量讀寫 Parquet 會又慢又吃記憶體。  
- 每日增量通常只影響最近幾天/當月，用「分區 + 條件讀寫」可以讓 I/O 和記憶體都只跟「受影響的月份」成正比。

### 其他小改動

- 用 `time.perf_counter()` 在函式開頭/結尾計時，並在 log 輸出本次處理筆數與耗時（秒），方便之後比對效能。

---
