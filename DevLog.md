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

## 2025-03-14 新增補貨建議 collection（replenishment）與 Transform 層

### 今天做了什麼

在「從 DB 抽資料上傳 Firebase」的流程中，新增**第二個 Firestore collection**，專門放補貨決策用的敏感資訊（成本、進貨來源、Note 解析），並抽成獨立的 **Transform 層**（`replenishment_service.py`），方便之後擴充補貨邏輯。

### 改了什麼

1. **新增 `replenishment_service.py`（Transform 層）**  
   - `prepare(df_stock)`：接收庫存主檔，解析 `Note` →「第一次入貨量」（regex 抽獨立數字）、「Note描述」（其餘文字）。  
   - 只輸出補貨用欄位：`ProductCode`, `LastInCost`, `AvgCost`, `InboundLocation`, `第一次入貨量`, `Note描述`。

2. **`firebase_service.py`**  
   - 新增 `_generate_replenishment_hash()`、`upload_replenishment_data(df)`。  
   - 上傳至 collection **`replenishment`**，快取 key 為 `repl:{ProductCode}`，邏輯與現有 products 上傳一致（只更新有變動、batch 400）。

3. **`POS_Sync_Tool.py`**  
   - 同一次 `get_stock_master_data()` 取得的 `df_stock` 先上傳 **products**（基本資訊），再經 `prepare_replenishment(df_stock)` 加工後上傳 **replenishment**（補貨建議用）。

### 為什麼這樣改

- 產品基本資訊與成本/備註分開存放，方便權限與前端分離（公開 vs 內部補貨建議）。  
- 補貨邏輯獨立成 `replenishment_service`，之後加「建議補貨量」「安全庫存」等計算時，只改這一層，不汙染 `pos_service` 或 Sync Tool。

### 下次要做的優化

- 在 `replenishment_service` 或後續流程中擴充補貨量建議、安全庫存等計算後再寫入 `replenishment`。  
- 若有需要，可在 `docs/使用說明.md` 或 README 補充 Firebase 兩大 collection（products / replenishment）的用途與欄位說明。

---
