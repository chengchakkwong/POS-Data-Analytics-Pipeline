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
- ~~若有需要，可在 `docs/使用說明.md` 或 README 補充 Firebase 兩大 collection（products / replenishment）的用途與欄位說明。~~ → 已於 2025-03-15 新增獨立說明文件並於使用說明中連結。

---

## 2025-03-15 新增 Firestore 雙 Collection 設計與安全說明文件

### 今天做了什麼

依 **docs/DEV_SOP.md** 開發後流程，為「products / replenishment 雙 collection」設計補上正式說明文件，並在使用說明中加入連結，方便查閱與交接。

### 改了什麼

1. **新增 `docs/Firestore_雙Collection設計與安全說明.md`**  
   - 說明設計目的與兩 collection 用途、讀取對象。  
   - 說明為何拆成兩個 collection：兩套獨立 Security Rules、前端讀取效能、敏感資料保護。  
   - 資料對照表（主要欄位與來源）。  
   - Firestore Security Rules 設定建議（products / replenishment 範例）。  
   - 與專案檔案對照（POS_Sync_Tool、firebase_service、replenishment_service）。

2. **更新 `docs/使用說明.md`**  
   - 在「二之一、Firebase 同步」的 Firestore 兩大 collection 對照表後，新增一句設計說明並連結至上述文件。

### 為什麼這樣改

- 雙 collection 設計（非冗餘、依用途與權限拆分）需要可對照的說明，方便後續設定 Security Rules 與交接。  
- 依 DEV_SOP「開發後」：有文件或規格變更時更新對應文件，並在 DevLog 記錄本次變更。

### 下次要做的優化

- 若實際在 Firebase Console 調整過 Security Rules，可把最終規則片段補進 `Firestore_雙Collection設計與安全說明.md` 或另存為範例檔供部署參考。

---

## 2025-03-15 統一補貨欄位英文命名與合併 Firebase 上傳區塊

### 今天做了什麼

將 replenishment 的補貨欄位改為英文以統一變數規則，並在 `POS_Sync_Tool` 的同一區塊內同時上傳 products 與 replenishment，移除重複的補貨上傳區塊；同步更新相關文件與 Firestore 寫入方式（整份覆寫以清除舊中文欄位）。

### 改了什麼

1. **`replenishment_service.py`**  
   - `第一次入貨量` → `FirstOrderQty`，`Note描述` → `NoteDescription`（`REPLENISHMENT_COLS` 與程式內欄位命名、註解一致）。

2. **`firebase_service.py`**  
   - `_generate_replenishment_hash()` 改為使用 `FirstOrderQty`、`NoteDescription`。  
   - `upload_replenishment_data()` 改為 `batch.set(..., merge=False)`，整份覆寫 document，下次同步後 Firestore 僅保留英文欄位，舊中文欄位會被清除。

3. **`POS_Sync_Tool.py`**  
   - 在「上傳到 Firebase」單一區塊內：先 `upload_stock_data(df_final)`，再 `prepare_replenishment(df_stock)` → `upload_replenishment_data(df_repl)`，log 改為「products + replenishment」。  
   - 刪除原先獨立的補貨上傳區塊（約 71–85 行），避免重複上傳。

4. **文件**  
   - `docs/Firestore_雙Collection設計與安全說明.md`、`docs/使用說明.md` 中 replenishment 欄位表與說明改為 `FirstOrderQty`、`NoteDescription`。

### 為什麼這樣改

- 專案其餘欄位多為英文，補貨欄位改英文可統一命名、方便維護與前端對接。  
- 上傳邏輯集中在一處，流程較清晰；replenishment 用整份覆寫可一次遷移既有已上傳的中文欄位至英文，無須另寫遷移腳本。

### 下次要做的優化

- 若前端或其他服務曾讀取 `第一次入貨量`、`Note描述`，需改為讀取 `FirstOrderQty`、`NoteDescription`。  
- 在 `replenishment_service` 或後續流程中擴充補貨量建議、安全庫存等計算後再寫入 `replenishment`（延續 2025-03-14 待辦）。

---

## 2025-03-15 新增專案心路歷程與架構決策文件

### 今天做了什麼

依整理好的「專案心路歷程與最終解決方案」大綱，新增 **`docs/專案心路歷程與架構決策.md`**，並在 Firestore 說明、DEV_SOP、使用說明中補上連結或對照，讓交接與回顧時有一份高層次的決策脈絡可查。

### 改了什麼

1. **新增 `docs/專案心路歷程與架構決策.md`**  
   - 起點：資料庫效能與客戶端計算（預測＋庫存同 document、1 次讀取）。  
   - 進階：權限隔離與 Fan-out／雙 collection 設計。  
   - 深水區：新品冷啟動、Min/Multiple、歷史雜訊與演算法限制。  
   - 最終解法：人機協作 (Human-in-the-Loop)、系統猜測與一鍵贊同／修改。

2. **`docs/Firestore_雙Collection設計與安全說明.md`**  
   - 一、設計目的：補上 Fan-out 寫入術語與心路歷程連結。  
   - 二、2：補上「預測量與庫存同 document、1 次讀取、客戶端計算」說明。  
   - 新增「六、延伸閱讀」連結至心路歷程文件。

3. **`docs/DEV_SOP.md`**  
   - 四、本 SOP 與現有文件對照表：新增一列「架構與決策脈絡」→ `docs/專案心路歷程與架構決策.md`。

4. **`docs/使用說明.md`**  
   - 在 Firebase 同步一節的設計說明處，補上心路歷程文件連結（可選，見下次要做的優化）。

### 為什麼這樣改

- 心路大綱是專案演進的單一真相來源，獨立成文件方便新人理解「為什麼這樣設計」、與現有技術文件對照。  
- Firestore 說明與 DEV_SOP 補連結後，開發前／查架構時會自然導向心路歷程。

### 下次要做的優化

- 若實作「冷啟動／人機協作」的 `guessed_min`、`guessed_multiple` 或 Web 一鍵贊同／修改，在 DevLog 與心路歷程文件中補上實作紀錄，並在使用說明中連結操作步驟。  
- 可在 `docs/使用說明.md` 開頭或「二之一、Firebase 同步」旁加一句：專案演進與架構決策見 **docs/專案心路歷程與架構決策.md**。

---

## 2025-03-15 報告撰寫指南納入 SOP 與文件對齊

### 今天做了什麼

將 **`docs/報告撰寫規格與風格指南.md`** 納入 **DEV_SOP** 的開發後流程與文件對照表，並在指南內補上與本專案 docs 的對照與 DEV_SOP 雙向連結。

### 改了什麼

1. **`docs/DEV_SOP.md`**  
   - 三、1 要寫／要更新的文件：新增一列「專案報告／對外文件」→ 對照 `docs/報告撰寫規格與風格指南.md`。  
   - 三、2 開發後對照表：新增一項「若有產出專案報告或對外文件，已依報告撰寫指南檢核格式與必填清單」。  
   - 四、本 SOP 與現有文件對照：新增一列「報告／對外文件撰寫」→ 報告撰寫指南；開發後寫什麼一列補上「報告撰寫指南」。

2. **`docs/報告撰寫規格與風格指南.md`**  
   - 新增 8.3「與本專案 docs 對照」：主報告／追蹤模板、安全設計／權限、架構決策脈絡對應至 `專案追蹤模板_老闆版.md`、`老闆版_WebApp_權限與實施報告.md`、`Firestore_雙Collection設計與安全說明.md`、`專案心路歷程與架構決策.md`。  
   - 補上與 DEV_SOP 的連結（開發後 → 專案報告／對外文件）。  
   - 版本歷史新增 v1.2。

### 為什麼這樣改

- 產出專案報告時有明確對照的規格與檢查項，與開發流程一致。  
- 指南內的「主報告、追蹤模板、安全設計」與實際檔名對齊，方便維護與查找。

---

## 2025-03-15 資料庫離線；待辦：從 goods_stock_movement 區分盤點與入貨

### 今天做了什麼

- 資料庫離線，無法繼續查詢與測試。
- 鎖定目標：利用 **goods_stock_movement**（或相關庫存異動表）的歷史紀錄，辨識並**分開 movement 的種類**——特別是 **盤點 (inventory count)** 與 **入貨 (inbound)**，因為後續估計 **Min / Multiple**（最低起訂量、下單倍數）只需要以**入貨**為依據，盤點會造成雜訊。

### 下次要做的（明天早上繼續）

- [ ] 連上 DB 後，查詢並檢視 **goods_stock_movement**（或對應表）的欄位與樣本資料。
- [ ] 找出可區分「盤點」與「入貨」的 **pattern**（例如：異動類型代碼、備註、數量正負、來源單據等）。
- [ ] 在程式或分析中**分開兩類 movement**，只取入貨紀錄用來推算 Min / Multiple，避免盤點資料干擾。
- [ ] 驗證分類結果後，再接到補貨／冷啟動邏輯（見專案心路歷程「人機協作」中的 guessed_min、guessed_multiple）。
