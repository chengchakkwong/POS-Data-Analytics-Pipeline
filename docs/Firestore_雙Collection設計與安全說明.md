# Firestore 雙 Collection 設計與安全說明

本文件說明 **POS_Sync_Tool** 為何將資料寫入兩個 Firestore collection（`products` 與 `replenishment`），以及如何依此設計設定安全規則，兼顧前端效能與敏感資料保護。

---

## 一、設計目的

同步腳本從 POS 抓取「商品庫存主檔」後，會寫入兩個 collection：

| Collection       | 用途                     | 主要讀取者           |
|------------------|--------------------------|----------------------|
| **products**     | 商品基本資訊、庫存、售價 | 前端賣場／查詢介面   |
| **replenishment**| 補貨建議、成本、備註     | 內部補貨／採購決策   |

從資料流角度可視為 **Fan-out 寫入**（同一來源寫入多處）；本專案依「用途」與「權限」拆成兩 collection，各自服務不同角色與場景。架構決策脈絡（含為何採客戶端計算、為何拆兩 collection）見 **[專案心路歷程與架構決策.md](專案心路歷程與架構決策.md)**。

---

## 二、為何拆成兩個 Collection？

### 1. 兩套完全獨立的安全規則

- Firestore 的 **Security Rules 是依 collection 設定的**。
- **products**：可開放給需要「商品名、價格、庫存」的前端（店員／顧客端）讀取。
- **replenishment**：可僅開放給後台、管理端，或只允許後端／特定 role 讀寫。
- 兩邊規則互不影響，不必在同一份資料上做複雜的欄位級權限。

### 2. 前端讀取效能與客戶端計算

- **products** 內將預測量與庫存量放在同一 document，Web 端 **1 次讀取** 即可在前端計算「庫存缺口」，兼顧 Firestore 讀取成本與即時性（客戶端計算）。
- 一般畫面只需「商品 + 庫存 + 售價」，只查 **products** 即可，文件小、查詢快。
- 補貨建議（成本、入貨量、備註等）只在管理介面需要，不必讓每個客戶端都載入，減少流量與延遲。

### 3. 私密／敏感資料保護

- 成本（LastInCost、AvgCost）、供應商、內部備註等放在 **replenishment**。
- 透過 Security Rules 讓 **replenishment** 僅對後台或特定角色開放，一般使用者連讀都讀不到，從規則層就做到嚴格保護。

---

## 三、資料對照

| Collection       | 主要欄位（示意） |
|------------------|------------------|
| **products**     | ProductCode, Barcode, Name, CurrStock, RetailPrice, Category, Supplier，以及其他庫存主檔欄位 |
| **replenishment**| 同 products 全部欄位 + LastInCost, AvgCost, InboundLocation, FirstOrderQty, NoteDescription, guessed_min, guessed_multiple |

- **products**：來自 `pos_service.get_stock_master_data()`，經欄位篩選後上傳。
- **replenishment**：同一份庫存主檔經 `replenishment_service.prepare()` 加工（解析 Note、補貨欄位）後上傳，並在後續流程由 `upload_guessed_min_multiple()` 寫入 `guessed_min` / `guessed_multiple`。

> 更新：在目前版本中，`replenishment` 的上游已改為直接帶著 `df_stock` 的全部欄位，再附加補貨欄位後上傳，因此：
>
> - 每一筆 `replenishment/{ProductCode}` document 會同時包含：
>   - 與 `products/{ProductCode}` 相同的一整包商品欄位（例如 ProductCode, Barcode, Name, CurrStock, RetailPrice, Category, Supplier，以及其他庫存主檔欄位）。
>   - 補貨專用欄位：LastInCost, AvgCost, InboundLocation, FirstOrderQty, NoteDescription 等。
>   - 後續估算流程寫入的 guessed_min / guessed_multiple。
> - 好處是：在做補貨決策或權限判斷時，僅查 `replenishment` 一個 collection 即可，不需要再 join `products`。

---

## 四、安全規則設定建議（Firebase Console）

可依實際登入方式（例如 Firebase Auth、自訂 token、Admin only）調整，以下為概念範例。

### products（可對外或店員讀取）

```javascript
// 範例：已登入使用者可讀，僅後端可寫
match /products/{productId} {
  allow read: if request.auth != null;
  allow write: if false;  // 僅透過 Admin SDK（POS_Sync_Tool）寫入
}
```

### replenishment（僅內部／管理端）

```javascript
// 範例：僅管理員或特定 role 可讀寫
match /replenishment/{productId} {
  allow read, write: if request.auth != null && request.auth.token.role == 'admin';
  // 或 allow read, write: if false;  若完全由後端透過 Admin SDK 存取
}
```

實際撰寫規則時請參照 [Firestore Security Rules 文件](https://firebase.google.com/docs/firestore/security/get-started)，並依專案登入與權限設計調整。

---

## 五、Hash 與快取策略（增量上傳）

同步程式為了節省 Firestore 寫入量，使用本地快取檔與 Hash 來判斷「哪些 document 需要重寫」。

- **本地快取檔**：`data/sync_cache.json`
  - 由 `firebase_service.FirebaseManager` 在初始化時讀入 (`_load_cache()`)，結束或同步後寫回 (`_save_cache()`)。
  - 以 `ProductCode` 或自訂 cache key（如 `repl:{ProductCode}`、`repl_min_mult:{ProductCode}`）作為索引。

- **products：`upload_stock_data()`**
  - 透過 `_generate_hash(item)` 產生該筆商品的指紋：
    - 使用「該筆紀錄的全部欄位」，先依欄位名稱排序後，以 `key=value` 形式串接，再用 MD5 計算 Hash。
    - 唯一例外：**忽略 `AvgCost` 欄位**，避免單純成本平均值的更新就造成大量重寫。
  - 若 Hash 與快取中記錄一致，則視為「未變」，略過寫入；否則才加入 batch 寫入 `products/{ProductCode}`。

- **replenishment：`upload_replenishment_data()`**
  - 透過 `_generate_replenishment_hash(item)` 產生補貨紀錄指紋：
    - 規則與 `_generate_hash` 一致：使用該筆補貨紀錄的全部欄位（同樣忽略 `AvgCost`），以 `key=value` 串接後做 MD5。
  - 若 Hash 未變則略過，否則寫入 `replenishment/{ProductCode}`（`merge=True`，只更新本次提供欄位）。

- **guessed_min / guessed_multiple：`upload_guessed_min_multiple()`**
  - 使用 `_generate_min_multiple_hash(item)`，僅依 `ProductCode`、`guessed_min`、`guessed_multiple` 三個欄位判斷是否需要更新，並以 `merge=True` 寫入同一份 `replenishment/{ProductCode}` document。

> 若需要在程式邏輯變更後「強制全量重寫」（例如新增欄位，想讓所有既有 document 也帶上），可以人工刪除 `data/sync_cache.json` 再執行同步工具；這次會視為首次上傳，全部 document 重新寫入，之後仍回到上述的增量同步邏輯。

---

## 五、與本專案檔案對照

| 項目           | 說明 |
|----------------|------|
| 同步入口       | `POS_Sync_Tool.py` |
| 上傳 products  | `firebase_service.FirebaseManager.upload_stock_data()` |
| 上傳 replenishment | `firebase_service.FirebaseManager.upload_replenishment_data()` |
| 補貨資料加工   | `replenishment_service.prepare()` |

操作步驟與指令請見 **[使用說明.md](使用說明.md)** 當中的「二之一、Firebase 同步（POS_Sync_Tool）」一節。

---

## 六、延伸閱讀

- **專案演進與決策脈絡**（起點、權限隔離、冷啟動、人機協作）：[專案心路歷程與架構決策.md](專案心路歷程與架構決策.md)
