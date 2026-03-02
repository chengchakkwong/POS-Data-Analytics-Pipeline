# 老闆版 WebApp 權限與實施報告

文件版本：v1.0  
更新日期：2026-03-02  
作者：ChakKwong（可自行修改）

---

## 1. 文件目的與適用對象

本報告用於定義「同一個 Vercel WebApp，分員工版與老闆版」的落地方案，並提供可執行的安全與開發計畫。  
適用對象：

- 專案負責人（你自己）
- 決策者（老闆）
- 外部專業人士（HR／面試官／合作工程師）

本文件重點是「可執行、可追蹤、可展示」，兼顧實務與專業呈現。

---

## 2. 背景與現況摘要

### 2.1 既有系統

- POS 資料由 Python pipeline 同步與分析（庫存、分類、預測）。
- 已有 `POS_Sync_Tool.py` 每日排程，將公開貨品資料上傳 Firebase。
- 現有 WebApp 為 Vite + React SPA，部署於 Vercel，並透過 `api/*.js` 提供後端 API。
- 前端目前無角色分流，所有通過 token 驗證者權限接近相同。

### 2.2 本次目標

在**不拆成兩個 App**的前提下，完成以下能力：

1. 同一個 WebApp 提供兩種視圖：員工版與老闆版  
2. 建立 `employee / owner / admin` 權限模型  
3. 安全提供老闆資料（成本、建議採購量、策略）  
4. 支援老闆版篩選、Excel 匯出、一鍵複製  
5. 保留後續調整「建議採購量邏輯」的彈性

---

## 3. 方案選型與結論

### 3.1 採用方案

採用「**同一個 Vercel App + 不同路由 + 角色權限控制**」。

建議路由規劃：

- `/app`：員工版（掃描、查庫存、回報）
- `/owner`：老闆版（成本、建議採購量、策略、匯出）

### 3.2 採用理由

- 降低開發與維運成本（單一 repo、單一部署）
- 可快速上線 MVP（先做最小可用版本）
- 容易做履歷展示（同系統多角色安全設計）

### 3.3 風險與控制

- 風險：若僅前端隱藏功能，資料仍可能外洩  
- 控制：API 層必做角色檢查 + 資料分層 + 最小權限回傳

---

## 4. 權限模型（RBAC）

### 4.1 角色定義

- `employee`：作業員，使用日常查詢與回報功能
- `owner`：決策者，可查看成本與建議採購量
- `admin`：系統管理者，負責權限、維運、稽核與手動作業

### 4.2 為何保留 admin

若未來把所有高權限都放 `owner`，會混淆「商業決策」與「系統管理」。  
保留 `admin` 可將維運權限隔離，減少誤操作與權限濫用風險，也更符合企業治理與展示專業形象。

### 4.3 權限矩陣（第一版）

| 功能/資源 | employee | owner | admin |
|---|---:|---:|---:|
| 查看一般庫存資料（名稱/分類/庫存） | ✅ | ✅ | ✅ |
| 查看供應商 | ✅（可視需求調整） | ✅ | ✅ |
| 查看成本（LastInCost/AvgCost） | ❌ | ✅ | ✅ |
| 查看建議採購量（Suggested_Order） | ❌ | ✅ | ✅ |
| 查看 ABC/XYZ 與策略標籤 | ❌（或只看部分） | ✅ | ✅ |
| 匯出 Excel（老闆欄位） | ❌ | ✅ | ✅ |
| 一鍵複製給供應商 | ❌ | ✅ | ✅ |
| 調整系統權限 / 手動重跑工作 | ❌ | ❌ | ✅ |

---

## 5. 資料分層與儲存策略

### 5.1 建議集合（collection）分層

建議在 Firebase 以集合分層，不同用途分開儲存：

- `stock_public`：員工版可見欄位
  - 例：`ProductCode`, `Name`, `CurrStock`, `Category`, `Supplier`, `RetailPrice`
- `stock_owner`：老闆版專用欄位
  - 例：`LastInCost`, `AvgCost`, `ABC_XYZ`, `Suggested_Order`, `Target_Stock`, `StrategyTag`

> 若你想維持既有 `products` 集合，也可先「同集合不同欄位 + API 過濾回傳」，但長期仍建議分層，維護更清楚。

### 5.2 建議採購量資料欄位（可追蹤版）

建議老闆版每筆資料新增以下欄位，方便未來調整邏輯時追溯：

- `suggested_order`（建議採購量）
- `logic_version`（例如 `v1-rule-based`）
- `as_of_date`（資料基準日）
- `updated_at`
- `source_job_id`（對應哪次排程產生）

---

## 6. API 與安全控制設計

### 6.1 安全原則

1. **不只做前端控制**：真正授權在 API  
2. **最小權限回傳**：employee API 不回傳成本/採購建議  
3. **敏感操作必記錄**：所有高權限讀寫留存稽核資訊  
4. **前後端雙層防護**：前端路由守衛 + 後端 role check

### 6.2 API 層建議

- 在共用驗證工具中增加 `requireRole(allowedRoles)`  
- owner 專用端點只允許 `owner/admin`
- `PATCH` 類操作依角色限制可寫欄位

建議端點示意：

- `GET /api/inventory`：employee 基礎欄位
- `GET /api/owner/inventory`：owner/admin 完整欄位
- `GET /api/owner/export?format=xlsx`：owner/admin 匯出
- `POST /api/auth`：登入，JWT 載入 `role`
- `POST /api/verify`：驗證 token，回傳角色

### 6.3 Firestore Rules 原則

- 將規則檔納入版控（避免只放 Console）
- 若敏感資料均經 API，建議 client 端不直接讀寫敏感集合
- 即使有 Rules，也不能省略 API 角色檢查

---

## 7. 功能規格（老闆版）

### 7.1 篩選與檢視

- 篩選條件：
  - 產品分類（Category）
  - 供應商（Supplier）
  - ABC/XYZ
  - 策略標籤（StrategyTag）
- 表格欄位（預設）：
  - `Date`
  - `Supplier`
  - `ProductCode`
  - `Name`
  - `Category`
  - `ABC_XYZ`
  - `CurrStock`
  - `Target_Stock`
  - `Suggested_Order`
  - `LastInCost`
  - `AvgCost`
  - `RetailPrice`
  - `Estimated_Order_Cost`
  - `StrategyTag`
  - `Remark`

### 7.2 一鍵複製（供應商格式）

預設先用簡潔格式，每行一品項：

`商品名 | 編號 | 建議量`

可後續擴充為：

`商品名 | 編號 | 建議量 | 參考成本 | 預估小計`

### 7.3 Excel 匯出

- 匯出範圍：目前篩選結果
- 建議檔名格式：`owner_inventory_plan_YYYYMMDD.xlsx`
- 匯出欄位與畫面欄位保持一致，降低溝通成本

---

## 8. 排程與資料更新

### 8.1 已有排程

- `POS_Sync_Tool.py`：每日執行（已運作）

### 8.2 建議更新策略（每日）

每日流程（建議）：

1. 更新公開庫存資料（員工端）
2. 計算/更新建議採購量（老闆端）
3. 寫入 `stock_owner` 並附 `as_of_date`, `logic_version`
4. API 提供老闆端查詢與匯出

> 目前單一公司，先不做多公司複雜化。未來如擴張，新增 `company_id` 欄位即可。

---

## 9. 一週 MVP 落地計畫（可直接追進度）

| 天數 | 任務 | 驗收標準 |
|---|---|---|
| Day 1 | JWT 加入 role、verify 回傳 role | 登入後可正確辨識 employee/owner/admin |
| Day 2 | API 增加 `requireRole`，補 owner 專用 endpoint | employee 無法呼叫 owner API |
| Day 3 | Firebase 資料分層（public/owner）與欄位整理 | 老闆資料可查，員工看不到敏感欄位 |
| Day 4 | 前端路由分流（`/app`, `/owner`）與 UI 守衛 | 非 owner/admin 進 `/owner` 會被阻擋 |
| Day 5 | 老闆儀表板（篩選、表格、策略） | 可按分類/供應商/ABC_XYZ 查詢 |
| Day 6 | Excel 匯出 + 一鍵複製 | 可下載報表並複製供應商文字 |
| Day 7 | 權限整測、錯誤處理、文件更新 | 完成驗收清單與風險確認 |

---

## 10. 主要風險與對策

| 風險 | 等級 | 對策 |
|---|---|---|
| API 未做 role 檢查造成資料外洩 | 高 | 所有敏感 API 強制 `requireRole` |
| 前端只做 UI 隱藏，未做後端授權 | 高 | 權限決策全部移到 API 層 |
| token 長期有效導致濫用風險 | 中 | 設定到期時間、強制重新驗證 |
| 建議採購量邏輯調整後無法追溯 | 中 | 保存 `logic_version` 與 `as_of_date` |
| 功能增加後文件未同步更新 | 中 | 每週固定更新 Decision Log + 里程碑 |

---

## 11. 成功指標（KPI）

建議以「安全 + 效率 + 可用性」三面向衡量：

- 安全面
  - owner 欄位對 employee API 回傳次數 = 0
  - 權限測試案例通過率 = 100%
- 效率面
  - 老闆查詢到匯出的操作時間 < 2 分鐘
  - 每日資料更新成功率 > 99%
- 可用性
  - 匯出成功率 > 98%
  - 一鍵複製內容可直接對外溝通（格式滿意度）

---

## 12. 對外展示（HR/面試）建議說法

可用以下敘述總結專案能力：

1. 在單一 WebApp 內完成多角色安全分流（employee/owner/admin）  
2. 以 API 授權與資料分層解決敏感商業數據保護問題  
3. 建立每日自動更新與決策輸出（建議採購量、策略報表）  
4. 提供可直接落地的管理功能（篩選、匯出、供應商溝通格式）

---

## 13. 後續可演進項目（第二階段）

- 建議採購量由規則版升級為模型版（可 A/B 比較）
- 增加操作稽核頁（admin）
- 建立異常警報（資料缺漏、更新失敗）
- 納入多公司架構（`company_id` 分區）

---

## 14. 文件維護規範

每次有下列變更時，必須更新本文件版本：

- 權限模型變更
- API 權限矩陣變更
- 建議採購量邏輯變更
- 匯出格式變更

建議版本格式：`v主版.次版`（例如 `v1.1`）

---

## 15. 文件拆分策略（現階段與後續）

### 15.1 現階段（短期）

為了降低維護負擔與提升更新速度，現階段維持 2 份核心文件：

- `docs/老闆版_WebApp_權限與實施報告.md`（主報告）
- `docs/專案追蹤模板_老闆版.md`（追蹤模板）

此方式可避免在功能快速調整期出現多份文件不同步問題。

### 15.2 後續（功能穩定後）

當 owner 版主要功能上線且權限設計穩定後，再拆分為 4 份文件：

- `docs/老闆版規劃書.md`
- `docs/權限與資料安全設計.md`
- `docs/里程碑與進度追蹤.md`
- `docs/Decision-Log.md`

拆分條件（滿足任一組）：

1. 路由分流、RBAC、owner API、匯出功能皆已上線，且連續兩週無重大變更。  
2. 需要正式對外溝通（如管理層審核、面試展示、跨部門交接）且文件讀者增加。

---

## 附錄 A：本版預設決策清單

- 老闆資料更新頻率：每日一次
- 權限模型：employee / owner / admin
- 建議採購量 = 進貨量（可後續調整邏輯）
- WebApp：同一個 Vercel App，不同路由分流
- 後端：既有 Vercel API 延伸，不新增第二套系統

