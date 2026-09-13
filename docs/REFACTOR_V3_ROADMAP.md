# Pipeline v3 重構路線圖

分支：`refactor/pipeline-v3`

## 目標架構（簡記）

- 定時（目標每 2 小時）：SQL → stock → Firestore（products / replenishment / inbound）
- 半月：本機 sales Parquet → ABC / 預測 → 只上傳結果欄位
- Sales 原始明細不上雲；分析在本地，省 Firestore quota

## 已完成

- [x] 開分支 `refactor/pipeline-v3`
- [x] 套件骨架：`src/pos_pipeline`、`cli`、`jobs` stub
- [x] `pyproject.toml` + `pip install -e .`
- [x] `config.py`（根目錄、`data/processed`、stock CSV）
- [x] `database/connection.py`（env、engine、check、execute_query）
- [x] `extraction/stock.py`（查詢、清洗、存 CSV）
- [x] `daily`：連線 → 抽 stock → 存檔
- [x] commit：v3 骨架 + stock extract
- [x] 共識：daily 上雲以 stock 為主；sales 留給本機分析

## 進行中：每日 Job → Firestore

- [x] `delivery_to_firebase/` 套件（client / products / replenishment / sync_state）
- [x] `FIREBASE_KEY_PATH` + 檢查金鑰是否存在
- [x] 初始化 Firebase Admin
- [x] 上傳 `products`（最小版 + Firestore hash 增量）
- [x] 上傳 `replenishment`（最小版 + Firestore hash 增量）
- [x] 串進 `jobs/daily.py`（全量呼叫已就緒；hash 相同則跳過寫入）
- [x] 上傳 `inbound_movements`（SID watermark + 本機驗證：344 筆）
- [ ] 寫入「上次同步時間」供 App 顯示
- [x] `.gitignore` 加入 `*.egg-info/`

## 稍後：半月 Analytics

- [ ] 本機 sales → 增量 Parquet
- [ ] ABC/XYZ
- [ ] Target Stock
- [ ] 結果欄位上傳 Firestore
- [ ] `python -m pos_pipeline.cli analytics`

## 再之後：Docker / 排程

部署說明：[`CLOUD_RUN_DAILY.md`](CLOUD_RUN_DAILY.md)

- [x] `Dockerfile.daily` + `.dockerignore` + `cloudbuild.yaml`
- [x] Artifact Registry（`us-central1` / `pos-pipeline`）
- [x] Cloud Build 產出 `daily:v0.1`（本機不需 Docker Desktop）
- [x] Cloud Run Job `pos-pipeline-daily` + Secret Manager
- [x] Cloud Scheduler：`0 9-19/2 * * *`（Asia/Hong_Kong；09:00–19:00 每 2 小時）
- [ ] 首次排程成功驗證（待 POS SQL 營業時段）
- [ ] Firestore 記錄 `lastSyncedAt`，App 顯示給使用者
- [ ] `Dockerfile.analytics`（半月）

## 刻意先不做

- [ ] 提交離線 demo / sample_data
- [ ] 提交 `.cursor`
- [ ] 改 ABC／預測公式（先行為對齊舊版）
- [ ] 把 sales 明細上傳 Firebase

## 對應舊檔（方便對照）

| 舊 | 新（方向） |
|----|------------|
| `pos_system_v2.py` / `pos_service.py` | `extraction/` + `jobs/daily.py` |
| `db_utils.py` | `database/connection.py` |
| `POS_Sync_Tool.py` + `firebase_service.py` | `jobs/daily.py` + `delivery_to_firebase/` |
| `abc_xyz_analysis.py` / `inventory_forecast.py` | 之後的 analytics job |