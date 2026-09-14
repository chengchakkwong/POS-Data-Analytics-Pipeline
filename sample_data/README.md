# Sample Data

> 這份目錄是由 `scripts/anonymize_data.py` 產生的**脫敏示範資料**，可以進版本庫。
> 真實資料位於 `data/`（已在 `.gitignore` 排除，不進版本庫）。

## 重新產生

```bash
python scripts/anonymize_data.py \
    --input-dir data/processed \
    --output-dir sample_data \
    --sku-count 200 \
    --seed 42 \
    --time-shift-days 90
```

先用 `--dry-run` 確認無誤再正式寫出：

```bash
python scripts/anonymize_data.py --seed 42 --dry-run
```

## 檔案說明

| 檔案 | 格式 | 說明 |
|------|------|------|
| `stock.csv` | CSV (UTF-8 BOM) | 脫敏後的商品主檔（182 SKU；另有 18 個 sales-only ghost GID） |
| `sales.parquet` | Parquet | 脫敏後的日銷售紀錄（供 pipeline 讀取） |
| `sales.csv` | CSV (UTF-8 BOM) | 同上，供用 Excel 瀏覽 |

## 這份 sample 服務哪些腳本

- `demo_pipeline.py`：離線 demo 入口，固定讀取 `sample_data/` 並輸出到 `demo_output/`
- `tests.test_analytics_job.OfflineAnalyticsPipelineTests`：v3 ABC/XYZ + target-stock 離線整合驗證
- 規格說明：[`docs/ANALYTICS_PIPELINE.md`](../docs/ANALYTICS_PIPELINE.md)
## 跑離線 demo

```bash
python demo_pipeline.py
```

輸出：

```text
demo_output/abc_xyz_analysis.csv
```

目前 seed=42 的 demo summary：

```text
Rows: 183

ABC class counts:
C           103
A            37
B            35
New           5
Excluded      3

XYZ class counts:
Z      138
Y       36
New      5
X        4
```

## 保留了什麼

| 特徵 | 說明 |
|------|------|
| 熱賣 / 中段 / 冷門 SKU 結構 | 確保 ABC 分級有足夠代表性 |
| Misc / Generic 例外商品 | AdjustedCost 邏輯依賴此訊號 |
| Generic demo label | `膠袋徵費`、`五金家品雜項` 會保留為固定示範標籤，用來展示 `Is_Generic=Yes` / `ABC_Class=Excluded` |
| 時間序列節奏 | 週末高峰、週期性波動（整批平移，節奏完整保留） |
| 價格與成本分佈形狀 | 均值和標準差在 ±5% 內 |
| `cost=0` 的例外情況 | 分析邏輯對零成本商品有特殊處理 |

## 刻意破壞了什麼（不可逆）

| 欄位 | 處理方式 |
|------|----------|
| `GoodsID` | 全部重新編號（10000 起跳） |
| 商品名稱 | 用假名取代 |
| 條碼 | 用假 EAN-13 取代；misc 統一用 `0000000000000` |
| `ProductCode` | 用合成 SKU code 取代 |
| `Supplier` | 用 `Supplier-0001` 這類合成標籤取代 |
| `InboundLocation` | 用 `LOC-0001` 這類合成標籤取代 |
| `Category` | 用 `CAT-0001` 這類合成標籤取代 |
| 絕對日期 | 整批平移固定天數 |
| 個別價格 | 每 SKU 乘以獨立亂數倍率（0.8–1.2） |

## 已知設計行為（非錯誤）

- **`XYZ_Class` 不含 `Excluded`**：`XYZ` 軸只會是 `X / Y / Z / New`。
  被 `ABC_Class = Excluded` 標記的 generic 商品，仍會依其銷售波動落到 `X / Y / Z`，不會自動寫成 `Excluded`。判斷是否排除請以 `ABC_Class` 或 `Is_Generic` 為準。
- **`FirstSaleDate` 部分為空**：分析使用 outer join，stock 有但分析期內無銷售紀錄的 SKU 會沒有 `FirstSaleDate`，這些通常落在 `ABC_Class = C`。屬預期現象，不視為資料錯誤。

---

*由 `scripts/anonymize_data.py` 產生，seed=42，sku-count=200，time-shift-days=90*
