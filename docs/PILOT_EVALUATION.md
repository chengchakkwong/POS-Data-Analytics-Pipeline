# Pilot Evaluation (public methodology) / 試行評估（公開方法）

This page describes **how** the single-store pilot is evaluated. It does not publish production Firestore data or raw survey responses.

## What we measure

### System-record KPIs (objective)

Derived from anonymized Firestore exports in the private operational app:

| KPI | Definition |
| --- | --- |
| ABC/XYZ coverage | Share of SKUs with classification labels synced from the pipeline |
| Replenishment gap visibility | SKUs where `Target_Stock` exceeds on-hand stock |
| Inbound SKU match rate | `SUM(arrivalQty)` matches `ChQty` for the same supplier + SKU + bill date window |
| Arrival adoption | Supplier-date combinations with scanned arrival records |
| Issue resolution | Resolved vs open reports; median hours to resolve |

### Internal feedback (self-reported, supplementary)

Anonymous staff/manager surveys capture:

- Self-estimated minutes for lookup or admin workflows (before vs after)
- Usability / traceability ratings
- Examples of manual override cases

Survey results are always reported with invited count, response count `n`, and collection period.

## Evidence boundaries

We **do not** claim the following without new instrumentation:

- Barcode lookup latency
- Scan success rate
- Daily active users
- Actual purchase order quantities (`Final_Order` is computed in UI only)

Target-stock values demonstrate data availability in the snapshot only, not how they were calculated. Per-SKU generation source, model/version, and run timestamp were not retained; the pilot therefore does not establish Prophet/NeuralProphet use or forecast accuracy.

目標庫存數值僅反映快照中資料可用，不代表其計算方式。由於未保存逐 SKU 的生成來源、模型／版本及執行時間，試行結果不作 Prophet／NeuralProphet 已使用或預測準確度的宣稱。

## Observation window

Recommended: **last 12 weeks** of stable usage after rollout. Record start/end dates in all public summaries.

## Next measurement cycle / 下一輪量測待辦

The following work is planned for a future observation window. It is **not** a published result and must not be described as a current outcome in a CV or public summary.

以下項目為下一個觀測期的待辦事項，**不是**已發布結果；完成量測前，不應在 CV 或公開摘要中表述為目前成效。

### Privacy-preserving usage events / 兼顧隱私的使用事件

- [ ] Record server-side `lookup_completed` events with a server timestamp, lookup method (`barcode`, `keyword`, or `product_code`), and outcome (`found`, `not_found`, or `error`).
- [ ] Record `arrival_created`, `issue_created`, `replenishment_viewed`, and `replenishment_exported` or `rule_updated` events where applicable.
- [ ] Use a stable pseudonymous operator identifier and role (`staff` or `admin`); do not include raw search text, barcodes, customer data, or staff names in analytics events.
- [ ] Define a retention period and access policy before collection (for example, retain de-identified events for 90–180 days).
- [ ] Collect at least 4 weeks of stable data; use 12 weeks when reporting adoption trends publicly.

- [ ] 以伺服器時間記錄 `lookup_completed` 事件，包含查詢方式（`barcode`、`keyword`、`product_code`）及結果（`found`、`not_found`、`error`）。
- [ ] 視功能情況記錄 `arrival_created`、`issue_created`、`replenishment_viewed`、`replenishment_exported` 或 `rule_updated` 事件。
- [ ] 使用穩定的去識別化操作員 ID 與角色（`staff` 或 `admin`）；分析事件不記錄原始搜尋字串、條碼、客戶資料或員工姓名。
- [ ] 收集前先定義保存期限與存取政策（例如去識別化事件保留 90–180 天）。
- [ ] 至少收集 4 週穩定資料；若公開報告採用趨勢，使用 12 週觀測期。

Once implemented, report lookup count, successful lookup rate, weekly/monthly active operators, and workflow-event counts with the observation window and denominator. A lookup success rate does **not** measure camera/scanner detection success; that requires separate scanner instrumentation.

完成後，可連同觀測期間及分母，報告查詢次數、成功查詢率、每週／每月活躍操作員與工作流程事件數。查詢成功率不等於相機／掃碼器偵測成功率；後者需另行加入掃碼器事件。

### Anonymous staff and manager feedback / 匿名店員與管理者回饋

- [ ] Prepare separate staff and manager questionnaires before the observation window closes.
- [ ] Ask about self-estimated workflow minutes, usability, traceability, and examples of manual overrides; do not present self-reported minutes as system-record timing.
- [ ] Record invitation count, response count `n`, collection dates, and whether the same respondents completed before/after questions.
- [ ] Publish only anonymized aggregates and clearly separate survey findings from system-record KPIs.

- [ ] 在觀測期結束前準備分別給店員及管理者的問卷。
- [ ] 詢問自行估計的流程分鐘數、易用性、可追溯性及人工覆核案例；自行回報的時間不得當作系統紀錄的耗時。
- [ ] 記錄受邀人數、回覆數 `n`、收集日期，以及前後問卷是否由相同受訪者完成。
- [ ] 僅公開匿名彙總結果，並清楚區分問卷發現與系統紀錄 KPI。

## Related repositories

- [`POS-Data-Analytics-Pipeline`](https://github.com/chengchakkwong/POS-Data-Analytics-Pipeline) — analytics + Firestore sync
- [`retail-operations-web-app`](https://github.com/chengchakkwong/retail-operations-web-app) — portfolio-safe downstream app

Evaluation scripts and survey templates live in the **private operational app** repository (not linked publicly).

## Published results

Approved aggregates from a local, de-identified Firestore snapshot:

| Metric | Result | Source |
| --- | --- | --- |
| Observation window | 2026-04-19 to 2026-07-13 | Export manifest |
| Replenishment SKUs | 7,289 | System records |
| ABC/XYZ coverage | 97.5% (7,109 / 7,289) | System records |
| Target-stock coverage | 97.4% (7,102 / 7,289) | System records |
| Replenishment candidates | 2,149 (29.5%) | `Target_Stock` vs usable on-hand stock |
| Arrival records | 392 from 4 de-identified operator labels | System records |
| Inbound SKU match rate | Not published | One-to-one reconciliation validation pending |
| Issue resolution rate | Not published | Test-record exclusion and workflow validation pending |
| Survey results | Not published | Survey collection pending |

Coverage metrics describe data readiness, not model accuracy. Replenishment candidates are a review list, not automatic purchase orders.

## Reporting language

- Safe: "During a single-store pilot window, Firestore records showed …"
- Safe: "In anonymous internal feedback (n=X), respondents reported …"
- Avoid: "Proven X% efficiency gain" without method, sample size, and data source separation.
