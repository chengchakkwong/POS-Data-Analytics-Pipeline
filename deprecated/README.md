# Deprecated

Superseded or transitional root scripts. Prefer `python -m pos_pipeline.cli …` for production.

| File | Status |
|------|--------|
| `POS_Sync_Tool.spec` | PyInstaller remnant for the removed `POS_Sync_Tool.py` |
| `upload_final_inventory_plan_to_firebase.py` | Replaced by v3 analytics Target_Stock upload |
| `supplier_upload.py` | Still runnable if you need Firestore `supplier`; not yet in v3. Run from repo root: `python deprecated/supplier_upload.py` |
| `replenishment_forecasting*.py` | Older forecast prototypes (if present) |

Imports still assume the repo root is on `PYTHONPATH` (run from project root).
