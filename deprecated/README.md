# Deprecated

Superseded scripts and packaging leftovers. **Do not treat these as supported entry points.**

Prefer:

```powershell
python -m pos_pipeline.cli daily
python -m pos_pipeline.cli analytics
python -m pos_pipeline.cli min-multiple
```

| File | Status |
|------|--------|
| `POS_Sync_Tool.spec` | PyInstaller remnant for removed `POS_Sync_Tool.py` |
| `upload_final_inventory_plan_to_firebase.py` | Replaced by v3 analytics Target_Stock upload. **Not runnable** (depends on removed `firebase_service.py`). |
| `supplier_upload.py` | Legacy Firestore `supplier` upload. **Not runnable** (depends on removed `pos_service` / `firebase_service` / `db_utils` imports). Re-implement under `pos_pipeline` if needed. |

Safe to delete the two `.py` files above when you no longer need them as historical reference.
