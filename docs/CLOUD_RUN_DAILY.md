# Cloud Run daily job

v3 `daily` job: POS SQL → stock CSV → Firestore `products` / `replenishment`.
Incremental writes use content hashes stored in Firestore `sync_state`.
Sales detail is not uploaded.

`inbound_movements` and App-facing `lastSyncedAt` are not in this job yet.

## Local run

```powershell
python -m pos_pipeline.cli daily
```

Needs `.env` (SQL) and `serviceAccountKey.json` in the project root.
If `FIREBASE_KEY_PATH` is set, that file path is used instead.

POS SQL is expected to be available about 09:00–20:00. If the database is down, the job prints an error and exits 1; it should not hang (login timeout is 5 seconds).

## Image

- `Dockerfile.daily` — Python 3.11, ODBC 18, `pos_pipeline`
- `.dockerignore` — excludes `.env`, keys, `data/`, docs, investigations
- `cloudbuild.yaml` — Cloud Build uses `-f Dockerfile.daily`

```powershell
gcloud builds submit --config cloudbuild.yaml .
```

Image tag pattern: `us-central1-docker.pkg.dev/<PROJECT_ID>/pos-pipeline/daily:v0.1`

## Secrets

Do not bake credentials into the image. Store at least:

- `db-server`, `db-database`, `db-uid`, `db-pwd`
- `firebase-sa-key` (service account JSON), mounted as `/secrets/serviceAccountKey.json`

Job env should include `FIREBASE_KEY_PATH=/secrets/serviceAccountKey.json` and `DB_DRIVER={ODBC Driver 18 for SQL Server}`.

## Schedule

Cloud Scheduler job (example name `pos-pipeline-daily-schedule`):

- Cron: `0 9-19/2 * * *`
- Time zone: `Asia/Hong_Kong`
- Effect: 09:00, 11:00, 13:00, 15:00, 17:00, 19:00

Logs: Cloud Logging (stdout from `print`). Do not write per-line logs to Firestore.
Operator-facing “last synced at” is not implemented yet.

## Manual execute

```powershell
gcloud run jobs execute pos-pipeline-daily --region us-central1
```
