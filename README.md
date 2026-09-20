# mlops-mini — Airflow Pipeline + MLflow Model Registry + FastAPI Serving

An end-to-end MLOps stack on the Wisconsin Diagnostic Breast Cancer (WDBC) data:
an **Airflow** pipeline ingests and validates an extract, trains a model and registers it in
the **MLflow Model Registry**, and a **FastAPI** service serves whatever version is selected.

Everything runs from one `docker compose up -d --build`. No configuration is required.

## Architecture

```mermaid
flowchart LR
    RAW[("data/raw/wdbc.csv")] --> AF
    subgraph AF["Airflow 3.3.2 · LocalExecutor · :18080"]
        direction LR
        I[ingest] --> V{validate<br/>fail if &gt;5% bad} --> S[split<br/>SHA-256 of sample_id]
        S --> SC[scale<br/>z-score, train only] --> T[train_and_register] --> R[report]
    end
    T -->|log run, register version, move @champion| M[("MLflow 3.8.1<br/>Tracking + Registry<br/>:15010")]
    API["FastAPI :18011<br/>/health · /predict"] -->|models:/…@champion<br/>or models:/…/N| M
    PG[("Postgres 16")] --- AF
```

Every task writes into `data/staging/<logical date>/`, so a finished run can be opened up
afterwards and re-running a date only overwrites that date.

## Services

| Service | Port | Purpose |
|---|---|---|
| `airflow-apiserver` | `18080` | Airflow UI and REST API |
| `airflow-scheduler` | – | runs task instances (LocalExecutor) |
| `airflow-dag-processor` | – | parses DAG files (a separate service in Airflow 3) |
| `airflow-triggerer` | – | required for Airflow to report itself healthy |
| `postgres` | – | Airflow metadata database |
| `mlflow` | `15010` | tracking server, artifact store and Model Registry |
| `api` | `18011` | FastAPI serving the selected model version |

Ports come from `.env` (`AIRFLOW_PORT`, `MLFLOW_PORT`, `API_PORT`) if you need to move them.

## The DAG: `wdbc_pipeline`

Six tasks, in order:

| # | Task | What it does |
|---|---|---|
| 1 | `ingest` | Reads `data/raw/wdbc.csv` and freezes it as `staging/<ds>/raw.parquet`. |
| 2 | `validate` | Flags nulls, negative measurements, unknown labels, duplicate ids and extreme outliers. Clean and rejected rows are written separately with a `validation_report.json`. **Fails the run with `AirflowFailException` when more than 5% of rows are bad**, which skips the retries — a broken extract will not repair itself. |
| 3 | `split` | 80/20 by bucketing a SHA-256 hash of `sample_id`. No random seed, so the split is identical on every machine and adding rows later cannot reshuffle the existing ones. |
| 4 | `scale` | Z-score normalisation with statistics taken from the **training half only**, persisted to `scaler.json`. Fitting on everything would leak the test rows into the model. |
| 5 | `train_and_register` | Trains a `RandomForestClassifier`, logs params, `test_roc_auc` / `test_accuracy` / `test_f1` and the model to MLflow, and registers a new version. The `champion` alias moves only when the new version is at least as good as the incumbent. |
| 6 | `report` | Writes `summary.json` for the run and appends one line to `staging/history.jsonl`. Re-running a date replaces that date's line instead of adding a second. |

---

## Step 1 — start everything

```bash
docker compose up -d --build
docker compose ps
```

Wait until `mlflow`, `api` and `airflow-apiserver` report **healthy** (30–60 seconds).

| What | Where | Credentials |
|---|---|---|
| Airflow UI | <http://localhost:18080> | `admin` / `admin` |
| MLflow UI | <http://localhost:15010> | none |
| API docs | <http://localhost:18011/docs> | none |

## Step 2 — run the pipeline

```bash
docker compose exec airflow-scheduler airflow dags test wdbc_pipeline 2026-08-25
```

All six tasks run and `breast-cancer-classifier` version 1 appears in the Model Registry at
<http://localhost:15010>. The DAG is also scheduled `@daily`, so it runs on its own; this
command is the explicit, foreground version for one logical date.

## Step 3 — serve predictions

```bash
curl -s http://localhost:18011/health
```
```json
{"status":"ok","model_loaded":true,"model_uri":"models:/breast-cancer-classifier@champion"}
```

```bash
docker compose exec airflow-scheduler python /opt/airflow/scripts/sample_request.py > sample_request.json
curl -s -X POST http://localhost:18011/predict \
  -H 'content-type: application/json' -d @sample_request.json
```
```json
{"prediction":"malignant","probability_benign":0.03,"served_by":"models:/breast-cancer-classifier@champion"}
```

`sample_request.py --label B` picks a benign row instead, which comes back as
`{"prediction":"benign","probability_benign":0.975,...}`.

## Step 4 — watch the validation gate refuse a bad extract

```bash
docker compose exec airflow-scheduler python /opt/airflow/scripts/corrupt_extract.py
docker compose exec airflow-scheduler airflow dags test wdbc_pipeline 2026-08-27
```

The `validate` task stops the run:

```
validation: 68 rejected of 569 (11.95% of rows)
airflow.sdk.exceptions.AirflowFailException: 12.0% of rows rejected, limit is 5%
```

Nothing downstream runs, and the registry is untouched. Put the extract back:

```bash
docker compose exec airflow-scheduler python /opt/airflow/scripts/corrupt_extract.py --repair
```

## Step 5 — switch or roll back the served version

Register a second version, then pin the API to an exact one:

```bash
docker compose exec airflow-scheduler airflow dags test wdbc_pipeline 2026-08-28
```

Set `MODEL_VERSION` in `.env` and recreate the API container — no code change, no rebuild:

```env
MODEL_VERSION=1
```
```bash
docker compose up -d api
curl -s http://localhost:18011/health
```
```json
{"status":"ok","model_loaded":true,"model_uri":"models:/breast-cancer-classifier/1"}
```

Clear `MODEL_VERSION` again and the API follows the `champion` alias, which the pipeline
moves on every successful promotion.

---

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | status, whether a model is loaded, and the URI it came from |
| `POST /predict` | `{"features": [30 floats]}` → `{"prediction", "probability_benign", "served_by"}` |
| `GET /model` | name, alias, version, run id and metrics of the loaded model |
| `POST /reload` | re-resolve the model URI immediately |
| `GET /readyz` | 503 until a model is loaded |
| `GET /metrics` | Prometheus metrics |
| `GET /docs` | OpenAPI UI |

`/health` answers **200 even when degraded**: an empty registry is an expected phase before
the first run, not a dead process. The API also retries the registry by itself (throttled to
once every 15 seconds), so it starts serving as soon as a model is promoted without anyone
calling `/reload`.

## Design notes worth reading

**The scaler travels with the registered model.** The forest is fitted on scaled data, so
serving it raw measurements would feed it numbers from a different range entirely — a benign
row and a malignant row came back with the *same* probability before this was fixed. What
gets registered is a `Pipeline(StandardScaler, RandomForest)` carrying the training
statistics, so the API can post raw features and stay correct. Two regression tests cover it.

**Validation quarantines rather than deletes.** Rejected rows go to `rejected.parquet` so a
bad extract can be inspected, and only the *proportion* of bad rows fails the run.

**The split has no random seed.** Hashing `sample_id` means the same row is always on the
same side, on any machine, however many rows are added later.

## Development

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r api/requirements.txt pytest ruff httpx

make test    # pytest  — 60 tests, no network, no MLflow server needed
make lint    # ruff
```

`scripts/make_extract.py` regenerates `data/raw/wdbc.csv` from scikit-learn's copy of the
same dataset, so the committed extract is reproducible rather than hand-maintained.

## Layout

```
dags/wdbc_pipeline.py       the six-task DAG
src/mlops/                  shared library, used by both the DAG and the API
  config.py                 settings from environment variables
  extract.py                reading the extract, per-run staging folders
  validate.py               quality checks and the 5% gate
  split.py                  deterministic hash split
  scale.py                  z-score fitted on train only
  train.py                  estimator, evaluation, MLflow logging, serving pipeline
  registry.py               register, promote by alias, load for serving
  report.py                 summary.json and history.jsonl
  drift.py                  PSI, reported alongside validation
api/main.py                 FastAPI serving layer
scripts/                    make_extract, corrupt_extract, sample_request
tests/                      60 unit tests
.github/workflows/ci.yml    lint + tests, image builds, compose validation
```

## Configuration that is easy to get wrong

These four settings in `docker-compose.yml` are not optional, and each fails in a way that
does not name itself:

| Setting | Symptom when missing |
|---|---|
| `AIRFLOW__CORE__EXECUTION_API_SERVER_URL` | every task fails with `httpx.ConnectError: Connection refused` — Airflow 3 runs tasks through the Task SDK, which calls back into the API server, and the default points at `localhost` |
| `AIRFLOW__API_AUTH__JWT_SECRET` | tasks fail with `ServerResponseError: Invalid auth token` — every Airflow service must sign with the same secret |
| `mlflow server --allowed-hosts` | `403 Invalid Host header - possible DNS rebinding attack detected`. It must list the **published** port as well as the in-network one, or the UI is unreachable from the host |
| `airflow-triggerer` | Airflow's home page reports itself unhealthy even though nothing uses deferrable operators |

## Notes

* `AIRFLOW__CORE__SIMPLE_AUTH_MANAGER_*` is Airflow 3's default auth manager and is meant for
  development. Do not expose this compose file to a network you do not control.
* MLflow uses SQLite and a local artifact volume — fine for one machine. Swap the backend
  store for Postgres and the artifact store for S3/MinIO before sharing it.
* If `airflow-init` stops with *"/opt/airflow/dags is empty inside the container"*, the repo
  sits outside a path your Docker VM shares (common with Colima or Lima). Move it under your
  home directory.
