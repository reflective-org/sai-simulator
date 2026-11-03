# S3 verification testing (beginner guide)

> Looking for how the reference objects in S3 are produced, rather than how to run against
> them? See [REGENERATING_GOLDENS.md](REGENERATING_GOLDENS.md). Nothing in this repo creates
> them — they come from `sai-simulator-api`'s pregeneration run.

This guide shows exactly how to run verification tests with **Celery parallel workers**.
It includes all required setup steps, including **how to install/start Redis**.

What this does:
- runs `get_outputs`
- compares local outputs with reference GeoTIFF files from S3
- writes a JSON report with pass/fail details

---

## Quick start (copy/paste)

If you want the shortest path, use this checklist from the repo root:

1) Install Python dependencies:

```bash
pip install -r requirements.txt
```

2) Start Redis (choose one):

**macOS + Homebrew**
```bash
brew install redis
brew services start redis
```

**Docker**
```bash
docker run -d --name redis-verification -p 6379:6379 redis:7-alpine
```

3) Confirm Redis is running:

```bash
redis-cli ping
```

Expected output:

```text
PONG
```

4) Set required environment variables:

```bash
export VALIDATION_S3_BUCKET=your-bucket-name
export VALIDATION_CELERY_BROKER_URL=redis://localhost:6379/0
export VALIDATION_CELERY_RESULT_BACKEND=redis://localhost:6379/0
```

5) Start a Celery worker (leave this terminal open):

```bash
export PYTHONPATH="$(pwd):${PYTHONPATH}"
celery -A tests.verification.celery_app worker --loglevel=INFO --concurrency=2
```

6) In a second terminal, run verification:

```bash
python -m tests.verification.run_verification_celery \
  --bucket your-bucket-name \
  --limit 20 \
  --output-json validation_celery_report.json
```

7) Open the report:

```bash
python -m json.tool validation_celery_report.json | less
```

---

## Full setup details

1. **Python environment**  
   From the repo root, install dependencies (includes `celery` and `boto3`):

   ```bash
   pip install -r requirements.txt
   ```

2. **Local simulator data**  
   You need the processed data, models, and cache directories the verification code expects (defaults below). If they are missing, follow the main project [README](../../README.md) (download, process, fit models, cache).

   Default paths (relative to repo root):

   | Role   | Default path        | Override env var        |
   |--------|---------------------|-------------------------|
   | Data   | `data/processed`    | `VALIDATION_DATA_DIR`   |
   | Models | `data/models`       | `VALIDATION_MODEL_DIR`  |
   | Cache  | `data/cache`        | `VALIDATION_CACHE_DIR`  |

3. **AWS credentials for S3**  
   Validation downloads reference objects from S3. Make sure your shell can access AWS.

   Quick check:

   ```bash
   aws sts get-caller-identity
   ```

   If that fails, configure credentials first (for example with `aws configure`).

4. **Redis** (for Celery broker and result backend)  
   Celery is configured by default to use Redis at `redis://localhost:6379/0`.

   **macOS (Homebrew):**

   ```bash
   brew install redis
   brew services start redis
   ```

   **Docker:**

   ```bash
   docker run -d --name redis-verification -p 6379:6379 redis:7-alpine
   ```

   Use another host/port if you prefer; set `VALIDATION_CELERY_BROKER_URL` and `VALIDATION_CELERY_RESULT_BACKEND` accordingly.

---

## Celery parallel verification — step by step

Run these from the **repository root** so imports like `tests.verification` resolve correctly.

### Step 1: Start Redis

Use one of these options:

**Option A: macOS + Homebrew**

```bash
brew install redis
brew services start redis
```

**Option B: Linux (system service example)**

```bash
sudo apt-get update
sudo apt-get install -y redis-server
sudo systemctl enable redis-server
sudo systemctl start redis-server
```

**Option C: Docker**

```bash
docker run -d --name redis-verification -p 6379:6379 redis:7-alpine
```

Verify Redis:

```bash
redis-cli ping
```

Expected: `PONG`

### Step 2: Set Celery broker/result URLs

Default broker and result backend are Redis on `localhost`:

```bash
export VALIDATION_CELERY_BROKER_URL=redis://localhost:6379/0
export VALIDATION_CELERY_RESULT_BACKEND=redis://localhost:6379/0
```

For a remote Redis:

```bash
export VALIDATION_CELERY_BROKER_URL=redis://user:pass@host:6379/0
export VALIDATION_CELERY_RESULT_BACKEND=redis://user:pass@host:6379/0
```

Optional task time limits (seconds):

```bash
export VALIDATION_CELERY_TASK_TIME_LIMIT=3600
export VALIDATION_CELERY_TASK_SOFT_TIME_LIMIT=3300
```

### Step 3: Start one or more Celery workers

Use **low concurrency** on a laptop to reduce OOM risk (each task runs `get_outputs` and heavy I/O).

```bash
celery -A tests.verification.celery_app worker --loglevel=INFO --concurrency=2
```

- Start with `--concurrency=1` or `2`.
- Increase slowly only if your machine stays stable.
- Keep this terminal running while tests are executing.

### Step 4: Configure S3 and verification parameters

Minimum:

```bash
export VALIDATION_S3_BUCKET=your-bucket-name
# Optional:
export VALIDATION_S3_PREFIX=ref/v1
export VALIDATION_S3_FILETYPE=geotiffs   # or plot_data
```

### Step 5: Dispatch verification jobs

The dispatcher enqueues one Celery task per case and waits for results, then writes a JSON report.

```bash
python tests/verification/run_verification_celery.py \
  --bucket your-bucket-name \
  --limit 20 \
  --output-json validation_celery_report.json
```

**Useful flags** (see also `--help`):

| Flag | Purpose |
|------|---------|
| `--bucket` | S3 bucket (required) |
| `--prefix` | Key prefix inside bucket |
| `--filetype` | `geotiffs` or `plot_data` |
| `--data-dir`, `--model-dir`, `--cache-dir` | Override local paths |
| `--decade` | e.g. `2091-2100` |
| `--ramp-up` | Ramp-up years (integer) |
| `--variables` | Comma-separated, e.g. `tas` or `tas,p-e` |
| `--limit` | Number of cases (with generated cases) |
| `--generate-count` | If set, overrides `--limit` for generation (same semantics as pytest helpers) |
| `--cases-json` | Path to a JSON array of case dicts instead of random generation |
| `--seed` | Integer seed, or `random` / `rand` for time-based seed |
| `--epsilon` | Relative-error denominator floor |
| `--poll-seconds` | Interval while waiting on task results |
| `--output-json` | Report path (default `validation_celery_report.json`) |

Example with explicit paths and variables:

```bash
python -m tests.verification.run_verification_celery \
  --bucket your-bucket-name \
  --prefix ref/v1 \
  --data-dir data/processed \
  --model-dir data/models \
  --cache-dir data/cache \
  --variables tas \
  --limit 10 \
  --decade 2091-2100 \
  --output-json ./reports/validation_celery_report.json
```

Exit code behavior:
- `0` = all cases passed
- non-zero = one or more cases failed

---

## Pytest (sequential)

For debugging or CI without Redis/Celery:

```bash
export VALIDATION_S3_BUCKET=your-bucket-name
pytest tests/verification/test_verification_s3.py -m s3_verification -v
```

Batch (all cases in one process):

```bash
pytest tests/verification/test_verification_s3.py -m s3_verification_batch -v
```

Environment variables match the main [conftest](conftest.py) docstring (`VALIDATION_LIMIT`, `VALIDATION_SEED`, etc.).

### Pytest driving Celery tasks (failures fail the test)

Use [test_verification_s3_celery.py](test_verification_s3_celery.py). Each case calls
`validate_case_task.delay(...).get(propagate=True)`, so any task exception (including
failed verification) makes pytest fail.

**No worker (eager mode, good for CI):** tasks run in the pytest process.

```bash
export VALIDATION_S3_BUCKET=your-bucket-name
export VALIDATION_CELERY_EAGER=1
pytest tests/verification/test_verification_s3_celery.py -m s3_verification_celery -v
```

**With a real worker:** omit `VALIDATION_CELERY_EAGER`; start Redis + Celery worker first
(same as parallel runs above).

Batch variant (one test, all cases):

```bash
pytest tests/verification/test_verification_s3_celery.py -m s3_verification_celery_batch -v
```

### GitHub Actions on EC2 (pull requests)

PRs use workflow [`.github/workflows/s3-validation-self-hosted.yml`](../../.github/workflows/s3-validation-self-hosted.yml).
The verification **job runs on a self-hosted runner** you install on EC2 (not on GitHub-hosted runners), because the simulator data lives on that machine.

1. Register a **self-hosted runner** on the EC2 instance (GitHub → repo **Settings** → **Actions** → **Runners**).
2. Give the runner the labels `self-hosted` and `sai-simulator-validation` (or edit `runs-on` in the workflow to match your labels).
3. Add repository **secret** `VALIDATION_S3_BUCKET`. Optional **variables**: `VALIDATION_S3_PREFIX`, `VALIDATION_DATA_DIR`, etc. (see workflow header comments).
4. Use an **IAM instance profile** on EC2 (or other AWS credentials) so `boto3` can read the reference objects from S3.

The workflow sets `VALIDATION_CELERY_EAGER=1` so you do not need Redis or a Celery worker on the EC2 box for CI.

Fork PRs are skipped (secrets are unavailable); only PRs from the same repository run this job.

---

## Troubleshooting

| Issue | What to check |
|-------|----------------|
| `Connection refused` to Redis | Redis is not running or wrong URL; run `redis-cli ping` and re-check `VALIDATION_CELERY_BROKER_URL` |
| Workers idle, tasks never run | Worker started with same `tests.verification.celery_app` app; broker URL matches |
| `celery.exceptions.NotRegistered: 'verification.verify_case_task'` | Restart the Celery worker after code/dependency changes. Then verify task registration: `python -c "from tests.verification.celery_app import celery_app; print('verification.verify_case_task' in celery_app.tasks)"` |
| `ModuleNotFoundError: No module named 'backend'` | Start worker from repo root and set `PYTHONPATH`: `export PYTHONPATH=\"$(pwd):${PYTHONPATH}\"` before `celery -A tests.verification.celery_app worker ...` |
| S3 errors | Bucket name, prefix, IAM/credentials, object keys match reference layout |
| OOM on workers | Lower `--concurrency`; run fewer cases; ensure only one heavy worker per machine |
| Import errors | Run commands from **repo root** and prefer module mode: `python -m tests.verification.run_verification_celery ...` |
| `cannot import name 'CredentialProvider' from 'redis'` | Your **redis-py** package is too old for this Celery version. Upgrade: `pip install --upgrade "redis>=5.0.0"` then restart the worker. Confirm with: `python -c "from redis import CredentialProvider; print('ok')"` |
| `aws: command not found` | Install AWS CLI first, then run `aws configure` |

---

## Related files

| File | Role |
|------|------|
| [celery_app.py](celery_app.py) | Celery app, broker/backend env vars |
| [celery_tasks.py](celery_tasks.py) | Task that runs `validate_case` per case |
| [run_verification_celery.py](run_verification_celery.py) | CLI dispatcher and JSON report |
| [validate_outputs.py](validate_outputs.py) | `get_outputs`, S3 download, GeoTIFF comparison |
| [conftest.py](conftest.py) | Pytest fixtures and env documentation |
| [REGENERATING_GOLDENS.md](REGENERATING_GOLDENS.md) | Where the S3 reference objects come from and how to rebuild them |
