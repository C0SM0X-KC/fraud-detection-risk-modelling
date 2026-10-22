# CLAUDE.md

Fraud detection and risk modelling on IEEE-CIS (Minor project, MIT Bengaluru).
Current state, results and every decision with its reason: **docs/PROJECT_STATUS.md** -
read it first in a new session.

## Running things

Native Python does not work on this machine (Windows Smart App Control blocks the
packages' DLLs). Everything runs in Docker; the repo is mounted at /app.

```bash
docker compose run --rm frds python -m frds <stage>        # prepare|features|train|fuse|evaluate|all
docker compose run --rm frds python -m frds tune|robustness|baf|stream
docker compose run --rm frds python -m pytest -q            # ~10 min alone; much slower under CPU contention
docker compose up api dashboard                             # product: :8000/docs and :8501
```

- Long jobs: start detached (`docker compose run -d ...`) and watch them with
  `bash scripts/watch_all.sh 120 600 3600` under the Monitor tool (re-arm every 30 min).
- Run one CPU-heavy job at a time: concurrent LightGBM/XGBoost containers
  oversubscribe the 16 cores and slow ~50x.
- Rebuild the image through the verified mirror (PyPI is very slow on this network):
  `docker compose build --build-arg PIP_INDEX_URL=https://mirrors.cloud.tencent.com/pypi/simple api`
- GPU tiers (LSTM, GNN, CatBoost) run on Kaggle: `python scripts/kaggle_job.py push|status|pull`
  with the host `.venv` (Kaggle CLI). Kaggle secrets reach only editor-started runs.
- Report: `node scripts/build_report.js` -> reports/Phase-3_Fraud_Detection_Report.docx.

## Rules

- Adopt any modelling change on **validation** only; the test span is for reporting.
- Every train-time feature must have an online equivalent: `features/online.py` must
  reproduce `features/build.py` exactly (tests/test_online_features.py).
- `.env` holds tokens and is gitignored; never print values.
- Do not commit or push without the user's explicit go-ahead.
- The user submits an individual report: do not change authorship, the title page or
  report section 5.3.
- The user has no ML background: explain results in plain language.
