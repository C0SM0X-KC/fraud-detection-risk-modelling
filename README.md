# Financial Transaction Fraud Detection and Risk Modelling

A real-time fraud detection and risk-scoring system for card-not-present payments,
built and evaluated on the [IEEE-CIS Fraud Detection][ieee] benchmark (590,540
transactions, 3.5% fraud) under a strictly chronological protocol.

Given one raw transaction, the system returns a calibrated 0-100 risk score, a
verdict, the action to take and the reasons behind it in plain English, in about
12 ms.

**Team:** Kabir Chhabra (235805300) · Mayank (235805360) · Nandurkar Shreyas Vivek (235805446)
**Guide:** Dr. Usha M · Department of CSE, Manipal Institute of Technology, Bengaluru

[ieee]: https://www.kaggle.com/competitions/ieee-fraud-detection

---

## Results

Held-out test span: days 141-183, 118,108 transactions, 3.44% fraud. Every modelling
choice was made on the validation span; the test span is used only for reporting.

| Metric | Value |
|---|---|
| AUC-PR, selected score (XGBoost + LightGBM blend) | **0.861** (no-skill floor 0.034, 25x) |
| AUC-PR, calibrated decision layer (served) | 0.848 |
| Expected calibration error | 0.00044 |
| High-risk tier (score 70-100) | 70.3% of all fraud in 2.6% of volume, at 93.1% precision |
| Modelled loss reduction | 68.2% |
| AUC-PR with a 30-day label-confirmation delay | 0.649 (75.5% of performance retained) |
| Latency, raw transaction to decision | 12 ms p50 / 19 ms p95 |
| Latency, including plain-English reasons | 55 ms p50 / 66 ms p95 |

Further findings:

- **Faithful explanations.** Removing an alert's top three stated reasons lowers its
  score by 92 points on average, against 1 point for three random features.
- **Adversarial robustness.** A PGD transfer attack on the 73 attacker-controllable
  features reduces fraud recall by at most 8.3% (relative), against about 20% reported
  in the literature.
- **Transfer.** On the Bank Account Fraud dataset the same protocol reaches AUC-PR 0.187
  (12.7x the floor) and 55% recall at 5% FPR. A fairness check shows applicants aged 50+
  are falsely flagged about three times as often as younger applicants.
- **Evaluated and not adopted** (decided on validation): Optuna tuning, account-profile
  features, CatBoost as a blend member, the stacking meta-learner, and the LSTM and
  graph-network tiers.

## How it works

The system is a five-stage pipeline with a feedback loop:

| Stage | Module | Output |
|---|---|---|
| 1. Ingestion | `frds/data/` | merged, time-ordered transactions |
| 2. Feature engineering | `frds/features/` | 346 causal behavioural, velocity and graph features |
| 3. Model layer | `frds/models/` | base model scores |
| 4. Fusion and risk | `frds/fusion/` | calibrated probability, 0-100 score, risk tier |
| 5. Decision and governance | `frds/evaluation/` | reason codes, drift, latency, reports |

### Model stack

| Tier | Models | Config key |
|---|---|---|
| Baseline | Logistic regression, random forest | `logistic_regression`, `random_forest` |
| Tabular | XGBoost, LightGBM, CatBoost | `xgboost`, `lightgbm`, `catboost` |
| Sequential | LSTM (alternative: temporal convolutional network) | `lstm`, `tcn` |
| Relational | Temporal heterogeneous graph neural network with attention | `hetero_gnn` |
| Fusion | Stacking meta-learner, equal-weight log-odds blends, isotonic calibration | always on |

`models.enabled` in `config/default.yaml` selects the tiers to run. The decision layer
chooses between the stack, each base model and the blends on validation.

### Risk tiers

| Score | Verdict | Action |
|---|---|---|
| 0-29 | Legitimate | Approve |
| 30-69 | Suspicious | Step-up authentication |
| 70-100 | Likely fraud | Hold for manual review |

### Design principles

- **Causal features.** Every aggregate (velocity counters, rolling statistics, graph
  degrees, target encodings) uses only transactions earlier than the one being scored.
  `tests/test_causal.py` alters future rows and asserts that no earlier feature changes.
- **Chronological evaluation.** Train, validation and test spans follow each other in
  time, and the split is made on the timestamp so that no instant straddles a boundary.
- **Calibration on unseen rows.** After the final refit, the calibrator is fitted on a
  holdout the refitted model never trained on, so it describes the model it is applied to.
- **Temporal graph model.** Each entity node carries a memory that is read before a
  transaction is scored and updated after it, so no transaction informs its own score.
- **Training-serving parity.** The online feature store reproduces the offline features
  exactly (`tests/test_online_features.py`), and the online score equals the offline score
  (`tests/test_product.py`).
- **Configuration-driven.** Every threshold, window and hyper-parameter lives in
  `config/default.yaml`.

## Getting started

### Docker (recommended)

```bash
docker compose build
docker compose run --rm frds python -m frds all      # run the full pipeline
docker compose up api dashboard                      # start the product
```

The analyst console is served at http://localhost:8501 and the API, with interactive
documentation, at http://localhost:8000/docs. The repository is mounted into the
container, so outputs are written to `data/`, `artifacts/` and `reports/` in the working
tree.

Docker is also the supported route on Windows machines where Smart App Control blocks
native Python packages. If PyPI is slow on your network, build through a mirror; every
downloaded file is verified against its official checksum before installation:

```bash
docker compose build --build-arg PIP_INDEX_URL=https://mirrors.cloud.tencent.com/pypi/simple
```

### Native installation

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt && pip install -e .
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-extra.txt                  # CatBoost, MLflow, Kafka client, .env loading
```

### Data

```bash
python -m frds acquire                # download IEEE-CIS through the Kaggle API
```

This requires an authenticated Kaggle CLI and acceptance of the
[competition rules][rules]. Alternatively, place `train_transaction.csv` and
`train_identity.csv` in `data/raw/`. API tokens are read from a `.env` file in the
repository root, which is excluded from version control.

To run the pipeline without a Kaggle account, generate a schema-compatible synthetic
fixture:

```bash
python -m frds synth --rows 60000
```

The fixture exists for testing only, and its metrics are not results. It is marked by a
`SYNTHETIC_FIXTURE` file in `data/raw/`; remove it and the CSVs before loading the real
data.

[rules]: https://www.kaggle.com/competitions/ieee-fraud-detection/rules

## Usage

### Pipeline

```bash
python -m frds all                    # every stage in order
python -m frds prepare                # or one stage at a time:
python -m frds features               #   each stage reads what the previous one wrote
python -m frds train
python -m frds fuse
python -m frds evaluate
```

Any configuration key can be overridden on the command line:

```bash
python -m frds all --set data.sample_rows=100000 --set "models.enabled=[lightgbm]"
```

### Additional stages

```bash
python -m frds tune                   # Optuna search for XGBoost and LightGBM
python -m frds robustness             # PGD transfer attack on attacker-controllable features
python -m frds baf                    # the same protocol on Bank Account Fraud, with a fairness check
python -m frds stream                 # replay through the streaming transport (in-memory)
docker compose --profile stream up    # the same through a Kafka broker
```

Set `tuning.use_tuned: true` to train with the tuned parameters, and
`tracking.mlflow: true` to log every stage to MLflow.

### Scoring transactions

**Console** (http://localhost:8501):

- *Check a transaction*: edit a real transaction and score it. Re-send the same card to
  see velocity raise the score, or confirm an outcome to update the card's fraud history.
- *Live ledger*: the test span replayed in arrival order, with the alert queue, the
  reasons behind each alert and live score-drift monitoring.
- *Model and monitoring*: model comparison, calibration, tiers and cost, explanation
  quality, drift, latency, robustness and the transfer results.

**API** (http://localhost:8000/docs):

```bash
curl -X POST localhost:8000/transactions/score -H "Content-Type: application/json" \
  -d '{"transaction": {"TransactionAmt": 950.0, "ProductCD": "W", "card1": 13926,
       "card4": "visa", "card6": "debit", "addr1": 315.0, "P_emaildomain": "gmail.com"}}'
```

| Endpoint | Purpose |
|---|---|
| `POST /transactions/score` | score one raw transaction |
| `POST /transactions/score-batch` | score a batch in arrival order |
| `POST /transactions/{id}/outcome` | record a confirmed outcome (feedback loop) |
| `GET /transactions/example` | a real transaction to start from |
| `GET /health` | service status |

Responses include `risk_score`, `probability`, `tier`, `verdict`, `action`,
`is_fraud_prediction`, `reasons` and `timings_ms`. An abridged run of
`scripts/smoke_api.py` against the running service:

```text
A known fraud transaction, submitted as new:
  risk 71  Likely fraud  -> Hold for manual review   p=0.7143
    + Smoothed fraud rate of this card's earlier transactions is 28.5%   (Entity fraud history)
    + Masked provider feature C14 is 0                                     (Vesta counts)
    - Device model is Blade A510 Build/MRA58K                              (Categorical profile)
After the analyst confirms it as fraud, the same card again:
  risk 90  Likely fraud  -> Hold for manual review
A legitimate transaction:
  risk  2  Legitimate    -> Approve                  21 ms
```

**Command line:**

```bash
python -m frds score '{"TransactionAmt": 950, "card1": 13926}'
```

### GPU training on Kaggle

The sequential and graph tiers are slow on a CPU and can be trained on a Kaggle GPU:

```bash
python scripts/kaggle_job.py push     # upload the code to a private dataset and start the job
python scripts/kaggle_job.py status
python scripts/kaggle_job.py pull     # download the score checkpoints into artifacts/
python -m frds train                  # reuses them and fits the tree tiers locally
```

Only the code is uploaded; the datasets are attached on Kaggle's side.

## Outputs

`python -m frds all` writes everything to `reports/`:

| Folder | Contents |
|---|---|
| `reports/figures/` | precision-recall and ROC curves, reliability, score distribution, drift, SHAP, faithfulness, robustness, label-delay ablation |
| `reports/tables/` | model comparison, risk tiers, drift, reason codes, SHAP importance, latency, robustness, transfer (CSV) |
| `reports/metrics/` | split, model, fusion and evaluation summaries (JSON); `evaluation_summary.json` holds every result |

The project report is generated from these files with `node scripts/build_report.js`.

## Testing

```bash
python -m pytest -q
```

The suite has 67 tests covering the causal feature primitives and future-leakage
checks, splitting, drift, metrics, calibration, the risk and cost model, online/offline
feature parity, the product path, fusion, tuning, robustness, transfer, streaming, and
an end-to-end run of every stage. `scripts/smoke_api.py` exercises a running API, and
`scripts/check_online_parity.py` replays real test transactions through the online store.

## Project structure

```
config/default.yaml     thresholds, windows and hyper-parameters
src/frds/
  data/                 acquisition, synthetic fixture, ingestion, chronological split
  features/             causal primitives, offline feature builder, online feature store
  models/               model tiers, registry, imbalance handling, training, tuning
  fusion/               stacking, blends, calibration, risk score and tiers
  evaluation/           metrics, drift, latency, SHAP, reasons, robustness
  product.py            raw transaction -> score, verdict, action, reasons
  pipeline.py, cli.py   stage orchestration and command line
  transfer.py           Bank Account Fraud transfer check
  stream.py             Kafka / in-memory streaming transport
serving/                FastAPI service and Streamlit analyst console
scripts/                report builder, experiments, Kaggle job, smoke tests
kaggle/                 GPU job entry point
tests/                  unit, parity, product and end-to-end tests
```

## Data licence

The IEEE-CIS data is subject to the Kaggle competition rules and is not redistributed in
this repository. Datasets, trained models and generated reports are excluded from
version control and are recreated by the pipeline.
