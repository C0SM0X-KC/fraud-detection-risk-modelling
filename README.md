# Financial Transaction Fraud Detection and Risk Modelling

Implementation of the pipeline specified in the Phase-1/Phase-2 report: a
five-stage fraud detection and risk-scoring system evaluated under a strictly
chronological protocol on the IEEE-CIS Fraud Detection benchmark.

Kabir Chhabra (235805300) · Mayank (235805360) · Nandurkar Shreyas Vivek (235805446)
CSE, MIT Bengaluru · Guide: Dr. Usha M

---

## What this repository is

The earlier phases produced a specification: six research gaps, six objectives,
a five-stage workflow, a five-layer architecture, a model stack and an
evaluation protocol. This repository is that specification turned into running
code. Every stage of the workflow figure has a module, every objective has a
measurable output, and every threshold and window sits in `config/default.yaml`
rather than being hard-coded.

```
stage 01  ingestion              frds/data/          merged.parquet
stage 02  feature engineering    frds/features/      features.parquet
stage 03  imbalance-aware models frds/models/        base_scores.npz
stage 04  fusion + risk          frds/fusion/        meta_learner, calibrator
stage 05  decision + governance  frds/evaluation/    reports/
```

## Quick start

```bash
python -m venv .venv && .venv/Scripts/activate     # Windows
pip install -r requirements.txt && pip install -e .
```

PyTorch is a separate install because the wheel depends on your CUDA version:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

Then either get the real data, or run against the fixture:

```bash
python -m frds acquire          # IEEE-CIS via the Kaggle API, or instructions
python -m frds synth --rows 60000   # schema-compatible fixture, no account needed
python -m frds all
```

`python -m frds all` runs every stage and writes figures, tables and metrics to
`reports/`. Individual stages (`prepare`, `features`, `train`, `fuse`,
`evaluate`) can be re-run alone; each reads what the previous one wrote.

Any config key can be overridden inline:

```bash
python -m frds all --set data.sample_rows=100000 --set "models.enabled=[lightgbm]"
```

## Getting the IEEE-CIS data

`python -m frds acquire` downloads it if the Kaggle CLI is installed and
authenticated. That needs a one-time manual step the API cannot do for you:
sign in, open the [competition rules page][rules], and accept them. Then
`Account → Settings → API → Create New Token`, and move `kaggle.json` to
`%USERPROFILE%\.kaggle\kaggle.json`.

Failing that, download the archive by hand and drop `train_transaction.csv` and
`train_identity.csv` into `data/raw/`.

[rules]: https://www.kaggle.com/competitions/ieee-fraud-detection/rules

### About the synthetic fixture

`python -m frds synth` writes CSVs with the IEEE-CIS schema - same column names,
same dtypes, same missingness pattern, 3.5% prevalence, a 182-day span - and
embeds tabular, temporal, relational and drift signal so every tier of the model
stack has something to find. It exists so the pipeline can be run and tested
without a Kaggle account.

**Its numbers are not results.** It marks itself with a `SYNTHETIC_FIXTURE` file
in `data/raw/`. Delete that and the CSVs before loading the real benchmark, and
never quote fixture metrics in the report.

## How each objective is met

| # | Objective (report Chapter 3) | Where | Output |
|---|---|---|---|
| 1 | Reproducible pipeline, strictly chronological split | `data/split.py` | `reports/metrics/split_summary.json` |
| 2 | Velocity, rolling, recency, mismatch and graph features | `features/` | `data/processed/feature_spec.json` |
| 3 | Five model families under one protocol | `models/` | `reports/tables/model_comparison.csv` |
| 4 | Stacking, isotonic calibration, 0-100 score, cost-sensitive tiers | `fusion/` | `reports/tables/risk_tiers_*.csv` |
| 5 | SHAP reason codes on every high-risk alert | `evaluation/explain.py` | `reports/tables/alert_reason_codes.csv` |
| 6 | PSI drift and p50/p95 inference latency | `evaluation/drift.py`, `latency.py` | `reports/tables/psi_*.csv`, `latency.csv` |

### The model stack (report Table 4.2)

| Tier | Model | Key |
|---|---|---|
| Baseline | Logistic regression, random forest | `logistic_regression`, `random_forest` |
| Tabular | XGBoost, LightGBM | `xgboost`, `lightgbm` |
| Sequential | LSTM, temporal convolutional network | `lstm`, `tcn` |
| Relational | Temporal heterogeneous GNN with attention | `hetero_gnn` |
| Foundation | TabPFN v2, in-context | `tabpfn` |
| Fusion | Stacking meta-learner + isotonic calibration | always on |

`models.enabled` in the config picks which run. `tabpfn` additionally needs
`pip install tabpfn`; a tier whose dependency is missing is skipped with a
message rather than taking the run down.

## Design decisions worth defending in a viva

**Every aggregate is causal.** Velocity counters, rolling amount statistics,
graph degrees and target encodings are all computed from rows strictly earlier
than the row being scored (`features/causal.py`). `tests/test_causal.py` rewrites
the future and asserts that no earlier feature value moves. This is the
difference between a feature a live scorer could have and one that encodes the
answer.

**The GNN is temporal, not a static snapshot.** Most of the reviewed work
aggregates over a card's entire neighbourhood, including transactions that had
not happened yet. Here each entity node carries a memory that is read before
the transaction is scored and written after, so a transaction never informs its
own score and no future transaction reaches it (`models/graph.py`).

**The stacker is fitted on out-of-fold scores, and is not adopted blindly.**
Base models are fitted over expanding time windows to build the meta-learner's
training matrix. But the meta-learner is trained on scores from *fold* models
and applied to scores from the *final* models, and that gap can leave the stack
below its own strongest member. So the decision layer scores every candidate -
the stack and each base model - on held-out data and carries the winner
forward. On IEEE-CIS the stack loses, and the code says so rather than
presenting an ensemble that did not earn its place.

**Refitting is paired with a calibration holdout.** On a chronological split
the validation span is the data closest in time to the test period, so
withholding all of it from the final model wastes the most relevant rows.
Refitting on it naively, however, leaves the calibrator fitted on the pre-refit
model and applied to the refitted one - measured, that degraded expected
calibration error sixfold. Instead the validation span is cut: the earlier part
joins the refit, the later part is held back and scored by the refitted model,
and that holdout is what the calibrator is fitted on and what the selector uses.
Both then see the same model on rows it never trained on.

**Two tier policies are reported, not one.** The report's fixed bands
(Low 0-29, Medium 30-69, High 70-100) are computed as specified, and on this
data they work well - the high band holds about 2.5% of volume and around 70% of
all fraud at over 90% precision. A second policy derives the boundaries from the
cost model (`p* = C_FP / (C_FP + C_FN)`) clipped by the analyst review budget.
Both go into `reports/tables/`, so the specified policy and the one an operation
would actually run can be compared rather than assumed equivalent.

**Latency is measured on the whole serving path.** All enabled tiers are
reloaded from disk and reconnected to the feature frame, the graph tier's
memories are warmed to the start of the test span and then scored online, and
fusion and calibration are inside the timed region. Single-row and batched
timings are both reported, because a pipeline that scores 100k rows in fifteen
seconds can still take 100 ms on a single-row call - and the single-row number
is the one that decides whether real-time interdiction is feasible.

## Reading the outputs

```
reports/
  figures/    pr_curves, roc_curves, reliability, score_distribution,
              psi_drift, model_comparison, shap_importance      (150 dpi PNG)
  tables/     model_comparison, risk_tiers_config, risk_tiers_derived,
              psi_score_drift, psi_feature_drift, prevalence_drift,
              alert_reason_codes, shap_global_importance, latency,
              importance_<model>                                (CSV)
  metrics/    split_summary, base_model_fits, fusion_summary,
              model_metrics, evaluation_summary                 (JSON)
  run.log
```

`evaluation_summary.json` is the one file that contains everything.

Accuracy appears in the metrics block as
`accuracy_not_a_headline_metric` alongside `accuracy_of_always_legitimate`. It
is recorded so the report can show why it is useless here, not because it is
being used.

## Tests

```bash
python -m pytest -q
```

37 tests: the causal primitives against hand-computed answers, the future-leakage
check, split and fold construction, sequence-window construction, PSI behaviour,
metric sanity, calibration improvement, the risk and cost model, and an
end-to-end smoke run of all five stages on a small fixture.

## Configuration

`config/default.yaml` holds every tunable: split fractions, velocity windows,
imbalance strategy, per-model hyperparameters, calibration method, cost
constants, review capacity, PSI thresholds and metric targets. Nothing in the
code carries a magic number that is not traceable to a line in that file, and
that file is traceable to a table in the report.

## Layout

```
config/default.yaml       every threshold, window and hyper-parameter
src/frds/
  config.py utils.py      configuration, logging, seeding, timing
  plots.py                report figures
  pipeline.py cli.py      stage orchestration and the command line
  data/                   acquisition, synthetic fixture, ingestion, splitting
  features/               causal primitives and the feature builder
  models/                 the five tiers, the registry, imbalance, training
  fusion/                 stacking, calibration, risk scoring and tiers
  evaluation/             metrics, drift, latency, SHAP explanation
tests/                    unit tests and the end-to-end smoke test
reports/                  generated figures, tables and metrics
```

## The demonstration

```bash
streamlit run serving/dashboard.py          # analyst console
uvicorn serving.api:app --port 8000         # scoring endpoint
```

The console replays the held-out test span in chronological order and shows the
system deciding as traffic arrives: the alert queue an analyst would work, the
SHAP reason codes behind any alert they select, the tier breakdown, and the
serving cost per batch. Replaying a historical span in order is the proxy for a
live stream that the report's assumptions describe, and it is the configuration
in which the latency requirement is actually met - single-transaction scoring
costs around 126 ms against 0.24 ms amortised in batches.

Both surfaces are driven by the same `FraudScorer`, so they cannot disagree
about what the system does. It loads only what the fusion stage selected: when
that stage prefers a single model to the stack, one model is served rather than
six, because serving the five tiers the evaluation rejected would misrepresent
the system.

## Status and what is not here yet

Implemented and verified end to end: all five workflow stages, all five model
tiers, fusion, calibration, both tier policies, the cost model, SHAP reason
codes, PSI drift and latency measurement.

Not yet built, and deliberately out of scope for this increment:

- **Kafka transport.** The replay harness stands in for it. A broker would add
  setup fragility to a demonstration without changing what is demonstrated.

- **Optuna tuning.** The `tuning` block in the config is read but no search is
  wired up; hyper-parameters are currently the hand-set values in the config.
- **Adversarial robustness.** Report Table 4.2 lists a PGD-style evaluation
  under the robustness tier. Not implemented.
- **Bank Account Fraud secondary dataset.** The transfer check described in
  report Table 4.4.
