# Project status and decision log

Last updated: 22 Oct 2026, ~01:00 IST, end of the overnight run. **Everything below is
done and verified unless marked otherwise.** A new session should read this first, then
CLAUDE.md. Nothing has been committed to git - that waits for the user.

## 1. What the project is (plain language)

A fraud-detection product for online card payments. One raw transaction goes in; out
comes a 0-100 risk score, a verdict (Legitimate / Suspicious / Likely fraud), the action
(approve / step-up authentication / hold for manual review) and plain-English reasons,
in about 12 ms. Built and evaluated on IEEE-CIS (590,540 real transactions, 3.5% fraud),
split strictly by time: train days 1-121 ("textbook"), validation days 121-141
("practice tests", used for every choice), test days 141-183 ("final exam", 118,108
transactions, only for reporting). Bank Account Fraud (BAF, 1M applications) is the
second dataset used as a transfer check.

## 2. Final results (test span)

| | Phase-3 report (Sept) | **Final system** |
|---|---|---|
| Selected decision score | XGBoost | XGBoost + LightGBM, equal-weight log-odds average |
| Best raw AUC-PR (no-skill floor 0.034) | 0.8566 | **0.8613** |
| Decision layer AUC-PR (calibrated, what is served) | ~0.843 | **0.8478** |
| Expected calibration error | 0.00096 | **0.00044** |
| High tier (70-100) | 69.1% of fraud @ 93.7% precision | **70.3% @ 93.1%**, 2.6% of volume |
| Medium tier (30-69) | 7.5% of fraud @ 47% | 6.6% @ 51.7% |
| Modelled loss reduction | 67.4% | **68.2%** |
| AUC-PR under a 30-day label delay ("honest headline") | 0.573 (67.5% kept) | **0.649 (75.5% kept)** |
| One raw transaction -> decision | 136 ms (precomputed rows, all models) | **12.2 ms p50 / 19.4 ms p95** |
| ... including plain-English reasons | - | **55.2 / 66.4 ms** (still under 100 ms) |
| Micro-batch of 256 | 0.22 ms/txn (precomputed) | 11.4 ms/txn (features computed online) |
| Streaming at 40 txn/s, publish -> decision | - | 72 ms p50 / 161 ms p95 |

All models (test AUC-PR): blend 0.8613, XGBoost 0.8586, LightGBM 0.8561, decision layer
0.8478, CatBoost 0.8274, random forest 0.7457, logistic regression 0.7361, GNN 0.1122,
LSTM 0.0961. Validation (the choosing metric): blend_top2 0.8277, XGBoost 0.8267,
blend_top3 0.8235, LightGBM 0.8214, stack 0.8090, CatBoost 0.7848.

Other measured results:
- **Explanations:** plain-English reason per feature, evidence summed by family (entity
  fraud history ~70% of evidence), and a deletion test: removing an alert's top 3 reasons
  drops its score 92.4 points on average vs 1.2 for 3 random features (92% of alerts fall
  below half their score). Reasons are exact TreeSHAP (they add up to the model output).
- **Robustness (PGD transfer attack):** 73 attacker-controllable features, surrogate
  fidelity 0.920; zero-budget control 0.3% evasion; at 1 std PGD 13.0% vs random 10.8%;
  fraud recall drop at most 8.3% relative (literature ~20%). PGD beats random only at 1 std.
- **Behavioural weakness found by probing the live API:** a fraud transaction re-sent as
  new scored 78 (held); the identical one 1 s later scored 10 (approved), because one
  prior identical transaction makes every value "1.0x this account's average" = looks
  established. Recorded as a limitation in report 4.12; not fixed (would change the
  evaluated system). Suggested mitigation: minimum account history before trusting
  account-norm features; treat near-duplicates within seconds as a velocity signal.
- **Feedback loop works:** confirming that transaction as fraud raised the next score on
  the same card from 71 to 90.
- **BAF transfer:** decision layer AUC-PR 0.187 (12.7x floor), 55.0% recall at 5% FPR;
  fixed 0-29/30-69/70+ bands put 99.9% in "low" at 1.5% prevalence (bands must be
  derived per deployment); fairness: FPR 11.8% for age 50+ vs 3.9% under 50 (ratio 0.33).
- **Label-delay ablation (XGBoost / LightGBM):** immediate 0.8586 / 0.8561; 30 days
  0.6486 / 0.6561; 60 days 0.6141 / 0.6256; fraud-history features removed 0.5808 / 0.5895.
- **Deep-tier grid (20 configs, Kaggle GPU):** LSTM test 0.092-0.111, GNN 0.098-0.131;
  best on validation 0.191 -> 0.111 (LSTM), 0.199 -> 0.113 (GNN). Deep tiers do not help.
- **Drift:** worst score PSI 0.0106 across the test span (stable).

## 3. Decisions and why (do not relitigate without new evidence)

1. **Docker for everything.** Windows Smart App Control blocks native package DLLs; the
   user keeps it on.
2. **Verified PyPI mirror for image builds.** PyPI's CDN runs at 0.1-0.35 MB/s on this
   ISP. Every file is checked against pypi.org checksums before install (user approved).
3. **xgboost-cpu in the image** (the Linux xgboost wheel drags in a 305 MB NVIDIA lib).
4. **Choose on validation only; report test.** Applied to every change below.
5. **Bug fixes:** `causal.expanding_mean/std` did not skip missing values, blanking most
   per-entity D-column averages (fixing it is why the 30-day result rose to 0.649); the
   fraud-rate prior leaked test prevalence (now train span only); the report generator
   split CSVs on bare commas, corrupting Table 4.3 (now RFC 4180).
6. **Fusion adds equal-weight log-odds blends** of the top 2/3 tabular models. The
   learned stack keeps losing (it is fitted on fold-model scores, applied to final-model
   scores). The blend of XGBoost + LightGBM is selected.
7. **Optuna tuning: run, not adopted.** 52 trials, +0.003 validation, lower on test
   (XGBoost 0.8586 -> 0.8545). Hand-set parameters kept (`tuning.use_tuned: false`).
   Outputs: reports/tuned_run/.
8. **Account profile (346 -> 452 features): not adopted.** Validation 0.8271 vs 0.8277.
   `features.account_profile: false`. Run archived in reports/profile_run/.
9. **CatBoost: benchmarked, not used for decisions.** Validation 0.7848 alone; adding it
   lowers the blend (0.8235 vs 0.8277). It stays in `models.enabled` so it appears in the
   comparison table.
10. **TabPFN: integrated, not evaluated.** Its weights need a licence step (TABPFN_TOKEN
    from ux.priorlabs.ai) that batch GPU runs could not complete, even from the editor.
    The user decided (22 Oct) that the report does not mention it at all.
11. **GPU work on Kaggle** (LSTM, GNN, CatBoost, deep grid) via scripts/kaggle_job.py;
    checkpoints are fingerprint-matched and reused locally.
12. **Report:** chapters 3-5 regenerated; front matter, title page and section 5.3 are the
    group template and must NOT be copied over the user's individual report.

## 4. Where everything is

- Product: `src/frds/product.py` (DecisionService, FastTree, BlendExplainer),
  `src/frds/features/online.py` (online store), `src/frds/evaluation/reasons.py`
  (plain-English reasons). API `serving/api.py`; console `serving/dashboard.py` +
  `serving/views/{check,ledger,monitor}.py`.
- New stages: `models/tuning.py` (Optuna), `evaluation/robustness.py`, `transfer.py`
  (BAF), `stream.py` (Kafka / in-memory), `tracking.py` (MLflow, off by default).
- Scripts: `overnight.py` (unattended runner), `compare_feature_sets.py` (adoption rule),
  `kaggle_job.py`, `smoke_api.py`, `check_online_parity.py`, `watch_all.sh` (supervisor),
  `ablation_label_delay.py` (+ figure), `tune_deep_tiers.py`.
- Results: `reports/` (tables, figures, metrics; ablation_label_delay/, tuning/,
  baseline_handset/, tuned_run/, profile_run/, overnight_status.json).
- Report: `reports/Phase-3_Fraud_Detection_Report.docx` (938 KB). New/updated sections:
  4.2 (blends, TabPFN note), 4.3, 4.5 (reason sentences, evidence families, faithfulness),
  4.7, 4.9 (new ablation numbers + why they improved), 4.10 tuning, 4.11 account profile
  + CatBoost, 4.12 robustness + behavioural weakness, 4.13 BAF + fairness, 4.14 deployed
  product, 4.15 fairness now measured, 4.16 conclusions, chapter 5 rewritten (5.3 left
  alone), abstract, Table 3.6. Tables 4.1-4.19, figures 4.1-4.10. No placeholders.
- Demo bundle: `fraud-detection-demo-bundle-v2/` (277 MB; copy into the repo root, then
  `docker compose up api dashboard`). The old bundle is untouched. Both are gitignored
  (Kaggle data must not be redistributed).
- Tests: 67 passed in the last full run (3 min in Docker); `tests/test_catboost.py` 3/3.

## 5. Verified end to end tonight

- `docker compose up -d --no-build api dashboard` -> API :8000, console :8501 (both left
  running). `scripts/smoke_api.py` passed: health, real fraud example scored and held
  with reasons, outcome feedback raises the next score, legit example approved, batch.
- Console pages rendered headlessly with Streamlit AppTest: check-a-transaction (form
  submit shows verdict + 10 reason rows), live ledger (stepping works), model and
  monitoring (12 tables) - no exceptions.

## 6. For the user in the morning

1. Read README "Results" and section 2 above; open the console at http://localhost:8501.
2. Copy chapters 3-5 (and the abstract numbers) from the regenerated .docx into your
   individual report; keep your own title page, certificate and 5.3.
3. Decide whether to commit: suggested as a new branch, e.g. `feature/product-and-phase1-gaps`.
4. Viva points: time-based split; 0.861 AUC-PR (25x guessing); 70% of fraud in a 2.6%
   queue at 93% precision; 0.649 under a 30-day label delay (honest headline); 12 ms per
   transaction; faithful reasons; what was tried and rejected, and why (sections 3.6-3.10).

## 7. Known follow-ups (optional)

- Fix the history-conditioning weakness (section 2) and the age disparity on BAF.
- Done 22 Oct, ~09:40: `scripts/check_online_parity.py 2000` on the first 2,000 real
  test rows: 1,986 (99.30%) equal on all 346 features. The 14 others differ only in
  `roll_uid_amt_std` / `roll_uid_amt_z`, where all prior amounts of the account are
  (near-)identical: both sides compute std as sqrt(E[x^2] - mean^2) from running sums,
  so a true std of 0 becomes rounding noise that differs between the global offline
  cumsum and the per-entity online sums. User accepted 14/2000; no fix (a stable
  formula would change training features). Not measured: whether those 14 scores change.
- Done 22 Oct, ~11:00, user's instruction: report cleaned and rewritten (326 blocks).
  (a) TabPFN removed entirely - objective 3 is now FOUR model families; 1.2 bullet, 2.2
  sentence, survey row + reference (7 studies, refs renumbered [4]-[15]), Table 3.1 B.4,
  Table 3.2 row, tools, 4.2 paragraph, abstract, 4.16, 5.1. Code still has it (registry,
  label map) - harmless. (b) Tone: every "honest", "reported rather than suppressed /
  omitted", "ran against expectation", "failed", "earlier version got it wrong" etc.
  rewritten as neutral findings; 4.8 retitled "Analysis of the Sequential and Relational
  Tiers". (c) Accuracy fixes: 4.2 lift/accuracy now from the selected blend and "third
  decimal" -> "two percentage points"; 4.5 says attribution is the average of the blend
  members' TreeSHAP (matches pipeline.py:394); 4.7 dropped "laptop GPU"; 3.4 BAF role
  "future work" -> "section 4.13"; 4.14 now states the 99.3% real-row parity; 4.9 "almost
  nothing further" -> "comparatively little". Untouched: title page, certificate,
  authorship, 5.3 (still says "five model tiers" and has the group placeholder - user's).
  Script backups: scratchpad report_scripts_backup/ (session-local).
  Phase-1 deck still says eight studies and lists TabPFN (already presented; left as is).
- Done 22 Oct, user-approved cleanup: deleted __pycache__/, .pytest_cache, src/frds.egg-info
  (Docker uses PYTHONPATH), reports/overnight.log + run.log, root Fig-4.1/4.2 PNGs
  (identical to reports/figures/from_phase1/image3/4.png) and the stopped containers
  frds-run-final / frds-run-overnight. The user pushes next, then decides the rest:
  ablation per-cell dirs (1.2 GB), baseline_handset/artifacts, profile_run, demo bundle v1,
  TabPFN code, one-off scripts.
- Done 22 Oct: comments trimmed across the code at the user's request. Docstrings cut to
  their first paragraph (script/CLI usage lines kept; cli.py's docstring is the --help
  epilog); ~50 comment blocks of 3+ lines rewritten as one or two lines; JS report-script
  headers, Dockerfile, compose, watch scripts and config/default.yaml likewise. Verified:
  AST (minus docstrings) identical in all 71 .py files, YAML values, `docker compose
  config`, Dockerfile/shell code lines and the report's document.xml all byte-identical.
- Open, not investigated (user stopped it): that parity run printed online featurisation
  p50 61 ms via `store.score_and_update`, vs 11 ms features / 12.2 ms total in
  reports/tables/latency_product.csv (measured via `DecisionService.score`). The report
  quotes the latter; the two are different code paths.
- Done 22 Oct, 09:20: reference [7] (Alarfaj and Shahzadi, IEEE Access vol. 13) corrected
  2024 -> 2025 in the Phase-1 deck (slides 6 and 18) and the report generator (literature-review table in
  report_chapters.js, reference list in report_results.js); report rebuilt. The DOI says
  2024 (online Sept 2024) but IEEE style cites the volume year. Deck slide 17
  "robustness measured" is now true - no change needed. Pre-edit copies are not in the repo.

## 8. Overnight log (21-22 Oct, condensed)

- Built: online store, product path, API/console, SHAP reasons + faithfulness, Optuna,
  robustness, BAF, stream, MLflow, Docker, Kaggle job, CatBoost, account profile.
- 22:00 account-profile features built; parity tests pass. 22:28 Kaggle: CatBoost, LSTM,
  GNN on new features. 22:45 Kaggle: deep grid. 23:14 unattended runner started.
- 23:20 account profile not adopted -> revert. 23:34 Kaggle: CatBoost on base features,
  not adopted. 00:44 runner done (13/13 steps, 67 tests). 00:49 final pass with CatBoost
  benchmarked + throttled stream. 00:55 API smoke + console checks. 00:57 report built.
- Incidents handled: Smart App Control (-> Docker); slow PyPI (-> verified mirror); CUDA
  wheel pulled by xgboost (-> xgboost-cpu); CPU oversubscription (-> one job at a time);
  Claude Code memory safeguard killed a client process (-> detached containers); Kaggle
  secrets unavailable to CLI runs; compose `pull_policy: build` rebuilt on every `up`
  (-> removed); baseline snapshot had a combined score file from the wrong run (-> rebuilt
  from per-model checkpoints).
- Done 22 Oct: README rewritten (professional, no TabPFN). Committed cc6d626 as
  shreyasnandurkar (no co-author line, user's instruction) and pushed to origin/master
  (github.com/C0SM0X-KC/fraud-detection-risk-modelling). CLAUDE.md and this file are
  gitignored (local only), as is the Phase-1 deck.
