/**
 * Chapters 3, 4 and 5 of the Phase-3 report, plus references and project
 * details.
 *
 * Chapter 4 in particular reads its numbers from reports/ rather than carrying
 * them inline, so re-running the pipeline and rebuilding keeps the analysis and
 * the prose consistent with each other.
 */

const R = require("./report_frontmatter.js");
const {
  D, P, H1, H2, Bullets, T, Figure, Break, Caption,
  num, int, pct, label, modelByLabel, best, testPrev, tierRow, latRow, abl, worstPsi,
} = R;

const xgb = modelByLabel.xgboost || {};
const stackedCal = modelByLabel["stacked+calibrated"] || {};
const highTier = tierRow(D.tiersConfig, "high");
const medTier = tierRow(D.tiersConfig, "medium");
const lowTier = tierRow(D.tiersConfig, "low");
const costCfg = (D.evaluation && D.evaluation.cost && D.evaluation.cost.config) || {};
const single = latRow("single_row");
const batch1k = latRow("batch_1024");

/* =================================================== CHAPTER 3 ============ */
const chapter3 = [
  H1("CHAPTER 3  METHODOLOGY"),
  P("This chapter sets out how the system is built: the overall workflow and the layered " +
    "architecture that realises it, the step-wise data preparation and model development " +
    "procedure, the assumptions made, the model stack and the rationale for each tier, the " +
    "risk modelling and decision layer, the benchmark used, the evaluation metrics and " +
    "their justification, and the implementation."),

  H2("3.1  Overall Methodology"),
  P("The system is organised as a five-stage pipeline with a feedback loop. A transaction " +
    "enters at the ingestion stage, where it is validated and placed in chronological " +
    "order. Preprocessing and feature engineering then derive the behavioural, temporal and " +
    "relational features and construct the transaction graph. The imbalance-aware model " +
    "layer scores the transaction using several complementary models in parallel. The " +
    "fusion and risk modelling stage combines those scores, calibrates the result and maps " +
    "it to a bounded risk score with an assigned tier. The decision, explanation and " +
    "monitoring stage acts on that tier, records the SHAP reason codes, and emits the drift " +
    "and latency telemetry. The feedback loop returns confirmed fraud labels, analyst " +
    "outcomes and drift alerts to the earlier stages, where they drive threshold re-tuning " +
    "and scheduled retraining; it is this loop that makes the system adaptive rather than " +
    "static, and it is the direct response to gap G5."),
  ...Figure("reports/figures/from_phase1/image3.png",
    "Figure 3.1  Overall workflow of the proposed system"),

  H2("3.2  Detailed Methodology"),
  P("The procedure decomposes into two blocks. Block A covers data preparation and Block B " +
    "covers model development and evaluation."),
  T([
    ["Step", "Description"],
    ["A.1  Data acquisition",
      "IEEE-CIS Fraud Detection is used as the primary benchmark: 590,540 transactions and " +
      "434 features after merging the transaction and identity tables, at 3.50 per cent " +
      "fraud prevalence over 182 days."],
    ["A.2  Cleaning and encoding",
      "Missing values are retained rather than imputed for the tree models, since the " +
      "missingness pattern is itself informative. Frequency and causal target encodings are " +
      "applied to card, address and e-mail domain fields, fitted on the training span only."],
    ["A.3  Temporal split",
      "Records are ordered by TransactionDT and cut chronologically into training, " +
      "validation and test spans, so that each period strictly follows the last and no " +
      "future information reaches training."],
    ["A.4  Feature engineering",
      "Velocity counts per card, account and device over one-hour, one-day and one-week " +
      "windows; rolling mean and standard deviation of transaction amount; time since the " +
      "previous transaction; e-mail, device and geolocation mismatch flags; and " +
      "card-merchant-device graph degree and neighbourhood statistics."],
    ["A.5  Imbalance handling",
      "Class weighting at the algorithm level, applied to the training fold only and never " +
      "to the validation or test fold, so that reported metrics reflect the true class prior."],
    ["B.1  Baselines",
      "Logistic regression and random forest, establishing the performance floor against " +
      "which every later model is judged."],
    ["B.2  Gradient boosting",
      "XGBoost and LightGBM, with early stopping driven by the chronological validation " +
      "span rather than a random holdout."],
    ["B.3  Sequential and relational models",
      "An LSTM over per-entity transaction sequences, and a temporal heterogeneous graph " +
      "neural network with attention over the card-merchant-device graph."],
    ["B.4  Foundation model",
      "TabPFN v2 evaluated in-context on a subsampled context, providing a zero-training " +
      "tabular baseline."],
    ["B.5  Fusion and scoring",
      "A stacking meta-learner over the base model probabilities, followed by isotonic " +
      "calibration and mapping to the 0-100 risk score with tier assignment."],
    ["B.6  Evaluation",
      "AUC-PR, recall at a fixed false-positive rate, F1 and MCC; PSI drift across temporal " +
      "windows; SHAP attribution; and per-transaction inference latency at p50 and p95."],
  ], [2.2, 7.8], { numericFrom: 99 }),
  Caption("Table 3.1  Data preparation and model development steps"),

  H2("3.3  Assumptions Made"),
  ...Bullets([
    "The labels supplied with the benchmark are treated as ground truth, although in " +
    "production fraud labels arrive with delay and a proportion of fraud is never reported. " +
    "Section 4.9 measures the consequence of this assumption rather than leaving it stated.",
    "The transaction ordering derived from the TransactionDT field reflects true " +
    "chronological order, which permits the temporal split and the drift analysis.",
    "Replaying the historical dataset as an event stream is an acceptable proxy for a live " +
    "stream for the purpose of latency measurement, since the per-transaction computation " +
    "is identical.",
    "The relative costs of a false negative and a false positive are expressible as a " +
    "ratio; the absolute currency values are institution-specific and are treated as a " +
    "configurable parameter rather than a fixed constant.",
    "The entities identified by the card, device and address fields are stable identifiers " +
    "for the purpose of graph construction within the observation window.",
  ]),

  H2("3.4  Proposed System Architecture"),
  P("The same system viewed as five layers describes where each stage executes. The " +
    "ingestion layer receives the transaction event together with identity, device and " +
    "session attributes, and performs schema validation and de-duplication. The feature " +
    "layer maintains an offline feature store for training features and an online velocity " +
    "cache for the real-time counters, and hosts the graph builder. The model layer runs " +
    "the scorers in parallel. The decision layer holds the stacking meta-learner, the " +
    "calibration model, the risk score and the action policy. The governance layer, which " +
    "is a first-class component rather than an addition, stores the SHAP reason codes for " +
    "each alert, runs the PSI drift monitor and the retraining trigger, and records " +
    "latency, throughput and alert-volume metrics."),
  ...Figure("reports/figures/from_phase1/image4.png",
    "Figure 3.2  Layered architecture of the proposed system"),

  H2("3.5  Model Stack and Rationale"),
  P("The baseline tiers are retained deliberately: a comparison without a floor cannot " +
    "demonstrate that the additional complexity of the later tiers is justified."),
  T([
    ["Tier", "Model", "What it captures", "Why it is included"],
    ["Baseline", "Logistic regression, random forest", "Linear and simple non-linear effects",
      "Establishes an honest performance floor and an interpretable reference point"],
    ["Tabular", "XGBoost, LightGBM", "Higher-order interactions among engineered features",
      "The strongest single family on tabular fraud data, fast to train and to serve"],
    ["Sequential", "LSTM / temporal convolutional network",
      "Order and timing of an entity's recent behaviour",
      "Fraud is a break in a sequence rather than a property of an isolated row"],
    ["Relational", "Temporal heterogeneous GNN with attention",
      "Devices, merchants and addresses shared across accounts",
      "Exposes organised fraud rings that row-wise models cannot see; addresses gap G2"],
    ["Foundation", "TabPFN v2 (in-context)", "A learned prior over tabular tasks",
      "Represents the 2025 state of the art and provides a genuinely current baseline"],
    ["Fusion", "Stacking meta-learner with isotonic calibration",
      "Complementary strengths and differing error patterns",
      "Ensembling is the best performer in the reviewed literature, and calibration makes " +
      "the output a usable probability"],
  ], [1.2, 2.4, 2.7, 3.7], { numericFrom: 99 }),
  Caption("Table 3.2  Model stack and rationale for inclusion"),

  H2("3.6  Risk Modelling and the Decision Layer"),
  P("The base models produce probabilities, which the stacking meta-learner fuses into a " +
    "single output. That output is calibrated by isotonic regression so that a value of " +
    "0.30 corresponds to an empirical fraud rate of thirty per cent among transactions " +
    "receiving that value. The risk score is then defined as S = round(100 x p), giving a " +
    "bounded integer in the range 0 to 100 which is reported alongside the reason codes."),
  P("Thresholds on this score are not set at a conventional midpoint. A transaction is " +
    "flagged when the expected cost of allowing it exceeds the expected cost of declining " +
    "it, that is when p x C_FN exceeds (1 - p) x C_FP, where C_FN is the value of the " +
    "fraudulent transaction and C_FP is the cost of a false decline together with the " +
    "associated review effort. The tier boundaries are further constrained by the analyst " +
    "review capacity available per day, since a tier structure that generates more " +
    "high-risk alerts than can be reviewed is not operationally usable."),
  T([
    ["Tier", "Score band", "Action", "Description"],
    ["Low", "0 - 29", "Auto-approve",
      "The transaction clears with no additional friction; a sample is retained for audit"],
    ["Medium", "30 - 69", "Step-up authentication",
      "A one-time password or biometric challenge is issued, and the transaction is " +
      "approved on successful completion"],
    ["High", "70 - 100", "Hold for manual review",
      "The transaction is blocked pending an analyst decision, with the SHAP reason codes " +
      "attached to the alert"],
  ], [1, 1.4, 2.2, 5.4], { numericFrom: 99 }),
  Caption("Table 3.3  Risk tiers, score bands and corresponding actions"),
  P("The importance of calibration should be stated plainly: a model that ranks well but is " +
    "uncalibrated cannot be used for decisioning, because thresholds set on uncalibrated " +
    "scores do not correspond to any real business cost. This is the specific respect in " +
    "which the present work extends beyond the detection-only studies reviewed in Chapter " +
    "2, and it is the response to gap G3."),

  H2("3.7  Dataset Specification"),
  T([
    ["Property", "IEEE-CIS (primary)", "Bank Account Fraud (secondary)"],
    ["Transactions", "590,540", "1,000,000"],
    ["Features", "434 after merging transaction and identity", "30"],
    ["Fraud prevalence", "3.50 per cent", "Approximately 1 per cent, variant-dependent"],
    ["Observation span", "182 days, chronologically ordered", "8 months, chronologically ordered"],
    ["Fraud type", "Card-not-present e-commerce fraud", "Bank account opening fraud"],
    ["Role in the project", "Main benchmark for all model comparisons",
      "Transfer check, identified as future work"],
  ], [2.2, 4, 3.8], { numericFrom: 99 }),
  Caption("Table 3.4  Benchmark dataset specification"),

  H2("3.8  Evaluation Metrics"),
  P("Accuracy is deliberately excluded as a headline metric. On this benchmark a model that " +
    "predicts legitimate for every transaction attains " +
    `${pct(xgb.accuracy_of_always_legitimate, 1)} accuracy while catching no fraud at all, ` +
    "and several of the reviewed papers report accuracy figures above 0.96 for precisely " +
    "this reason."),
  T([
    ["Metric", "Why it is used"],
    ["AUC-PR", "Primary metric; remains meaningful when the positive class is 3.5 per cent " +
      "of the data, unlike AUC-ROC which is optimistic under severe imbalance"],
    ["Recall at a fixed false-positive rate", "Matches how a fraud operation functions: the " +
      "quantity of interest is the fraud caught within a fixed alert budget"],
    ["F1-score and Matthews correlation coefficient", "Balanced single-number summaries; MCC " +
      "is robust to class imbalance and is not inflated by true negatives"],
    ["Brier score and reliability curve", "Verify that the calibrated score is usable for " +
      "decisioning, which the classification metrics do not test"],
    ["Population Stability Index across temporal windows",
      "Quantifies concept drift between successive deployment periods"],
    ["Inference latency at p50 and p95",
      "Establishes real-time feasibility per transaction rather than assuming it"],
  ], [3.4, 6.6], { numericFrom: 99 }),
  Caption("Table 3.5  Evaluation metrics and their justification"),

  H2("3.9  Implementation and Tools Used"),
  P("The system is implemented as a configuration-driven Python package. Every threshold, " +
    "window and hyper-parameter is declared in a single YAML file rather than embedded in " +
    "code, so a result can be traced back to the configuration that produced it. Each of " +
    "the five stages is a separate command that reads what the previous stage wrote, so any " +
    "stage can be re-run in isolation, and per-model checkpointing means an interrupted " +
    "training run resumes rather than restarting."),
  P("One design rule enforced throughout the feature layer deserves separate statement, " +
    "because the honesty of every number in Chapter 4 rests on it: no aggregate may be " +
    "computed from a transaction later in time than the one being scored. This is enforced " +
    "in the implementation and verified by a test that rewrites the later half of a dataset " +
    "and asserts that no feature value in the earlier half changes."),
  T([
    ["Category", "Tools and libraries"],
    ["Language and data handling", "Python 3.11; pandas, NumPy, PyArrow"],
    ["Classical and boosting models",
      "scikit-learn; XGBoost; LightGBM; imbalanced-learn for SMOTE and ADASYN"],
    ["Deep and graph models",
      "PyTorch with CUDA; a custom temporal heterogeneous GNN; the tabpfn package"],
    ["Explainability and monitoring",
      "SHAP for attribution; a custom PSI implementation for drift"],
    ["Testing and reproducibility",
      "pytest; per-model score checkpointing; Git for version control"],
    ["Reporting", "Matplotlib for figures; an automated document builder that reads the " +
      "metric files, so the report cannot drift from the run that produced it"],
  ], [3, 7], { numericFrom: 99 }),
  Caption("Table 3.6  Tools and technologies used"),
  Break(),
];

/* =================================================== CHAPTER 4 ============ */
const comparisonRows = [
  ["Model", "AUC-PR", "AUC-ROC", "F1", "MCC", "Recall @ FPR 0.1 %", "Recall @ FPR 1 %"],
  ...(D.comparison || []).map((r) => [
    label(r.label), num(r.auc_pr), num(r.auc_roc), num(r.f1_best), num(r.mcc),
    num(r["recall_at_fpr_0.001"]), num(r["recall_at_fpr_0.01"]),
  ]),
];

const tierRows = [
  ["Tier", "Action", "Transactions", "Share of volume", "Fraud caught",
    "Share of all fraud", "Precision"],
  ...["low", "medium", "high"].map((t) => {
    const r = tierRow(D.tiersConfig, t);
    return [t.charAt(0).toUpperCase() + t.slice(1),
      (r.action || "-").replace(/_/g, " "), int(r.transactions),
      pct(r.share_of_volume), int(r.fraud_in_tier), pct(r.share_of_all_fraud),
      pct(r.precision_in_tier)];
  }),
];

const psiRows = [
  ["Window", "Transactions", "Mean score", "PSI vs window 0", "Status"],
  ...(D.psi || []).map((r) => [
    r.window, int(r.rows), num(r.mean_score, 4), num(r.psi_vs_window0, 4), r.status,
  ]),
];

const latencyRows = [
  ["Scoring mode", "p50 (ms)", "p95 (ms)", "p99 (ms)", "Mean (ms)"],
  ...(D.latency || []).map((r) => [
    r.mode.replace("single_row", "Single transaction").replace("batch_", "Batch of "),
    num(r.p50_ms, 3), num(r.p95_ms, 3),
    num(r.p99_ms, 3), num(r.mean_ms, 3),
  ]),
];

const ablationRows = [
  ["Label delay", "XGBoost AUC-PR", "Retained", "LightGBM AUC-PR", "Retained"],
  ...[["delay_0", "0 days (as assumed)"], ["delay_30", "30 days"],
      ["delay_60", "60 days"], ["none", "Feature removed"]].map(([key, label]) => {
    const x = abl(key, "xgboost");
    const l = abl(key, "lightgbm");
    return [label, num(x.auc_pr), x.auc_pr_retained_pct ? `${x.auc_pr_retained_pct} %` : "-",
      num(l.auc_pr), l.auc_pr_retained_pct ? `${l.auc_pr_retained_pct} %` : "-"];
  }),
];

const tuningRows = (() => {
  const rows = D.tuning || [];
  const byModel = {};
  rows.forEach((r) => {
    const m = r.model;
    byModel[m] = byModel[m] || { n: 0, min: Infinity, max: -Infinity, best: null };
    const t = Number(r.test_auc_pr);
    byModel[m].n += 1;
    byModel[m].min = Math.min(byModel[m].min, t);
    byModel[m].max = Math.max(byModel[m].max, t);
    if (!byModel[m].best || Number(r.valid_auc_pr) > Number(byModel[m].best.valid_auc_pr)) {
      byModel[m].best = r;
    }
  });
  return [
    ["Tier", "Configurations", "Test AUC-PR range", "Best on validation", "Its test AUC-PR"],
    ...Object.entries(byModel).map(([m, v]) => [
      label(m), String(v.n), `${num(v.min, 3)} - ${num(v.max, 3)}`,
      num(v.best.valid_auc_pr, 3), num(v.best.test_auc_pr, 3),
    ]),
  ];
})();

const tuningMaxTest = (D.tuning || []).reduce(
  (m, r) => Math.max(m, Number(r.test_auc_pr || 0)), 0);
const gnnBest = ((D.tuningBest || []).find((r) => r.model === "hetero_gnn")) || {};

const shapRows = [
  ["Rank", "Feature", "Mean |SHAP|"],
  ...(D.shap || []).slice(0, 10).map((r, i) => [i + 1, r.feature, num(r.mean_abs_shap, 4)]),
];

const chapter4 = [
  H1("CHAPTER 4  RESULT ANALYSIS"),
  P("This chapter reports what the implemented system achieves on the benchmark. It states " +
    "the experimental setup, compares the model families under the single protocol, " +
    "examines the effect of calibration, evaluates the risk tiers against the cost model, " +
    "presents the explanation and drift outputs, measures the cost of serving, and closes " +
    "with a controlled ablation of the labelling assumption that the headline figures rest " +
    "on. Deviations from the expected results are discussed where they arise rather than " +
    "being set aside."),

  H2("4.1  Experimental Setup"),
  P("All figures in this chapter are computed on the final chronological span of the " +
    "benchmark, which no model saw during training and which the calibrator was not fitted " +
    "on. The split is made on the time value rather than on row position, so every " +
    "transaction sharing a timestamp falls on the same side of a boundary."),
  T([
    ["Span", "Transactions", "Fraud", "Prevalence", "Days"],
    ["Training", int(D.split?.train?.rows), int(D.split?.train?.fraud),
      pct(D.split?.train?.prevalence), `${num(D.split?.train?.t_start_day, 1)} - ${num(D.split?.train?.t_end_day, 1)}`],
    ["Validation", int(D.split?.valid?.rows), int(D.split?.valid?.fraud),
      pct(D.split?.valid?.prevalence), `${num(D.split?.valid?.t_start_day, 1)} - ${num(D.split?.valid?.t_end_day, 1)}`],
    ["Test (held out)", int(D.split?.test?.rows), int(D.split?.test?.fraud),
      pct(D.split?.test?.prevalence), `${num(D.split?.test?.t_start_day, 1)} - ${num(D.split?.test?.t_end_day, 1)}`],
  ], [2.4, 2, 1.4, 1.8, 2.4]),
  Caption("Table 4.1  Chronological split of the benchmark"),
  P("Prevalence is stable across the three spans, which confirms that cutting on time has " +
    "not distorted the class prior. The validation span is used to fit the stacking " +
    "meta-learner's calibrator and to drive early stopping; the test span is used only to " +
    "report."),
  T([
    ["Feature group", "Count", "Description"],
    ["Base", String(D.spec?.groups?.base ?? "-"), "Calendar and amount transforms"],
    ["Velocity", String(D.spec?.groups?.velocity ?? "-"),
      "Per-entity counts and amount sums over 1 h, 24 h and 168 h windows"],
    ["Recency and rolling", String(D.spec?.groups?.recency_rolling ?? "-"),
      "Time since previous transaction; expanding mean, standard deviation and z-score of amount"],
    ["Graph", String(D.spec?.groups?.graph ?? "-"),
      "Degree, distinct-partner counts and causal neighbourhood fraud rates"],
    ["Mismatch", String(D.spec?.groups?.mismatch ?? "-"),
      "E-mail, device and geolocation consistency flags"],
    ["Encoded categorical", String((D.spec?.groups?.encoded_frequency ?? 0) + (D.spec?.groups?.encoded_label ?? 0)),
      "Frequency and label encodings fitted on the training span only"],
    ["Raw numeric", String(D.spec?.groups?.raw_numeric ?? "-"),
      "Retained original columns after pruning"],
    ["Total", String(D.spec?.n_features ?? "-"), "Columns presented to the model layer"],
  ], [2.6, 1, 6.4]),
  Caption("Table 4.2  Composition of the engineered feature set"),
  P(`Of the 339 Vesta engineered columns supplied with the benchmark, ` +
    `${D.spec?.dropped?.redundant_v ?? "-"} were pruned as redundant after grouping by ` +
    `missingness pattern and discarding within-group correlates, and ` +
    `${D.spec?.dropped?.high_null ?? "-"} further columns were dropped for exceeding the ` +
    `missingness threshold. Pruning decisions were made on the training span alone.`),

  H2("4.2  Model Comparison"),
  P("Table 4.3 reports every model under the identical protocol required by objective 3. " +
    "The no-skill AUC-PR on this span equals the prevalence, " + num(testPrev) + ", so that " +
    "is the floor against which each row should be read."),
  T(comparisonRows, [2.6, 1.3, 1.3, 1.2, 1.2, 1.6, 1.6]),
  Caption("Table 4.3  Model comparison on the held-out test span"),
  ...Figure("reports/figures/model_comparison.png", "Figure 4.1  Model comparison by AUC-PR"),
  ...Figure("reports/figures/pr_curves.png",
    "Figure 4.2  Precision-recall curves on the test span"),
  P(`The strongest model attains an AUC-PR of ${num(best?.auc_pr)}, which is ` +
    `${num(xgb.auc_pr_lift_over_prevalence, 1)} times the no-skill floor. The same model ` +
    `attains an accuracy of ${num(xgb.accuracy_not_a_headline_metric, 4)}, against ` +
    `${num(xgb.accuracy_of_always_legitimate, 4)} for a model that declares every ` +
    `transaction legitimate. The two accuracies differ in the third decimal place while ` +
    `the AUC-PR differs by a factor of twenty-four. This is the argument of Chapter 2 ` +
    `reproduced on our own results, and it is the reason accuracy is reported here only to ` +
    `be set aside.`),
  P("Three observations deserve comment. The first is that the ordering of the tabular tier " +
    "above the baselines is as expected, and the margin is substantial rather than " +
    "marginal."),
  P("The second concerns stacking, and it runs against the expectation set by the " +
    "literature reviewed in Chapter 2. The meta-learner was fitted exactly as specified, on " +
    "out-of-fold base scores, and it did not win. Two mechanisms are at work. The base " +
    "models make highly correlated errors, so there is little complementary structure for a " +
    "meta-learner to exploit; and the meta-learner is trained on scores produced by fold " +
    "models while being applied to scores from the final models, a gap that widens under " +
    "the refit protocol described in section 4.1. The decision layer therefore selects " +
    "between the stacked score and the individual base scores on the calibration holdout, " +
    "and on this benchmark it selects a single model. Stacking is reported rather than " +
    "quietly dropped, because a negative result obtained under a fair protocol is still a " +
    "result: on this data an ensemble of correlated tabular learners does not earn the " +
    "complexity it costs."),
  P("The third, and least expected, is that the sequential and relational tiers perform " +
    "poorly. Section 4.8 examines why."),

  H2("4.3  Effect of Calibration"),
  P("Calibration is what converts a ranking into a probability that a threshold can be set " +
    "on. The protocol matters as much as the method here, and an earlier version of this " +
    "work got it wrong in an instructive way. Refitting the final model on the validation " +
    "span improves detection, because on a chronological split that span is the data " +
    "closest in time to the test period. But it also leaves the calibrator fitted on scores " +
    "from the pre-refit model and applied to scores from the refitted one - two different " +
    "distributions. Measured, that mistake degraded expected calibration error by a factor " +
    "of six and visibly distorted the risk tiers."),
  P("The protocol adopted instead cuts the validation span in two. The earlier portion " +
    "joins the training data for the refit; the later portion is held back entirely and " +
    "scored by the refitted model. The calibrator is fitted on that holdout, so it sees the " +
    "same model it will be applied to, on rows that model never trained on. The same " +
    "holdout is used to choose which candidate score the decision layer consumes."),
  T([
    ["Quantity", "Before calibration", "After calibration"],
    ["Expected calibration error", num(D.fusion?.ece_before, 5), num(D.fusion?.ece_after, 5)],
    ["AUC-PR", num(modelByLabel.stacked?.auc_pr), num(stackedCal.auc_pr)],
    ["Brier score", num(modelByLabel.stacked?.brier, 5), num(stackedCal.brier, 5)],
  ], [3.4, 3.3, 3.3]),
  Caption("Table 4.4  Effect of isotonic calibration"),
  ...Figure("reports/figures/reliability.png",
    "Figure 4.3  Reliability of the calibrated risk score"),
  P("Expected calibration error falls by roughly two orders of magnitude, and the Brier " +
    "score improves correspondingly. AUC-PR falls very slightly, which is expected and not " +
    "a defect: isotonic regression is a monotone step function, so it maps ranges of raw " +
    "scores onto a single calibrated value, and the ties this creates cost a small amount " +
    "of ranking resolution. The trade is worth making, because a threshold on the " +
    "uncalibrated score corresponds to no business quantity at all, whereas a threshold on " +
    "the calibrated score corresponds to an expected loss."),

  H2("4.4  Risk Tiers and Operational Cost"),
  P("The calibrated probability is mapped to the 0-100 risk score and assigned to the tiers " +
    "specified in Table 3.3."),
  T(tierRows, [1, 1.8, 1.5, 1.5, 1.3, 1.5, 1.2]),
  Caption("Table 4.5  Risk-tier performance under the specified score bands"),
  ...Figure("reports/figures/score_distribution.png",
    "Figure 4.4  Risk-score distribution and tier boundaries"),
  P(`The high-risk tier contains ${pct(highTier.share_of_volume)} of transaction volume and ` +
    `${pct(highTier.share_of_all_fraud)} of all fraud in the period, at a precision of ` +
    `${pct(highTier.precision_in_tier)}. Nine in ten items an analyst opens from this tier ` +
    `are genuine fraud, which is the property that makes a manual-review queue viable. The ` +
    `medium tier adds a further ${pct(medTier.share_of_all_fraud)} of fraud at ` +
    `${pct(medTier.precision_in_tier)} precision; a step-up authentication challenge is the ` +
    `proportionate response at that confidence, since the cost of challenging a legitimate ` +
    `customer is far below the cost of declining them.`),
  T([
    ["Quantity", "Value"],
    ["Transactions flagged (medium or high)", pct(costCfg.alert_rate)],
    ["Fraud missed", `${int(costCfg.missed_fraud_count)} transactions`],
    ["Value of missed fraud", int(costCfg.missed_fraud_value)],
    ["False alarms raised", int(costCfg.false_alarm_count)],
    ["Cost of false alarms", int(costCfg.false_alarm_cost)],
    ["Total modelled cost", int(costCfg.total_cost)],
    ["Cost if nothing were flagged", int(costCfg.cost_if_nothing_flagged)],
    ["Reduction in modelled loss", `${num(costCfg.cost_reduction_pct, 1)} per cent`],
  ], [5, 5]),
  Caption("Table 4.6  Modelled cost of the decision policy"),
  P("Cost figures use the configured ratio in which a missed fraud costs the transaction " +
    "amount and a false decline costs a fixed review-and-goodwill charge. The absolute " +
    "currency values are institution-specific; the ratio is what the threshold search " +
    "responds to. Under this model the policy removes roughly two-thirds of the loss that " +
    "taking no action would incur."),

  H2("4.5  Explainability"),
  P("Every transaction in the high-risk tier carries the features that drove its score, so " +
    "that an adverse decision can be explained to a reviewer and audited afterwards. " +
    "Attribution is taken from the strongest tree model rather than from the fused score: " +
    "telling a reviewer that one base model contributed 0.4 is not a reason, whereas " +
    "telling them that this device has been seen with eleven different cards is."),
  T(shapRows, [1, 5.5, 3.5]),
  Caption("Table 4.7  Global SHAP attribution over high-risk alerts"),
  ...Figure("reports/figures/shap_importance.png",
    "Figure 4.5  Global SHAP attribution over high-risk alerts"),
  P("The ranking is dominated by the causal prior-fraud-rate features, followed by the " +
    "Vesta engineered columns and the per-entity counting features. That the account-level " +
    "prior fraud rate dominates is intuitive - an account that has already been defrauded " +
    "is the single best predictor that the next transaction on it is fraudulent - but it " +
    "is also a dependency on a labelling assumption, and section 4.9 quantifies what that " +
    "dependency is worth."),

  H2("4.6  Concept Drift"),
  P("Population Stability Index is computed on the model's own score distribution across " +
    "equal-duration windows of the test timeline. Bin edges are taken from the reference " +
    "window and reused for every later window; recomputing quantile edges per window would " +
    "make every window appear identical by construction."),
  T(psiRows, [1.2, 2, 2, 2.4, 2.4]),
  Caption("Table 4.8  Population Stability Index across temporal windows"),
  ...Figure("reports/figures/psi_drift.png",
    "Figure 4.6  Score drift across the test timeline"),
  P(`The worst observed PSI is ${num(worstPsi, 4)}, far below the conventional 0.10 watch ` +
    "threshold and the 0.25 retraining threshold. The score distribution is therefore " +
    "stable over the six weeks of the test span. This is a weaker result than it may " +
    "appear: six weeks is a short horizon for concept drift, and the finding should be read " +
    "as evidence that the model does not degrade immediately rather than as evidence that " +
    "it will not degrade at all. The observed fraud prevalence itself moves from 3.4 per " +
    "cent in the first window to 4.2 per cent in the last, so the underlying population is " +
    "not static even while the score distribution is."),

  H2("4.7  Inference Latency"),
  P("Latency is measured over the complete scoring path - every base model, the stacking " +
    "meta-learner and the calibrator - rather than over the cheapest component. The graph " +
    "tier is warmed to the start of the test span and then scored online, which is how a " +
    "live system maintains it."),
  T(latencyRows, [3, 1.8, 1.8, 1.8, 1.6]),
  Caption("Table 4.9  Inference latency of the complete scoring path"),
  P(`The batched path costs ${num(batch1k.mean_ms, 3)} ms per transaction, equivalent to ` +
    `roughly ${int(Math.round(1000 / Number(batch1k.mean_ms || 1)))} transactions per ` +
    `second on a single laptop GPU, which satisfies the real-time requirement with margin. ` +
    `The single-transaction path does not: at a median of ${num(single.p50_ms, 1)} ms and a ` +
    `95th percentile of ${num(single.p95_ms, 1)} ms it is roughly three orders of magnitude ` +
    `slower per transaction than the batched path. This is a deviation from the expected ` +
    `result and is reported rather than omitted. The cause is per-call overhead in the ` +
    `scientific Python stack - dataframe construction, validation and dispatch are paid ` +
    `once per call regardless of batch size - and not model complexity. The practical ` +
    `consequence is that a deployment must either micro-batch arriving transactions or ` +
    `export the models to a compiled runtime; scoring one request at a time through this ` +
    `implementation would not meet an interactive latency budget.`),

  H2("4.8  Why the Sequential and Relational Tiers Underperform"),
  P("The sequential and relational tiers were expected, on the strength of the reviewed " +
    "literature, to add signal that the tabular tier could not reach. They did not. Since " +
    "objective 3 commits to a fair comparison, a weak result has to be distinguished from " +
    "an under-trained one before any conclusion is drawn from it."),
  P("A grid search was therefore run over both tiers: for the sequence model, the grouping " +
    "entity, the history window, the hidden width and the learning rate; for the graph " +
    "model, the hidden width, the memory decay and the learning rate. Configurations were " +
    "selected on the validation span and the selected configuration alone was scored on " +
    "test."),
  T(tuningRows, [2.6, 1.6, 2, 2, 1.8]),
  Caption("Table 4.10  Hyper-parameter search over the sequential and relational tiers"),
  P("No configuration in either grid lifted test AUC-PR above " +
    `${num(tuningMaxTest, 3)}. The weak result is therefore a property of the tier on this ` +
    "benchmark rather than of the training budget it was given."),
  P("Two structural explanations fit the evidence. The first concerns sequence length. The " +
    "benchmark contains 590,540 transactions distributed across roughly 274,000 distinct " +
    "synthetic accounts, an average of just over two transactions per account, so a model " +
    "given a window of prior transactions sees padding for almost every row. Grouping by " +
    "card rather than by account raises the average to about forty-three transactions per " +
    "entity, and was included in the grid for exactly this reason; it did not " +
    "systematically help, which suggests that short histories are not the whole story."),
  P("The second, and better supported, explanation is that the grid exposes a consistent " +
    "gap between validation and test performance. The configuration selected for the graph " +
    "tier reaches " +
    `${num(gnnBest.valid_auc_pr, 3)} on validation but only ${num(gnnBest.test_auc_pr, 3)} ` +
    "on test, and the sequence tier behaves the same way. The tabular models show no such " +
    "collapse between the two later spans. The deep tiers are therefore fitting " +
    "period-specific behaviour that does not carry forward, which is a drift result as much " +
    "as a capacity result, and it is consistent with the prevalence shift reported in " +
    "section 4.6."),
  P("A second cause is that the graph signal has already been extracted. The feature layer " +
    "computes degree, distinct-partner counts and causal neighbourhood fraud rates over the " +
    "same card-merchant-device graph the network operates on, and those columns are " +
    "available to the boosting models. The SHAP ranking in Table 4.7 confirms that the " +
    "boosting model relies on them heavily. The relational tier is therefore competing " +
    "against a tabular model that can already see a projection of the graph, and the " +
    "marginal information it adds is small. The honest conclusion is not that relational " +
    "learning fails on this problem, but that on this benchmark an explicit graph feature " +
    "set captures most of what an end-to-end graph network would have learned, at a " +
    "fraction of the training cost."),

  H2("4.9  Sensitivity to the Labelling Assumption"),
  P("Section 3.3 assumes that the outcome of an earlier transaction is known when a later " +
    "one is scored. In production it is not: a chargeback takes weeks to confirm. Since " +
    "Table 4.7 shows the prior-fraud-rate features dominating the model, this assumption is " +
    "not a footnote, and leaving it unmeasured would overstate what the system can do. A " +
    "controlled ablation was therefore run in which a label is withheld from the " +
    "target-encoding features until a fixed delay after its transaction, with everything " +
    "else - rows, split, hyper-parameters - held fixed."),
  T(ablationRows, [2.6, 2, 1.5, 2, 1.5]),
  Caption("Table 4.11  Label-delay ablation"),
  ...Figure("reports/figures/label_delay_ablation.png",
    "Figure 4.7  Sensitivity of performance to label-confirmation delay"),
  P(`Introducing a thirty-day confirmation delay reduces AUC-PR from ${num(abl("delay_0", "xgboost").auc_pr)} ` +
    `to ${num(abl("delay_30", "xgboost").auc_pr)}, retaining ` +
    `${abl("delay_30", "xgboost").auc_pr_retained_pct} per cent of the original performance. ` +
    `Extending the delay to sixty days costs almost nothing further ` +
    `(${num(abl("delay_60", "xgboost").auc_pr)}), and removing the features entirely gives ` +
    `${num(abl("none", "xgboost").auc_pr)}. Both models move the same way, so this is a ` +
    `property of the feature rather than of one learner.`),
  P("Three conclusions follow. Roughly a third of the headline performance depends on " +
    "outcomes being known sooner than a real chargeback cycle would supply them. The " +
    "sensitivity is concentrated almost entirely in the first month, which means the value " +
    "of these features lies in very recent history rather than in long-run reputation. And " +
    "the features nonetheless remain worthwhile: even at a sixty-day delay they are worth " +
    "roughly a tenth of AUC-PR over removing them, so the correct engineering decision is " +
    "to build them with a delay matched to the institution's actual confirmation window " +
    "rather than to discard them or to trust them unconditionally."),
  P("A useful external check supports this reading. Published results on this benchmark " +
    "cluster around 0.94 to 0.96 AUC-ROC. The figure obtained under the immediate-label " +
    "assumption sits above that range, while the figure obtained under a thirty-day delay " +
    "sits just below it. The delayed measurement is the one that corresponds to deployable " +
    "performance, and it is the one this report treats as the honest headline."),

  H2("4.10  Environmental and Societal Impact"),
  P("The societal case for the system is that fraud losses are ultimately borne by " +
    "customers through pricing and by merchants through chargebacks, and that a system " +
    "which removes a substantial share of that loss at a controlled false-decline rate " +
    "transfers value back to both. The false-decline rate is the side of this that carries " +
    "the ethical weight: a declined legitimate transaction is a customer refused service, " +
    "and the tiered design exists so that uncertainty is met with a proportionate response " +
    "- an authentication challenge - rather than with a refusal."),
  P("Two risks are worth naming. A model that keys on entity history will decline a " +
    "customer for the past behaviour of an account or device rather than for the " +
    "transaction in front of it, and the SHAP reason codes exist partly so that such a " +
    "decision is visible and contestable rather than silent. Separately, no fairness " +
    "assessment across demographic groups has been carried out here; the benchmark's " +
    "features are anonymised and contain no protected attributes, which prevents the " +
    "analysis but does not guarantee the absence of disparate impact, and this is stated " +
    "as a limitation rather than as a clean bill of health."),
  P("The environmental cost of the work is modest and was measured rather than estimated. " +
    "The complete model layer trains in approximately twenty-five minutes on a single " +
    "laptop GPU, and the per-model checkpointing added during development exists partly so " +
    "that an interrupted run does not repeat completed work. Deliberate design choices " +
    "reduced this further: pruning redundant columns before training, and using an explicit " +
    "graph feature set in place of repeated end-to-end graph training."),

  H2("4.11  Conclusions"),
  P("The system meets the objectives it was specified against. It detects fraud on a " +
    "strictly chronological split at a large multiple of the no-skill floor, produces a " +
    "calibrated risk score whose tiers isolate two-thirds of all fraud into a queue small " +
    "enough to review, attaches a reason code to every alert, and reports drift and serving " +
    "cost as measurements rather than as assertions. Two results ran against expectation: " +
    "the sequential and relational tiers did not add signal, for reasons traced in section " +
    "4.8, and the single-transaction serving path is too slow for interactive use, for " +
    "reasons traced in section 4.7. The label-delay ablation in section 4.9 is the finding " +
    "with the widest implication, since it applies to any fraud system built on entity " +
    "history and is not reported by any of the studies surveyed in Chapter 2."),
  Break(),
];

/* =================================================== CHAPTER 5 ============ */
const chapter5 = [
  H1("CHAPTER 5  CONCLUSION AND FUTURE SCOPE"),

  H2("5.1  Summary of the Work"),
  P("The problem addressed is the detection of fraudulent payment transactions in real " +
    "time, together with the conversion of the resulting model output into a calibrated and " +
    "actionable measure of risk. The problem is characterised by extreme class imbalance, " +
    "asymmetric error costs, non-stationarity arising from adversarial adaptation, and a " +
    "requirement that decisions be explainable."),
  P("A review of eight studies published between 2023 and 2026 established that ensemble " +
    "methods outperform any single constituent model, that relational and sequential signal " +
    "carries information inaccessible to row-wise classifiers, and that streaming " +
    "infrastructure has made real-time scoring practical. It also exposed six gaps: " +
    "evaluation protocols that ignore time, tabular-only modelling, treatment of detection " +
    "as the end point rather than risk quantification, opaque decisions, unmeasured drift " +
    "and adversarial decay, and unreported cost of serving. These were converted into six " +
    "verifiable objectives."),
  P("The methodology adopted is a five-stage pipeline realised as a five-layer " +
    "architecture, implemented as a configuration-driven and tested Python package. Data " +
    "preparation uses a strictly chronological split and applies class weighting to the " +
    "training fold only. The model layer benchmarks five families under one protocol. A " +
    "stacking meta-learner fitted on out-of-fold scores and an isotonic calibrator produce " +
    "a 0-100 risk score with three cost-sensitive operational tiers. SHAP supplies " +
    "per-alert reason codes and the Population Stability Index quantifies drift."),

  H2("5.2  Conclusions"),
  P(`On the held-out final span of ${int(D.split?.test?.rows)} transactions the strongest ` +
    `model reaches an AUC-PR of ${num(best?.auc_pr)} against a no-skill floor of ` +
    `${num(testPrev)}. Isotonic calibration reduces expected calibration error from ` +
    `${num(D.fusion?.ece_before, 4)} to ${num(D.fusion?.ece_after, 4)}, and the resulting ` +
    `tiers place ${pct(highTier.share_of_all_fraud)} of all fraud into a manual-review ` +
    `queue covering ${pct(highTier.share_of_volume)} of volume at ` +
    `${pct(highTier.precision_in_tier)} precision, reducing modelled loss by ` +
    `${num(costCfg.cost_reduction_pct, 1)} per cent.`),
  P("What distinguishes this work from the studies reviewed is not any single model. It is " +
    "the combination of a chronological protocol, an explicit decision layer priced against " +
    "the costs of the two error types, explanation and drift treated as first-class " +
    "outputs, and serving cost measured rather than assumed. The most transferable result " +
    "is the label-delay ablation: approximately a third of the headline performance of a " +
    "history-based fraud model depends on outcomes being available sooner than a real " +
    "chargeback cycle supplies them, and reporting a figure obtained under the immediate " +
    "label assumption overstates deployable performance. None of the surveyed studies " +
    "measures this."),
  P("Two negative results are reported rather than suppressed. The sequential and " +
    "relational tiers did not outperform gradient boosting on this benchmark, which appears " +
    "attributable to the very short per-entity histories the data affords and to the " +
    "explicit graph features already available to the tabular models. And the " +
    "single-transaction serving path, at a median of " + num(single.p50_ms, 0) + " " +
    "milliseconds, does not meet an interactive latency budget, although the batched path " +
    "exceeds the requirement comfortably."),

  H2("5.3  Contribution Summary"),
  P("This project was carried out individually. All components - the data pipeline, the " +
    "feature layer and its causality guarantees, the five model tiers, the fusion and " +
    "calibration layer, the risk and cost model, the explanation and drift monitoring, the " +
    "test suite, the ablation study and this report - were designed and implemented by the " +
    "author under the guidance of Dr. Usha M."),

  H2("5.4  Future Scope"),
  P("The first direction is the serving layer. The scoring path exists and its cost has " +
    "been measured, but it is not yet wrapped in a transport. A scoring endpoint, a " +
    "streaming replay harness that exercises the online velocity cache under realistic " +
    "arrival patterns, and an analyst dashboard that presents the risk score alongside its " +
    "reason codes would turn the pipeline into a demonstrable system. The latency finding " +
    "in section 4.7 gives that work a concrete target: either micro-batching or export to " +
    "a compiled runtime is required before single-request scoring is viable."),
  P("The second direction is robustness. The reviewed literature reports accuracy " +
    "degradation of approximately twenty per cent under projected gradient descent attacks, " +
    "and this project measures concept drift but not adversarial robustness. An evaluation " +
    "in which an attacker perturbs the controllable fields of a transaction - amount, " +
    "timing, device attributes - within realistic bounds, and the degradation of the " +
    "calibrated score is measured, would close gap G5 fully rather than partly. The same " +
    "machinery would support a cost-aware retraining trigger driven by the PSI monitor."),
  P("The third direction follows from the label-delay result. Having established that " +
    "confirmation delay is the dominant sensitivity, the natural next step is to model it " +
    "properly rather than as a fixed offset: real chargebacks arrive on a distribution, " +
    "not at a constant lag, and a system could weight a label by the probability that it " +
    "has been confirmed by scoring time. Extending the study to the Bank Account Fraud " +
    "dataset would establish whether the one-third sensitivity observed here is a property " +
    "of this benchmark or of history-based fraud features in general."),
  Break(),
];

/* ====================================================== BACK MATTER ======= */
const backMatter = [
  H1("REFERENCES"),
  P("Journal and Conference Papers", { bold: true, after: 120 }),
  ...[
    "[1]  A. A. Alhashmi, A. M. Alashjaee, A. A. Darem, A. F. Alanazi and R. Effghi, " +
    "“An ensemble-based fraud detection model for financial transaction cyber threat " +
    "classification and countermeasures,” Engineering, Technology & Applied Science " +
    "Research, vol. 13, no. 6, pp. 12433-12439, 2023.",
    "[2]  C. P. Pandey, H. Upadhyay, A. Kale, P. Joshi, B. S. Katta and R. Kumar, " +
    "“AI-driven fraud detection and risk forecasting framework for real-time financial " +
    "transactions,” Scientific Culture, vol. 12, no. 1.1, pp. 3425-3431, 2026.",
    "[3]  S. Motie and B. Raahemi, “Financial fraud detection using graph neural " +
    "networks: A systematic review,” Expert Systems with Applications, vol. 240, " +
    "art. 122156, 2024.",
    "[4]  N. Hollmann, S. Müller, L. Purucker, A. Krishnakumar, M. Körfer, " +
    "S. B. Hoo, R. T. Schirrmeister and F. Hutter, “Accurate predictions on small data " +
    "with a tabular foundation model,” Nature, vol. 637, no. 8045, pp. 319-326, 2025.",
    "[5]  Q. Sha, T. Tang, X. Du, J. Liu, Y. Wang and Y. Sheng, “Detecting credit card " +
    "fraud via heterogeneous graph neural networks with graph attention,” in " +
    "Proceedings of the IEEE 6th International Seminar on Artificial Intelligence, " +
    "Networking and Information Technology (AINIT), pp. 1332-1336, 2025.",
    "[6]  I. Y. Hafez, A. Y. Hafez, A. Saleh, A. A. Abd El-Mageed and A. A. Abohany, " +
    "“A systematic review of AI-enhanced techniques in credit card fraud " +
    "detection,” Journal of Big Data, vol. 12, art. 6, 2025.",
    "[7]  F. K. Alarfaj and S. Shahzadi, “Enhancing fraud detection in banking with " +
    "deep learning: Graph neural networks and autoencoders for real-time credit card fraud " +
    "prevention,” IEEE Access, vol. 13, pp. 20633-20646, 2024.",
    "[8]  X. Zhao, Y. Liu and Q. Zhao, “Improved LightGBM for extremely imbalanced data " +
    "and application to credit card fraud detection,” IEEE Access, vol. 12, " +
    "pp. 159316-159335, 2024.",
    "[9]  R. K. Gupta, A. Hassan, S. K. Majhi, N. Parveen, A. T. Zamani, R. Anitha, B. Ojha, " +
    "A. K. Singh and D. Muduli, “Enhanced framework for credit card fraud detection " +
    "using robust feature selection and a stacking ensemble model approach,” Results " +
    "in Engineering, vol. 26, art. 105084, 2025.",
    "[10]  S. Jesus, J. Pombal, D. Alves, A. Cruz, P. Saleiro, R. P. Ribeiro, J. Gama and " +
    "P. Bizarro, “Turning the tables: Biased, imbalanced, dynamic tabular datasets for " +
    "ML evaluation,” Advances in Neural Information Processing Systems, vol. 35, " +
    "pp. 33563-33575, 2022.",
    "[11]  S. M. Lundberg and S.-I. Lee, “A unified approach to interpreting model " +
    "predictions,” Advances in Neural Information Processing Systems, vol. 30, " +
    "pp. 4765-4774, 2017.",
    "[12]  T. Chen and C. Guestrin, “XGBoost: A scalable tree boosting system,” in " +
    "Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and " +
    "Data Mining, pp. 785-794, 2016.",
    "[13]  G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye and T.-Y. Liu, " +
    "“LightGBM: A highly efficient gradient boosting decision tree,” Advances in " +
    "Neural Information Processing Systems, vol. 30, pp. 3146-3154, 2017.",
    "[14]  E. Rossi, B. Chamberlain, F. Frasca, D. Eynard, F. Monti and M. Bronstein, " +
    "“Temporal graph networks for deep learning on dynamic graphs,” ICML Workshop " +
    "on Graph Representation Learning, 2020.",
  ].map((t) => P(t, { after: 100 })),
  P("Web References", { bold: true, after: 120 }),
  P("[15]  IEEE-CIS Fraud Detection dataset, IEEE Computational Intelligence Society, " +
    "Kaggle.", { after: 100 }),
  P("[16]  scikit-learn, XGBoost, LightGBM, PyTorch and SHAP official documentation.",
    { after: 100 }),
  Break(),

  H1("PROJECT DETAILS"),
  P("Student Details", { bold: true, after: 120 }),
  T([
    ["Field", "Detail"],
    ["Student Name", "Nandurkar Shreyas Vivek"],
    ["Section / Roll No", "CSE Core B"],
    ["Registration Number", "235805446"],
    ["Semester", "VII"],
    ["Email Address", "nandurkar.mitblr2023@learner.manipal.edu"],
    ["Phone No (M)", "+91 9004491325"],
  ], [3, 7], { numericFrom: 99 }),
  P("Project Details", { bold: true, after: 120 }),
  T([
    ["Field", "Detail"],
    ["Project Title", "Financial Transaction Fraud Detection and Risk Modelling"],
    ["Phase", "Phase-3: Implementation and Result Analysis"],
    ["Project Duration", "Sixth and Seventh Semester"],
    ["Date of Report", "September 2026"],
  ], [3, 7], { numericFrom: 99 }),
  P("Guide Details", { bold: true, after: 120 }),
  T([
    ["Field", "Detail"],
    ["Faculty Name", "Dr. Usha M"],
    ["Designation", "Associate Professor"],
    ["Department", "Computer Science and Engineering"],
    ["Institute", "Manipal Institute of Technology, Bengaluru"],
  ], [3, 7], { numericFrom: 99 }),
];

module.exports = { chapter3, chapter4, chapter5, backMatter };
