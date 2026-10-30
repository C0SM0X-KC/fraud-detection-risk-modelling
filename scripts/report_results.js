/** Chapters 3-5, references and project details; numbers are read from reports/. */

const R = require("./report_frontmatter.js");
const { extensionSections } = require("./report_extensions.js");
const { analyticsEda, analyticsSegments } = require("./report_analytics.js");
const {
  D, P, H1, H2, Bullets, T, Figure, Break, Caption,
  num, int, pct, label, modelByLabel, best, testPrev, tierRow, latRow, abl, worstPsi,
} = R;

const xgb = modelByLabel.xgboost || {};
const bestRow = modelByLabel[best?.label] || xgb;
const stackedCal = modelByLabel["stacked+calibrated"] || {};
const highTier = tierRow(D.tiersConfig, "high");
const medTier = tierRow(D.tiersConfig, "medium");
const lowTier = tierRow(D.tiersConfig, "low");
const costCfg = (D.evaluation && D.evaluation.cost && D.evaluation.cost.config) || {};
const single = latRow("single_row");
const batch1k = latRow("batch_1024");
const selected = D.fusion?.selected_score || "stacked";
const selectedRaw = modelByLabel[selected] || {};
const decision = (D.models || []).find((m) => String(m.label).startsWith("decision layer")) || {};
const members = (D.fusion?.blend_members || []).map(label);
const selectedPhrase = String(selected).startsWith("blend")
  ? `an equal-weight average, in log-odds, of ${members.join(" and ")}`
  : selected === "stacked" ? "the stacked ensemble" : `a single model, ${label(selected)}`;
const faith3 = (D.faith || []).find((r) => String(r.k) === "3") || {};
const groupRows = [
  ["Evidence family", "Features", "Mean |SHAP| (summed)", "Share"],
  ...(D.groups || []).slice(0, 8).map((r) => [r.group, r.features, num(r.mean_abs_shap, 3), pct(r.share, 1)]),
];
const faithRows = [
  ["Features removed (k)", "Mean score before", "Fall, top-k reasons", "Fall, k random features",
    "Ratio", "Alerts falling below half (top-k)"],
  ...(D.faith || []).map((r) => [r.k, num(r.mean_score_before, 1), num(r.mean_score_drop_top, 1),
    num(r.mean_score_drop_random, 2), `${num(r.drop_ratio_top_vs_random, 0)}x`,
    pct(r.share_falling_below_half_top, 1)]),
];
const sampleReason = ((D.reasons || [])[0] || {}).reason_text || "";

/* =================================================== CHAPTER 3 ============ */
const chapter3 = [
  H1("CHAPTER 3  METHODOLOGY"),
  P("This chapter sets out how the system is built, from the workflow and architecture to " +
    "the evaluation metrics and the implementation."),

  H2("3.1  Overall Methodology"),
  P("The system is a five-stage pipeline with a feedback loop (Figure 3.1): ingestion " +
    "validates and orders each transaction; feature engineering derives behavioural, " +
    "temporal and relational features; the model layer scores it with several models; " +
    "fusion and risk modelling combines and calibrates the scores into a risk score and " +
    "tier; and the decision stage acts, records reason codes and emits drift and latency " +
    "telemetry. Confirmed labels, analyst outcomes and drift alerts feed back to drive " +
    "re-tuning and retraining, the response to gap G5."),
  ...Figure("reports/figures/workflow.png",
    "Figure 3.1  Overall workflow of the proposed system", 600),
  P("Viewed as an analytics process, the work follows the CRISP-DM cycle (Cross-Industry " +
    "Standard Process for Data Mining), and each of its phases maps to a part of this report:"),
  ...Bullets([
    "Business understanding: the cost of fraud, the asymmetry of the two error types and " +
    "the operational actions available (Chapter 1 and Section 3.6).",
    "Data understanding: data-quality profiling, exploratory analysis and statistical " +
    "testing of the benchmark (Sections 4.2 and 4.3).",
    "Data preparation: cleaning, encoding, the chronological split and feature engineering " +
    "(Table 3.1, block A).",
    "Modelling: the model families and the decision layer (Sections 3.5 and 3.6, Table 3.1, " +
    "block B).",
    "Evaluation: model comparison, calibration, risk tiers, confidence intervals and the " +
    "segment breakdown (Sections 4.4 to 4.7), followed by explanation, drift and " +
    "sensitivity analyses (Sections 4.8 to 4.12).",
    "Deployment: the online feature store, API, analyst console and streaming transport " +
    "(Section 4.15).",
  ]),

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
    ["A.6  Exploratory analysis",
      "Data-quality profile, distributions of fraud by amount, time and segment, fraud " +
      "concentration across accounts, and chi-square and Mann-Whitney tests with effect " +
      "sizes, all on the training span only."],
    ["B.1  Baselines",
      "Logistic regression and random forest, establishing the performance floor against " +
      "which every later model is judged."],
    ["B.2  Gradient boosting",
      "XGBoost and LightGBM, with early stopping driven by the chronological validation " +
      "span rather than a random holdout."],
    ["B.3  Sequential and relational models",
      "An LSTM over per-entity transaction sequences, and a temporal heterogeneous graph " +
      "neural network with attention over the card-merchant-device graph."],
    ["B.4  Fusion and scoring",
      "A stacking meta-learner over the base model probabilities, followed by isotonic " +
      "calibration and mapping to the 0-100 risk score with tier assignment."],
    ["B.5  Evaluation",
      "AUC-PR, recall at a fixed false-positive rate, F1 and MCC, with day-block bootstrap " +
      "confidence intervals and a breakdown by segment; PSI drift across temporal " +
      "windows; SHAP attribution; and per-transaction inference latency at p50 and p95."],
  ], [2.2, 7.8], { numericFrom: 99 }),
  Caption("Table 3.1  Data preparation and model development steps"),

  H2("3.3  Assumptions Made"),
  ...Bullets([
    "The labels supplied with the benchmark are treated as ground truth, although in " +
    "production fraud labels arrive with delay and a proportion of fraud is never reported. " +
    "Section 4.11 measures the consequence of this assumption rather than leaving it stated.",
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
  P("Viewed as five layers (Figure 3.2): ingestion validates and de-duplicates events; " +
    "the feature layer holds the offline feature store, the online velocity cache and the " +
    "graph builder; the model layer runs the scorers; the decision layer holds fusion, " +
    "calibration, the risk score and the action policy; and the governance layer stores " +
    "reason codes, monitors drift and records latency and alert volume."),
  ...Figure("reports/figures/architecture.png",
    "Figure 3.2  Layered architecture of the proposed system", 600),

  H2("3.5  Model Stack and Rationale"),
  P("The baseline tiers are retained deliberately: a comparison without a floor cannot " +
    "demonstrate that the additional complexity of the later tiers is justified."),
  T([
    ["Tier", "Model", "What it captures", "Why it is included"],
    ["Baseline", "Logistic regression, random forest", "Linear and simple non-linear effects",
      "Establishes the performance floor and an interpretable reference point"],
    ["Tabular", "XGBoost, LightGBM", "Higher-order interactions among engineered features",
      "The strongest single family on tabular fraud data, fast to train and to serve"],
    ["Sequential", "LSTM / temporal convolutional network",
      "Order and timing of an entity's recent behaviour",
      "Fraud is a break in a sequence rather than a property of an isolated row"],
    ["Relational", "Temporal heterogeneous GNN with attention",
      "Devices, merchants and addresses shared across accounts",
      "Exposes organised fraud rings that row-wise models cannot see; addresses gap G2"],
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
  P("Thresholds on an uncalibrated score correspond to no real business cost, so " +
    "calibration is what makes decisioning possible; this is the response to gap G3."),

  H2("3.7  Dataset Specification"),
  T([
    ["Property", "IEEE-CIS (primary)", "Bank Account Fraud (secondary)"],
    ["Transactions", "590,540", "1,000,000"],
    ["Features", "434 after merging transaction and identity", "30"],
    ["Fraud prevalence", "3.50 per cent", "Approximately 1 per cent, variant-dependent"],
    ["Observation span", "182 days, chronologically ordered", "8 months, chronologically ordered"],
    ["Fraud type", "Card-not-present e-commerce fraud", "Bank account opening fraud"],
    ["Role in the project", "Main benchmark for all model comparisons",
      "Transfer and fairness check (section 4.14)"],
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
  P("The system is a configuration-driven Python package: every threshold, window and " +
    "hyper-parameter is declared in one YAML file, and each stage is a separate command " +
    "that can be re-run in isolation. One rule underpins every result in Chapter 4: no " +
    "aggregate may use a transaction later than the one being scored, verified by a test " +
    "that rewrites the later half of a dataset and checks that no earlier feature changes."),
  T([
    ["Category", "Tools and libraries"],
    ["Language and data handling", "Python 3.12; pandas, NumPy, PyArrow"],
    ["Classical and boosting models",
      "scikit-learn; XGBoost; LightGBM; CatBoost; imbalanced-learn for SMOTE and ADASYN; " +
      "Optuna for Bayesian hyper-parameter search"],
    ["Deep and graph models",
      "PyTorch on a cloud GPU (Kaggle, NVIDIA T4); a custom temporal heterogeneous GNN"],
    ["Data analysis and statistics",
      "SciPy for chi-square and Mann-Whitney tests; bootstrap confidence intervals; an " +
      "interactive analytics page in the console built with Altair"],
    ["Explainability and monitoring",
      "SHAP (exact TreeSHAP) with plain-language reason codes; a custom PSI implementation " +
      "for offline and live drift"],
    ["Serving and demonstration",
      "FastAPI scoring endpoint; Streamlit analyst console; Kafka (or an in-process queue) " +
      "as the streaming transport; MLflow for optional experiment tracking"],
    ["Testing and reproducibility",
      "pytest; Docker for a reproducible build; per-model score checkpointing; Git"],
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


const chapter4 = [
  H1("CHAPTER 4  RESULT ANALYSIS"),
  P("This chapter follows the four levels of analytics introduced in Section 1.1. After the " +
    "setup (Section 4.1) it describes the data and tests which attributes are associated " +
    "with fraud (Sections 4.2 and 4.3), compares the predictive models (Section 4.4), " +
    "evaluates the calibrated score and the prescriptive decision policy (Sections 4.5 and " +
    "4.6), and measures how reliable the results are and where the system is weak (Section " +
    "4.7). Explanation, drift, serving cost, the labelling assumption, further experiments, " +
    "robustness, transfer, the deployed product and wider impact follow (Sections 4.8 to " +
    "4.16), with conclusions in Section 4.17."),

  H2("4.1  Experimental Setup"),
  P("Unless stated otherwise, every model result in this chapter is computed on the final " +
    "chronological span of the " +
    "benchmark, which no model saw during training and which the calibrator was not fitted " +
    "on. The split is made on the time value rather than on row position, so every " +
    "transaction sharing a timestamp falls on the same side of a boundary. The exploratory " +
    "and statistical analysis in Sections 4.2 and 4.3 uses the training span only."),
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
    ["Entity aggregates", String(D.spec?.groups?.entity_aggregates ?? "-"),
      "Per-account, card and device running means of amount and of C, D and V fields, and this row's ratio to them"],
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

  ...analyticsEda,

  H2("4.4  Model Comparison"),
  P("Table 4.7 reports every model under the identical protocol required by objective 3. " +
    "The no-skill AUC-PR on this span equals the prevalence, " + num(testPrev) + ", so that " +
    "is the floor against which each row should be read."),
  T(comparisonRows, [2.6, 1.3, 1.3, 1.2, 1.2, 1.6, 1.6]),
  Caption("Table 4.7  Model comparison on the held-out test span"),
  ...Figure("reports/figures/pr_curves.png",
    "Figure 4.8  Precision-recall curves on the test span"),
  P(`The strongest model attains an AUC-PR of ${num(best?.auc_pr)}, which is ` +
    `${num(bestRow.auc_pr_lift_over_prevalence, 1)} times the no-skill floor. The same model ` +
    `attains an accuracy of ${num(bestRow.accuracy_not_a_headline_metric, 4)}, against ` +
    `${num(bestRow.accuracy_of_always_legitimate, 4)} for a model that declares every ` +
    `transaction legitimate. The two accuracies differ by about two percentage points while ` +
    `the AUC-PRs differ by a factor of ${num(bestRow.auc_pr_lift_over_prevalence, 0)}. This ` +
    `reproduces the argument of Chapter 2 on the project's own results, and it is why ` +
    `accuracy is not used to compare models.`),
  P("The tabular tier ranks well above the baselines. Stacking, identified in Chapter 2 as " +
    "the strongest fusion strategy, is outperformed here by a simpler alternative: the base " +
    "models make highly correlated errors, leaving little for a meta-learner to exploit, and " +
    "the meta-learner is trained on fold-model scores but applied to final-model scores. The " +
    "decision layer therefore chooses on the calibration holdout between the stack, each " +
    "base score and equal-weight log-odds averages of the strongest tabular models, and " +
    `selects ${selectedPhrase} (validation AUC-PR ` +
    `${num(D.fusion?.validation_ranking?.[selected])} against the stack's ` +
    `${num(D.fusion?.validation_ranking?.stacked)}). The sequential and relational tiers ` +
    "score well below the tabular tier; Section 4.12 analyses why."),

  H2("4.5  Effect of Calibration"),
  P("Calibration converts a ranking into a probability that a threshold can be set on. The " +
    "validation span is cut in two: the earlier portion joins the training data for the " +
    "final refit, and the later portion is held back, scored by the refitted model and used " +
    "to fit the isotonic calibrator, so the calibrator sees the same model it will be " +
    "applied to. In a preliminary run, a calibrator fitted before the refit increased " +
    "expected calibration error six-fold."),
  T([
    ["Quantity", "Before calibration", "After calibration"],
    ["Expected calibration error", num(D.fusion?.ece_before, 5), num(D.fusion?.ece_after, 5)],
    ["AUC-PR", num(selectedRaw.auc_pr), num(decision.auc_pr)],
    ["Brier score", num(selectedRaw.brier, 5), num(decision.brier, 5)],
  ], [3.4, 3.3, 3.3]),
  Caption("Table 4.8  Effect of isotonic calibration"),
  ...Figure("reports/figures/reliability.png",
    "Figure 4.9  Reliability of the calibrated risk score"),
  P("Expected calibration error falls by roughly two orders of magnitude and the Brier " +
    "score improves. AUC-PR falls very slightly because isotonic regression maps ranges of " +
    "raw scores to one value, creating ties; the trade is worth making, since only a " +
    "threshold on the calibrated score corresponds to an expected loss."),

  H2("4.6  Risk Tiers and Operational Cost"),
  P("The calibrated probability is mapped to the 0-100 risk score and assigned to the tiers " +
    "specified in Table 3.3."),
  T(tierRows, [1, 1.8, 1.5, 1.5, 1.3, 1.5, 1.2]),
  Caption("Table 4.9  Risk-tier performance under the specified score bands"),
  ...Figure("reports/figures/score_distribution.png",
    "Figure 4.10  Risk-score distribution and tier boundaries"),
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
  Caption("Table 4.10  Modelled cost of the decision policy"),
  P("Cost figures use the configured ratio in which a missed fraud costs the transaction " +
    "amount and a false decline costs a fixed review-and-goodwill charge; the absolute values " +
    "are institution-specific. The policy removes roughly two-thirds of the loss that taking " +
    "no action would incur."),

  ...analyticsSegments,

  H2("4.8  Explainability"),
  P("Every transaction in the high-risk tier carries the features that drove its score, so " +
    "that an adverse decision can be explained and audited. Attribution is taken from the " +
    "score that makes the decision" +
    (String(selected).startsWith("blend")
      ? ", an equal-weight log-odds average, so it is exactly the average of the members' " +
        "TreeSHAP values. "
      : ". ") +
    "On every run the base value plus the contributions is checked to reconstruct the " +
    "model's output to within 1e-5. The contributions are in log-odds before calibration, so " +
    "they preserve each reason's direction but not the scale of the displayed risk score."),
  ...Figure("reports/figures/shap_importance.png",
    "Figure 4.12  Global SHAP attribution over high-risk alerts"),
  P("The ranking is dominated by the causal prior-fraud-rate features at account and card " +
    "level, followed by the per-entity aggregate ratios, the Vesta engineered columns and the " +
    "counting features. That the account-level prior fraud rate dominates matches Section " +
    "4.2 - fraud repeats on accounts already used for fraud - but it is also a dependency " +
    "on a labelling assumption, which section 4.11 quantifies."),

  P("Two additions make the explanations usable and checkable. First, every attribution is " +
    "rendered as a sentence about the transaction rather than a feature name - the first " +
    `alert in the test span reads: "${sampleReason}". Features whose meaning the data ` +
    "provider masked are labelled as masked rather than given an invented interpretation. " +
    "Second, attributions are summed within evidence families, so that correlated columns " +
    "are read as one piece of evidence rather than many."),
  T(groupRows, [4.2, 1.2, 2.2, 1.4]),
  Caption("Table 4.13  Attribution by evidence family over high-risk alerts"),
  P("The explanations were tested with a deletion test: for each alert the k features with " +
    "the largest positive attribution are reset to their median over ordinary traffic and " +
    "the transaction is re-scored, against k features chosen at random."),
  T(faithRows, [1.6, 1.5, 1.6, 1.7, 1, 2]),
  Caption("Table 4.14  Explanation faithfulness: deletion test on high-risk alerts"),
  P(`Removing the top three stated reasons of an alert lowers its risk score by ` +
    `${num(faith3.mean_score_drop_top, 1)} points on average, against ` +
    `${num(faith3.mean_score_drop_random, 1)} for three random features, and takes ` +
    `${pct(faith3.share_falling_below_half_top, 0)} of alerts below half their score. The ` +
    "stated reasons therefore identify the evidence on which each decision rests."),

  H2("4.9  Concept Drift"),
  P("Population Stability Index (PSI) is computed on the model's score distribution across " +
    "equal-duration windows of the test timeline, with bin edges fixed from the first window."),
  ...Figure("reports/figures/psi_drift.png",
    "Figure 4.13  Score drift across the test timeline"),
  P(`The worst observed PSI is ${num(worstPsi, 4)}, far below the conventional 0.10 watch ` +
    "threshold and the 0.25 retraining threshold, so the score distribution is stable over " +
    "the six weeks of the test span - a short horizon, which shows no degradation right " +
    "after training rather than indefinite stability. Fraud prevalence itself moves from " +
    "3.4 to 4.2 per cent across the windows. The analyst console computes the same PSI live " +
    "and raises a retraining alert above 0.25."),

  H2("4.10  Inference Latency"),
  P("Table 4.15 times every enabled tier together - all base models, the meta-learner and " +
    "the calibrator - on precomputed feature rows, an upper bound on the cost of the full " +
    "benchmark stack. Section 4.15 times the path a deployed request actually takes."),
  T(latencyRows, [3, 1.8, 1.8, 1.8, 1.6]),
  Caption("Table 4.15  Inference latency of the complete benchmark stack"),
  P(`The batched path costs ${num(batch1k.mean_ms, 3)} ms per transaction, equivalent to ` +
    `roughly ${int(Math.round(1000 / Number(batch1k.mean_ms || 1)))} transactions per ` +
    `second. One transaction at a time through the full stack takes ${num(single.p50_ms, 1)} ` +
    `ms at the median and ${num(single.p95_ms, 1)} ms at the 95th percentile, owing to ` +
    `per-call overhead in the Python stack rather than model complexity. This shaped the ` +
    `serving design: the product path of section 4.15 serves only the selected score through ` +
    `the boosters' native single-row interface, and meets the requirement.`),

  H2("4.11  Sensitivity to the Labelling Assumption"),
  P("Section 3.3 assumes that the outcome of an earlier transaction is known when a later " +
    "one is scored. In production it is not: a chargeback takes weeks to confirm. Since " +
    "the prior-fraud-rate features dominate the model (Figure 4.12), a controlled ablation " +
    "withholds each label from those features until a fixed delay after its transaction, " +
    "with everything else held fixed."),
  T(ablationRows, [2.6, 2, 1.5, 2, 1.5]),
  Caption("Table 4.16  Label-delay ablation"),
  P(`Introducing a thirty-day confirmation delay reduces AUC-PR from ${num(abl("delay_0", "xgboost").auc_pr)} ` +
    `to ${num(abl("delay_30", "xgboost").auc_pr)}, retaining ` +
    `${abl("delay_30", "xgboost").auc_pr_retained_pct} per cent of the original performance. ` +
    `Extending the delay to sixty days costs comparatively little further ` +
    `(${num(abl("delay_60", "xgboost").auc_pr)}), and removing the features entirely gives ` +
    `${num(abl("none", "xgboost").auc_pr)}. Both models move the same way, so this is a ` +
    `property of the feature rather than of one learner.`),
  P(`Three conclusions follow. About ` +
    `${num(100 - Number(abl("delay_30", "xgboost").auc_pr_retained_pct), 0)} per cent of the ` +
    "headline performance depends on outcomes being known sooner than a real chargeback " +
    "cycle would supply them. The sensitivity is concentrated in the first month. And the features remain " +
    `worthwhile: at a sixty-day delay they are still worth ` +
    `${num(abl("delay_60", "xgboost").auc_pr - abl("none", "xgboost").auc_pr, 3)} AUC-PR ` +
    "over removing them, so they should be built with a delay matched to the institution's " +
    "actual confirmation window."),
  P("Published results on this benchmark cluster around 0.94 to 0.96 AUC-ROC; the " +
    "immediate-label figure sits above that range and the thirty-day figure just below it. " +
    "The delayed measurement corresponds to deployable performance, and this report adopts " +
    "it as the headline figure for deployment."),

  ...extensionSections,

  H2("4.16  Environmental and Societal Impact"),
  P("Fraud losses are borne by customers through pricing and by merchants through " +
    "chargebacks, so removing a share of them at a controlled false-decline rate returns " +
    "value to both. A declined legitimate transaction is a customer refused service, and the " +
    "tiered design meets uncertainty with an authentication challenge rather than a refusal. " +
    "A model keyed on entity history may act on an account's past rather than the " +
    "transaction in front of it, and the reason codes make such a decision visible and " +
    "contestable. IEEE-CIS carries no protected attributes; on Bank Account Fraud, section " +
    "4.14 found a threefold age disparity in false-positive rate that must be corrected " +
    "before deployment."),
  P("The environmental cost is modest: the tabular tiers train in about an hour on a " +
    "sixteen-core laptop CPU and the GPU tiers in under fifteen minutes on one cloud GPU, " +
    "helped by column pruning, per-model checkpointing and explicit graph features in place " +
    "of repeated graph training."),

  H2("4.17  Conclusions"),
  P("The system meets the objectives it was specified against. It detects fraud on a " +
    `strictly chronological split at ${num(best?.auc_pr_lift_over_prevalence, 1)} times the ` +
    "no-skill floor; produces a calibrated risk score whose high tier isolates " +
    `${pct(highTier.share_of_all_fraud, 0)} of all fraud into ` +
    `${pct(highTier.share_of_volume, 1)} of volume; attaches a faithful, plain-language ` +
    "reason to every alert; and reports drift, serving cost, adversarial robustness and " +
    "transfer to a second dataset as measurements rather than assertions. The exploratory " +
    "and statistical analysis shows why a model is needed: no single attribute has more " +
    "than a small association with fraud, and more than half of fraud repeats on an " +
    "account already used for fraud. The segment analysis locates the main weakness, " +
    "accounts seen for the first time (section 4.7). It is delivered " +
    "as a product: a single raw transaction is decided, with its features computed online, " +
    "inside the latency requirement."),
  P("Several findings refine the expectations set by the literature. On this benchmark the " +
    "sequential and relational tiers add no signal beyond the tabular tier (section 4.12), " +
    "and stacking, hyper-parameter tuning, the account profile and a third boosting model " +
    "do not improve on a simple two-model average under validation-based selection " +
    "(sections 4.4 and 4.12). On Bank Account Fraud the method transfers, while the " +
    "score bands must be derived for each deployment and the age disparity in " +
    "false-positive rate must be corrected before use (section 4.14). The label-delay ablation in section 4.11 " +
    "remains the finding with the widest implication, since it applies to any fraud system " +
    "built on entity history and is not reported by any of the studies surveyed in " +
    "Chapter 2."),
  Break(),
];

/* =================================================== CHAPTER 5 ============ */
const firstSeen = (D.segments || []).find(
  (r) => r.dimension === "Account history" && r.segment === "first seen") || {};
const chapter5 = [
  H1("CHAPTER 5  CONCLUSION AND FUTURE SCOPE"),

  H2("5.1  Summary of the Work"),
  P("The problem addressed is the detection of fraudulent payment transactions in real " +
    "time, together with the conversion of the resulting model output into a calibrated and " +
    "actionable measure of risk. The problem is characterised by extreme class imbalance, " +
    "asymmetric error costs, non-stationarity arising from adversarial adaptation, and a " +
    "requirement that decisions be explainable."),
  P("A review of seven studies published between 2023 and 2026 exposed six gaps: " +
    "evaluation protocols that ignore time, tabular-only modelling, detection treated as the " +
    "end point rather than risk quantification, opaque decisions, unmeasured drift and " +
    "adversarial decay, and unreported cost of serving. These became six verifiable " +
    "objectives, and a seventh covers the analysis of the data and of the results."),
  P("The methodology follows the CRISP-DM cycle and is realised as a five-stage pipeline " +
    "and a five-layer " +
    "architecture in a tested Python package, with a strictly chronological split, four " +
    "model families under one protocol, a decision layer selecting on validation, and an " +
    "isotonic calibrator producing a 0-100 risk score with three cost-sensitive tiers. SHAP " +
    "supplies plain-language reason codes and PSI quantifies drift. Exploratory data " +
    "analysis, chi-square and Mann-Whitney tests with effect sizes, day-block bootstrap " +
    "confidence intervals and a segment-level breakdown of performance describe the data " +
    "and qualify the results. An online feature " +
    "store turns the pipeline into a product that scores one raw transaction at a time, " +
    "exposed through an API, an analyst console and a streaming transport."),

  H2("5.2  Conclusions"),
  P(`On the held-out final span of ${int(D.split?.test?.rows)} transactions the strongest ` +
    `model reaches an AUC-PR of ${num(best?.auc_pr)} against a no-skill floor of ` +
    `${num(testPrev)}. Isotonic calibration reduces expected calibration error from ` +
    `${num(D.fusion?.ece_before, 4)} to ${num(D.fusion?.ece_after, 4)}, and the resulting ` +
    `tiers place ${pct(highTier.share_of_all_fraud)} of all fraud into a manual-review ` +
    `queue covering ${pct(highTier.share_of_volume)} of volume at ` +
    `${pct(highTier.precision_in_tier)} precision, reducing modelled loss by ` +
    `${num(costCfg.cost_reduction_pct, 1)} per cent.`),
  P("The data analysis explains and qualifies these figures. No single attribute is more " +
    "than weakly associated with fraud, so detection depends on combining many signals, " +
    `and ${pct(D.analytics?.concentration?.fraud_on_account_with_earlier_fraud, 1)} of ` +
    "fraud repeats on an account already used for fraud, which is why account history is " +
    "the strongest evidence the model uses. The headline AUC-PR carries a 95 per cent " +
    `confidence interval of ${num(D.bootstrap?.[0]?.ci_low, 3)} to ` +
    `${num(D.bootstrap?.[0]?.ci_high, 3)}. The main limitation is located precisely: on ` +
    "accounts seen for the first time, " +
    `${pct(firstSeen.share_of_volume, 0)} of test volume, the system catches ` +
    `${pct(firstSeen.fraud_caught_flagged, 0)} of fraud, against 90 per cent or more once ` +
    "an account has any history."),
  P("What distinguishes this work from the studies reviewed is not any single model. It is " +
    "the combination of a chronological protocol, an explicit decision layer priced against " +
    "the costs of the two error types, explanation and drift treated as first-class " +
    "outputs, and serving cost measured rather than assumed. The most transferable result " +
    "is the label-delay ablation: about " +
    `${num(100 - Number(abl("delay_30", "xgboost").auc_pr_retained_pct), 0)} per cent of the ` +
    "headline performance of a history-based fraud model depends on outcomes being " +
    "available sooner than a real chargeback cycle supplies them, and reporting a figure " +
    "obtained under the immediate label assumption overstates deployable performance. Under " +
    `a thirty-day delay this system still reaches ${num(abl("delay_30", "xgboost").auc_pr)} ` +
    "AUC-PR, and this report adopts that figure as its headline for deployable performance. " +
    "None of the surveyed studies measures this."),
  P("The sequential and relational tiers did not outperform gradient boosting, owing to " +
    "short per-entity histories and period-specific fitting, and tuning, extra features and " +
    "extra learners changed validation scores by less than the run-to-run variation. On " +
    "Bank Account Fraud the method transfers but the fixed tiers do not, and an age " +
    "disparity in false-positive rate must be corrected before deployment."),

  H2("5.3  Contribution Summary"),
  P("The individual contribution of Nandurkar Shreyas Vivek (235805446) to the project " +
    "comprises the following:"),
  ...Bullets([
    "Data analytics (Sections 4.2, 4.3 and 4.7): the data-quality profile, exploratory " +
    "analysis of fraud by amount, time and segment, the concentration analysis, chi-square " +
    "and Mann-Whitney tests with effect sizes, day-block bootstrap confidence intervals, " +
    "the segment-level performance breakdown, and the Fraud analytics page of the console.",
    "Corrections to the feature layer and evaluation: running means that skip missing " +
    "values (which raised the thirty-day-delay result from 0.573 to " +
    `${num(abl("delay_30", "xgboost").auc_pr, 3)} AUC-PR), a fraud-rate prior that had ` +
    "used test-span prevalence, and the report's table parsing.",
    "The decision layer's equal-weight log-odds blend, now the selected score (Section 4.4).",
    "The deployed product (Section 4.15): the online feature store reproducing every " +
    "offline feature, the single-transaction scoring path, the reworked API, the " +
    "check-a-transaction, live-ledger and monitoring views, plain-language reason codes " +
    "with the deletion test of their faithfulness (Section 4.8), the feedback loop and the " +
    "streaming transport.",
    "The further experiments and checks: Optuna tuning, the account profile, CatBoost and " +
    "the GPU grid for the deep models (Section 4.12), the adversarial robustness study " +
    "(Section 4.13), and transfer and fairness on Bank Account Fraud (Section 4.14).",
    "Reproducibility (Docker, GPU jobs on Kaggle), the automated tests for the new " +
    "components, the data-analytics framing of the report, and this report.",
  ]),

  H2("5.4  Future Scope"),
  P("The first direction follows from the label-delay result. Having established that " +
    "confirmation delay is the dominant sensitivity, the natural next step is to model it " +
    "properly rather than as a fixed offset: real chargebacks arrive on a distribution, not " +
    "at a constant lag, and a system could weight a label by the probability that it has " +
    "been confirmed by scoring time, and train under the same delay it will be deployed " +
    "with."),
  P("The second direction is fairness and robustness, building on two issues this work " +
    "measured. The age disparity found on Bank Account Fraud calls for group-aware " +
    "thresholds or a fairness-constrained training objective, evaluated on the same " +
    "chronological protocol. The history-conditioning weakness found by probing the live " +
    "service - a single prior transaction makes an account look established - calls for a " +
    "minimum-history rule and for near-duplicate transactions to count as a velocity " +
    "signal; a problem-space attack that edits raw transactions through the online store, " +
    "rather than features, would measure such weaknesses directly."),
  P("The third direction is operational. The serving path meets the latency requirement on " +
    "one process; a production deployment would add horizontal scaling behind the streaming " +
    "transport, persistence for the online feature store, and a retraining pipeline driven " +
    "by the live PSI monitor and by confirmed outcomes from the analyst feedback loop. The " +
    "tier boundaries, which section 4.14 showed do not transfer between operating points, " +
    "would be derived per deployment from its cost model and review capacity."),
  P("A fourth direction follows from the segment analysis of Section 4.7. Accounts seen for " +
    "the first time are where detection is weakest, so a dedicated model or rule set for " +
    "them, built only on signals that need no history - device, e-mail domain, address " +
    "distance and velocity across the card - would target the largest remaining gap."),
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
    "[4]  Q. Sha, T. Tang, X. Du, J. Liu, Y. Wang and Y. Sheng, “Detecting credit card " +
    "fraud via heterogeneous graph neural networks with graph attention,” in " +
    "Proceedings of the IEEE 6th International Seminar on Artificial Intelligence, " +
    "Networking and Information Technology (AINIT), pp. 1332-1336, 2025.",
    "[5]  I. Y. Hafez, A. Y. Hafez, A. Saleh, A. A. Abd El-Mageed and A. A. Abohany, " +
    "“A systematic review of AI-enhanced techniques in credit card fraud " +
    "detection,” Journal of Big Data, vol. 12, art. 6, 2025.",
    "[6]  F. K. Alarfaj and S. Shahzadi, “Enhancing fraud detection in banking with " +
    "deep learning: Graph neural networks and autoencoders for real-time credit card fraud " +
    "prevention,” IEEE Access, vol. 13, pp. 20633-20646, 2025.",
    "[7]  X. Zhao, Y. Liu and Q. Zhao, “Improved LightGBM for extremely imbalanced data " +
    "and application to credit card fraud detection,” IEEE Access, vol. 12, " +
    "pp. 159316-159335, 2024.",
    "[8]  R. K. Gupta, A. Hassan, S. K. Majhi, N. Parveen, A. T. Zamani, R. Anitha, B. Ojha, " +
    "A. K. Singh and D. Muduli, “Enhanced framework for credit card fraud detection " +
    "using robust feature selection and a stacking ensemble model approach,” Results " +
    "in Engineering, vol. 26, art. 105084, 2025.",
    "[9]  S. Jesus, J. Pombal, D. Alves, A. Cruz, P. Saleiro, R. P. Ribeiro, J. Gama and " +
    "P. Bizarro, “Turning the tables: Biased, imbalanced, dynamic tabular datasets for " +
    "ML evaluation,” Advances in Neural Information Processing Systems, vol. 35, " +
    "pp. 33563-33575, 2022.",
    "[10]  S. M. Lundberg and S.-I. Lee, “A unified approach to interpreting model " +
    "predictions,” Advances in Neural Information Processing Systems, vol. 30, " +
    "pp. 4765-4774, 2017.",
    "[11]  T. Chen and C. Guestrin, “XGBoost: A scalable tree boosting system,” in " +
    "Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and " +
    "Data Mining, pp. 785-794, 2016.",
    "[12]  G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye and T.-Y. Liu, " +
    "“LightGBM: A highly efficient gradient boosting decision tree,” Advances in " +
    "Neural Information Processing Systems, vol. 30, pp. 3146-3154, 2017.",
    "[13]  E. Rossi, B. Chamberlain, F. Frasca, D. Eynard, F. Monti and M. Bronstein, " +
    "“Temporal graph networks for deep learning on dynamic graphs,” ICML Workshop " +
    "on Graph Representation Learning, 2020.",
  ].map((t) => P(t, { after: 100 })),
  P("Web References", { bold: true, after: 120 }),
  P("[14]  IEEE-CIS Fraud Detection dataset, IEEE Computational Intelligence Society, " +
    "Kaggle.", { after: 100 }),
  P("[15]  scikit-learn, XGBoost, LightGBM, PyTorch and SHAP official documentation.",
    { after: 100 }),
  Break(),

  H1("PROJECT DETAILS"),
  P("Student Details", { bold: true, after: 120 }),
  T([
    ["Field", "Detail"],
    ["Student Name", "Nandurkar Shreyas Vivek"],
    ["Registration Number", "235805446"],
    ["Semester", "VII"],
    ["Group Members", "Mayank (235805360); Kabir Chhabra (235805300)"],
    ["Email Address", "nandurkar.mitblr2023@learner.manipal.edu"],
    ["Phone No (M)", "+91 9004491325"],
  ], [3, 7], { numericFrom: 99 }),
  P("Project Details", { bold: true, after: 120 }),
  T([
    ["Field", "Detail"],
    ["Project Title", "Financial Transaction Fraud Detection and Risk Modelling"],
    ["Phase", "Phase-3: Implementation and Result Analysis"],
    ["Project Duration", "Sixth and Seventh Semester"],
    ["Date of Report", "October 2026"],
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
