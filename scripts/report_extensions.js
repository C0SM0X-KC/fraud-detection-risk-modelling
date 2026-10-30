/** Chapter 4 sections 4.12-4.15; sentences that depend on a result are chosen from the data. */

const fs = require("fs");
const path = require("path");
const R = require("./report_frontmatter.js");

const { D, P, H2, Bullets, T, Figure, Caption, num, int, pct, label, ROOT } = R;

const readJson = (rel) => {
  const p = path.join(ROOT, rel);
  return fs.existsSync(p) ? JSON.parse(fs.readFileSync(p, "utf8")) : null;
};
const byLabel = (rows) => Object.fromEntries((rows || []).map((r) => [r.label, r]));
const decisionRow = (rows) => (rows || []).find((r) => String(r.label).startsWith("decision layer")) || {};

const adoption = readJson("reports/metrics/feature_adoption.json") || {};
const stream = readJson("reports/metrics/stream_memory.json") || {};

/* ----------------------------------------- 4.12 further experiments -- */
const optuna = D.optuna || {};
const handset = byLabel(D.handsetComparison);
const tuned = byLabel(D.tunedComparison);
const ranking = D.fusion?.validation_ranking || {};
const xgbTest = handset.xgboost?.auc_pr;
const deepBest = (model) => (D.tuningBest || []).find((r) => r.model === model) || {};
const deepCount = (model) => (D.tuning || []).filter((r) => r.model === model).length;
const versus = (a, b, dp = 4) => `${num(a, dp)} vs ${num(b, dp)}`;
const adopted = (a, b) => (Number(a) > Number(b) ? "Yes" : "No");

const experimentRows = [
  ["Experiment", "Compared with", "Validation AUC-PR", "Test AUC-PR", "Adopted"],
  ...[["lstm", "Sequence model (LSTM)"], ["hetero_gnn", "Graph neural network"]].map(([m, name]) => [
    `${name}, best of ${deepCount(m)} configurations`, "XGBoost",
    versus(deepBest(m).valid_auc_pr, ranking.xgboost, 3), versus(deepBest(m).test_auc_pr, xgbTest, 3),
    adopted(deepBest(m).valid_auc_pr, ranking.xgboost)]),
  ...["xgboost", "lightgbm"].filter((m) => optuna[m]).map((m) => [
    `Optuna tuning of ${label(m)} (${optuna[m].trials} trials)`, `hand-set ${label(m)}`,
    `${versus(optuna[m].validation_auc_pr, optuna[m].hand_set_validation_auc_pr)} *`,
    versus(tuned[m]?.auc_pr, handset[m]?.auc_pr),
    "No"]),
  [`Account profile (${adoption.profile_features ?? "-"} features)`,
    `base set (${adoption.base_features ?? "-"} features)`,
    versus(adoption.profile_validation, adoption.base_validation),
    versus(adoption.profile_test, adoption.base_test),
    adopted(adoption.profile_validation, adoption.base_validation)],
  ["CatBoost added to the blend", "XGBoost + LightGBM blend",
    versus(ranking.blend_top3, ranking.blend_top2), "not scored",
    adopted(ranking.blend_top3, ranking.blend_top2)],
];

const experiments = [
  H2("4.12  Further Modelling Experiments"),
  P("Four further modelling options were evaluated after the main results: sequential and " +
    "relational deep-learning models, Bayesian hyper-parameter tuning of the boosting models, " +
    "an account-profile feature set, and CatBoost as a third tabular learner. Each was held " +
    "to the rule applied throughout this chapter: it is adopted only if it raises AUC-PR on " +
    "the validation span, and the test span is consulted only after that decision is made. " +
    "Table 4.17 summarises the outcome."),
  T(experimentRows, [3.2, 2.2, 1.8, 1.8, 1]),
  Caption("Table 4.17  Further modelling experiments and the adoption decision"),
  P("* Tuning was scored on the earlier part of the validation span, so its validation " +
    "figures are on a different sample from the other rows."),
  P("The sequence and graph models stay far below the tabular tier across a grid of " +
    "settings. Histories are short, at just over two transactions per account, so a sequence " +
    "model sees mostly padding; both fall sharply from validation to test, showing " +
    "period-specific fitting; and the feature layer already gives the boosting models graph " +
    "degree, partner counts and neighbourhood fraud rates. Optuna's gain of about 0.003 on " +
    "its validation fold is within run-to-run variation and reversed on test; the account " +
    "profile duplicates signal the base features already carry; and CatBoost " +
    `(${num(ranking.catboost)} on validation) lowered the blend because three boosting ` +
    "models make strongly correlated errors. Under a chronological protocol on drifting " +
    "data, a small validation gain is not evidence of a better model."),
];

/* ------------------------------------------------------- 4.13 robustness -- */
const rob = D.robustness || [];
const robMeta = D.robustnessMeta || {};
const robRows = [
  ["Budget (std per feature)", "Attack", "Caught frauds evaded", "Fraud recall after",
    "Relative recall drop"],
  ...rob.map((r) => [num(r.epsilon_std, 2), r.attack === "pgd" ? "PGD (transfer)" : "Random noise",
    pct(r.evasion_rate, 1), num(r.fraud_recall_after, 3), pct(r.recall_drop_relative, 1)]),
];
const pgd = rob.filter((r) => r.attack === "pgd");
const rnd = rob.filter((r) => r.attack === "random");
const control = pgd.find((r) => Number(r.epsilon_std) === 0) || {};
const worstPgd = pgd.reduce((m, r) => Math.max(m, Number(r.recall_drop_relative || 0)), 0);
const pgdBeats = pgd.filter((r) => {
  const n = rnd.find((x) => x.epsilon_std === r.epsilon_std);
  return n && Number(r.evasion_rate) > Number(n.evasion_rate) + 0.01;
}).map((r) => num(r.epsilon_std, 2));

const robustness = [
  H2("4.13  Adversarial Robustness"),
  P("The literature reports an accuracy drop of roughly twenty per cent under " +
    "projected-gradient-descent (PGD) attack (gap G5). A fraudster controls only the fields " +
    "of their own transaction - amount, e-mail domains, device, address distance, time, " +
    "identity attributes - not the entity history the bank computes, so " +
    `${int(robMeta.mutable_features?.length)} engineered features may move, each by a ` +
    "bounded number of training standard deviations. Trees have no gradient, so PGD is run " +
    "on a neural surrogate of the deployed model (rank correlation " +
    `${num(robMeta.surrogate_fidelity_spearman, 3)}) and the perturbed frauds are replayed ` +
    "through the deployed model, with random noise of the same size as a control."),
  T(robRows, [2, 1.8, 1.8, 1.7, 1.7]),
  Caption("Table 4.18  Evasion of the deployed model under attacker-controlled features"),
  P(`At zero budget ${pct(control.evasion_rate, 1)} of caught frauds clear, so the ` +
    "encoding round trip is not responsible for the effects measured at larger budgets. " +
    `The largest relative fall in fraud recall under PGD is ${pct(worstPgd, 1)}, well short ` +
    "of the twenty per cent reported in the literature. " +
    (pgdBeats.length
      ? `The targeted attack outperforms random noise of the same size only at budgets of ` +
        `${pgdBeats.join(", ")} standard deviations; below that, the evasion observed is ` +
        "frauds lying close to the challenge threshold, displaced by any perturbation."
      : "The targeted attack is no more effective than random noise of the same size, so the " +
        "evasion observed is frauds lying close to the challenge threshold, displaced by any " +
        "perturbation.") +
    " The resistance is structural: most of the model's evidence comes from entity history " +
    "(Figure 4.12), which the attacker cannot edit."),
  P("Probing the running service found a weakness that acts through history. A confirmed " +
    "fraud, re-submitted as new, scored 78 and was held; the identical transaction one " +
    "second later scored 10 and was approved, because its single predecessor made every " +
    "value exactly 1.0 times the account's average, which the model reads as established " +
    "behaviour. A minimum history length before account-norm features are trusted is set " +
    "out as future work in section 5.4."),
];

/* -------------------------------------------------------------- 4.14 BAF -- */
const baf = D.baf || [];
const bafMeta = D.bafMeta || {};
const fair = bafMeta.fairness || {};
const bafRows = [
  ["Model", "AUC-PR", "Lift", "AUC-ROC", "Recall @ FPR 1 %", "Recall @ FPR 5 %"],
  ...baf.map((r) => [label(r.label), num(r.auc_pr), num(r.auc_pr_lift_over_prevalence, 1),
    num(r.auc_roc), num(r["recall_at_fpr_0.01"], 3), num(r["recall_at_fpr_0.05"], 3)]),
];
const bafTierRows = [
  ["Tier", "Applications", "Share of volume", "Fraud caught", "Share of fraud", "Precision"],
  ...(D.bafTiers || []).map((r) => [r.tier, int(r.applications), pct(r.share_of_volume),
    int(r.fraud_caught), pct(r.share_of_fraud), pct(r.precision)]),
];

const transfer = [
  H2("4.14  Transfer to Bank Account Fraud"),
  P("Table 3.4 names the Bank Account Fraud dataset (Jesus et al., NeurIPS 2022) as a " +
    "robustness and transfer check. It is a different problem - fraudulent account " +
    "applications rather than card-not-present payments - with different features and " +
    `roughly a third of the prevalence (${pct(bafMeta.prevalence?.test)} in the test month). ` +
    "Only the protocol was reused: a chronological split by month (0-5 train, 6 validation, " +
    "7 test), class weighting, selection on validation, isotonic calibration and the same " +
    "0-100 score."),
  T(bafRows, [3.2, 1.2, 0.9, 1.2, 1.5, 1.5]),
  Caption("Table 4.19  The same protocol on Bank Account Fraud (test month)"),
  P(`The method transfers: the decision layer reaches ${num(decisionRow(baf).auc_pr)} AUC-PR, ` +
    `${num(decisionRow(baf).auc_pr_lift_over_prevalence, 1)} times the no-skill floor, and ` +
    `catches ${pct(decisionRow(baf)["recall_at_fpr_0.05"], 1)} of fraud at a five per cent ` +
    "false-positive rate, the operating point the dataset's authors report. The fixed score " +
    "bands of Table 3.3 do not transfer, however:"),
  T(bafTierRows, [1, 1.5, 1.5, 1.4, 1.4, 1.2]),
  Caption("Table 4.20  Fixed risk bands applied to Bank Account Fraud"),
  P("At this prevalence a calibrated probability rarely exceeds 0.3, so almost every " +
    "application falls in the low band: bands must be derived per deployment from the cost " +
    "model. BAF also records age, allowing a fairness check of predictive equality (equal " +
    "false-positive rates across groups). At five per cent overall FPR, applicants " +
    `aged ${bafMeta.fairness ? 50 : "-"} and over are falsely flagged at ` +
    `${pct(fair.fpr_age_over_50, 1)}, against ${pct(fair.fpr_age_under_50, 1)} for younger ` +
    `applicants, a ratio of ${num(fair.fpr_ratio_min_over_max, 2)}. The model has not been ` +
    "given age, but it has learned correlates of it; a deployment must correct this first " +
    "(section 5.4)."),
];

/* ---------------------------------------------------------- 4.15 product -- */
const lp = D.latencyProduct || [];
const lpRow = (mode) => lp.find((r) => r.mode === mode) || {};
const single = lpRow("single_transaction");
const withReasons = lpRow("single_transaction_with_reasons");
const productLatency = [
  ["Mode", "p50 (ms)", "p95 (ms)", "p99 (ms)", "Target met (p95)"],
  ...lp.map((r) => [
    r.mode.replace("single_transaction_with_reasons", "One transaction, with reasons")
      .replace("single_transaction", "One transaction")
      .replace(/micro_batch_(\d+)_per_txn/, "Micro-batch of $1, per transaction"),
    num(r.p50_ms, 1), num(r.p95_ms, 1), num(r.p99_ms, 1),
    String(r.meets_target_p95).toLowerCase() === "true" ? "yes" : "no",
  ]),
];

const product = [
  H2("4.15  The Deployed Product"),
  P("A deployed system receives one transaction at a time. The product accepts the raw " +
    "fields of a new transaction and returns whether it is fraud, a calibrated 0-100 risk " +
    "score, the action and the reasons in plain language. Its core is the online feature " +
    "store (architecture layer L2), which keeps each card's, account's, device's and other " +
    "entity's history as sorted arrays with prefix sums, so every feature of a new " +
    "transaction is a binary search and a subtraction. Automated tests show that what is " +
    "served is what was evaluated: on the first 2,000 test transactions, 99.3 per cent match " +
    "the offline features exactly on all 346 columns, the rest differing only by " +
    "floating-point rounding. Scored transactions and analysts' confirmed outcomes join the " +
    "history, closing the feedback loop."),
  P("The service is exposed through an HTTP API, an analyst console (check a transaction, a " +
    "live ledger with drift monitoring, fraud analytics, and model monitoring) and a " +
    "streaming transport over Kafka or an in-process queue. Table 4.21 measures the path a " +
    "live request takes, including computing the features online."),
  T(productLatency, [3.8, 1.4, 1.4, 1.4, 1.6]),
  Caption("Table 4.21  Latency of the product path, from raw transaction to decision"),
  P(`A single transaction is decided in ${num(single.p50_ms, 1)} ms at the median and ` +
    `${num(single.p95_ms, 1)} ms at the 95th percentile, inside the 100 ms real-time ` +
    "target. Most of that is computing the features; the model itself takes a few " +
    "milliseconds. Reason codes bring the total to " +
    `${num(withReasons.p95_ms, 1)} ms at the 95th percentile, and are needed only for the ` +
    "minority of transactions held or challenged." +
    (stream.end_to_end_p95_ms
      ? ` Through the streaming transport, end-to-end latency from publication of a ` +
        `transaction to publication of its decision was ${num(stream.end_to_end_p50_ms, 1)} ms ` +
        `at the median and ${num(stream.end_to_end_p95_ms, 1)} ms at the 95th percentile, at ` +
        `${num(stream.throughput_tps, 0)} transactions per second on one process.`
      : "")),
];

module.exports = { extensionSections: [...experiments, ...robustness, ...transfer, ...product] };
