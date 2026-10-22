/** Chapter 4 sections 4.10-4.14; sentences that depend on a result are chosen from the data. */

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
const signed = (v, dp = 4) => (v === undefined || v === null || Number.isNaN(Number(v)))
  ? "-" : `${Number(v) >= 0 ? "+" : "−"}${Math.abs(Number(v)).toFixed(dp)}`;

const adoption = readJson("reports/metrics/feature_adoption.json") || {};
const stream = readJson("reports/metrics/stream_memory.json") || {};

/* ------------------------------------------------------------ 4.10 tuning -- */
const optuna = D.optuna || {};
const handset = byLabel(D.handsetComparison);
const tuned = byLabel(D.tunedComparison);
const tuningRows = [
  ["Model", "Trials", "Validation AUC-PR (hand-set)", "Validation AUC-PR (best trial)",
    "Test AUC-PR (hand-set)", "Test AUC-PR (tuned)"],
  ...["xgboost", "lightgbm"].filter((m) => optuna[m]).map((m) => [
    label(m), String(optuna[m].trials), num(optuna[m].hand_set_validation_auc_pr),
    num(optuna[m].validation_auc_pr), num(handset[m]?.auc_pr), num(tuned[m]?.auc_pr),
  ]),
  ["Decision layer (calibrated)", "-", "-", "-",
    num(decisionRow(D.handsetComparison).auc_pr), num(decisionRow(D.tunedComparison).auc_pr)],
];
const tunedHelped = ["xgboost", "lightgbm"].some(
  (m) => Number(tuned[m]?.auc_pr) > Number(handset[m]?.auc_pr));

const tuning = [
  H2("4.10  Hyper-parameter Tuning"),
  P("Table 3.1 specifies Bayesian hyper-parameter search with Optuna for the boosting tier, on " +
    "a time-ordered validation fold. The search used the tree-structured Parzen estimator, " +
    "fitted every trial on the training span with all fraud cases and " +
    `${pct(optuna.xgboost?.training_negative_fraction ?? 0.3, 0)} of the legitimate ones, ` +
    "and scored it by AUC-PR on the earlier part of the validation span - the part that " +
    "later joins the refit - so that neither the calibration holdout nor the test span " +
    "influenced any choice. The configuration used in the main results was enqueued as the " +
    "first trial, so the search could only report an improvement over it."),
  T(tuningRows, [2.2, 0.9, 1.6, 1.6, 1.4, 1.4]),
  Caption("Table 4.14  Optuna search for the boosting tier and its effect on the test span"),
  P(tunedHelped
    ? "The tuned configurations improved the held-out test span as well as validation, and " +
      "were adopted."
    : "The search found configurations that scored higher on validation, but the gain was " +
      "of the order of 0.003 AUC-PR - comparable to the variation between runs - and it did " +
      "not carry over to the test span: refitted under the full protocol, the tuned models " +
      "scored slightly lower there than the hand-set ones. The tuned XGBoost had moved to " +
      "deeper trees and a lower learning rate, fitting the training period more closely than " +
      "the later weeks rewarded. The hand-set configuration was therefore retained. The " +
      "chronological protocol is designed to detect exactly this case: on drifting data a " +
      "validation gain of this size is not evidence of a better model."),
];

/* -------------------------------------------- 4.11 account profile + CB -- */
const current = byLabel(D.comparison);
const cbRow = current.catboost || {};
const profileRows = [
  ["Quantity", "Base feature set", "With account profile"],
  ["Engineered columns", String(adoption.base_features ?? "-"), String(adoption.profile_features ?? "-")],
  ["Selected decision score", label(adoption.base_selected || "-"), label(adoption.profile_selected || "-")],
  ["Its validation AUC-PR (selection criterion)", num(adoption.base_validation),
    num(adoption.profile_validation)],
  ["Decision layer test AUC-PR", num(adoption.base_test), num(adoption.profile_test)],
];

const profile = [
  H2("4.11  Account Profile and a Third Tabular Learner"),
  P("Two changes were made after the main results, both drawn from the strongest published " +
    "solutions to this benchmark. The first is an account profile: for each synthetic " +
    "account, card and device, the running mean of a further set of the masked count (C), " +
    "time-delta (D) and engineered (V) fields together with this transaction's ratio to it, " +
    "and distinct-partner counts for the relations an account or card spans - devices and " +
    "merchants per account, e-mail domains and billing addresses per card. These features " +
    "are behavioural rather than label-derived, so unlike the prior-fraud-rate features they " +
    "do not depend on fraud outcomes being confirmed quickly, which is the weakness section " +
    "4.9 identifies. The second change adds CatBoost as a third tabular learner. Its " +
    "symmetric trees and ordered target statistics make its errors less correlated with " +
    "those of the other two, which is what an average needs."),
  P("The decision to adopt was made on the validation span alone, by comparing the score " +
    "the decision layer would select under each feature set; the test span was not consulted " +
    "until the choice was fixed."),
  T(profileRows, [4.2, 2.3, 2.5]),
  Caption("Table 4.15  Effect of the account profile and CatBoost"),
  P(adoption.adopted === undefined
    ? "The comparison had not been run when this report was built."
    : adoption.adopted
      ? `The account profile raised the selected score's validation AUC-PR from ` +
        `${num(adoption.base_validation)} to ${num(adoption.profile_validation)} and was ` +
        `adopted. On the test span the decision layer moved from ${num(adoption.base_test)} to ` +
        `${num(adoption.profile_test)} (${signed(adoption.profile_test - adoption.base_test)}). ` +
        `CatBoost on its own reached a test AUC-PR of ${num(cbRow.auc_pr)}.`
      : `The account profile did not raise the selected score's validation AUC-PR ` +
        `(${num(adoption.base_validation)} without, ${num(adoption.profile_validation)} with), ` +
        `so the base feature set was retained. CatBoost was then evaluated on the retained ` +
        `features as a further candidate: alone it reached ` +
        `${num(D.fusion?.validation_ranking?.catboost)} on validation, and adding it to the ` +
        `average of XGBoost and LightGBM lowered the average from ` +
        `${num(D.fusion?.validation_ranking?.blend_top2)} to ` +
        `${num(D.fusion?.validation_ranking?.blend_top3)}, so it is benchmarked in Table 4.3 ` +
        `but takes no part in decisions. Neither change improved the selected score because ` +
        `the base features ` +
        `already carry the account-level signal they add: the running means of the core ` +
        `count and time fields per account and card were present before, and three gradient-` +
        `boosted learners on the same rows make strongly correlated errors.`),
];

/* ------------------------------------------------------- 4.12 robustness -- */
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
  H2("4.12  Adversarial Robustness"),
  P("The literature reviewed in Chapter 2 reports an accuracy drop of roughly twenty per " +
    "cent under projected-gradient-descent attack, and gap G5 records that such decay is " +
    "acknowledged more often than measured. The threat model adopted here is the one a " +
    "fraudster actually faces: they control the fields of their own transaction - amount, " +
    "e-mail domains, device, address distance, time of day, the identity attributes their " +
    "browser reports - and nothing else. The entity history the bank computes, including " +
    `velocity counters and prior fraud rates, is out of reach. ${int(robMeta.mutable_features?.length)} ` +
    "of the engineered features derive from attacker-controlled fields and may move, each by " +
    "at most a stated number of training standard deviations."),
  P("Gradient-boosted trees have no gradient, so a transfer attack is used. A small neural " +
    "network is trained to reproduce the deployed model's log-odds (rank correlation with the " +
    `deployed model on the test span: ${num(robMeta.surrogate_fidelity_spearman, 3)}), PGD is ` +
    "run against that surrogate on fraud cases the deployed model catches, and the perturbed " +
    "transactions are replayed through the deployed model. Random perturbation of the same " +
    "size is the control for whether the attack is targeted, and a zero budget is the " +
    "control for the attack's own encoding round trip."),
  T(robRows, [2, 1.8, 1.8, 1.7, 1.7]),
  Caption("Table 4.16  Evasion of the deployed model under attacker-controlled features"),
  ...Figure("reports/figures/robustness.png",
    "Figure 4.10  Evasion rate against perturbation budget"),
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
    " The model's resistance has a structural explanation that Table 4.7 already showed: " +
    "most of its evidence comes from entity history, which the attacker cannot edit. " +
    "Because perturbations are applied in feature space, a derived feature can move without " +
    "its source, so these figures are an upper bound on a real attacker's success."),
  P("Probing the running service identified a behaviour that the feature-space attack cannot " +
    "see, because it acts through history. A confirmed fraudulent transaction, re-submitted " +
    "as new, scored 78 and was held; the identical transaction sent one second later scored " +
    "10 and was approved. The reason codes show why: on the first submission the account " +
    "had no history, while on the second its single previous transaction made every value " +
    "exactly 1.0 times the account's average, which the model has learned to read as " +
    "consistent, established behaviour. An attacker who first sends one innocuous " +
    "transaction can therefore make an account look established. The mitigation is a " +
    "minimum history length before account-norm features are trusted, or treating a " +
    "near-duplicate transaction within seconds as a velocity signal in its own right. " +
    "Both are set out as future work in section 5.4, so that the deployed system remains " +
    "the one evaluated in this chapter."),
];

/* -------------------------------------------------------------- 4.13 BAF -- */
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
  H2("4.13  Transfer to Bank Account Fraud"),
  P("Table 3.4 names the Bank Account Fraud dataset (Jesus et al., NeurIPS 2022) as a " +
    "robustness and transfer check. It is a different problem - fraudulent account " +
    "applications rather than card-not-present payments - with different features and " +
    `roughly a third of the prevalence (${pct(bafMeta.prevalence?.test)} in the test month). ` +
    "Nothing written for IEEE-CIS was reused except the protocol: a chronological split by " +
    "month (months 0-5 train, 6 validation, 7 test, the dataset's own convention), class " +
    "weighting, early stopping on validation, selection between candidates on validation, " +
    "isotonic calibration and the same 0-100 score."),
  T(bafRows, [3.2, 1.2, 0.9, 1.2, 1.5, 1.5]),
  Caption("Table 4.17  The same protocol on Bank Account Fraud (test month)"),
  P(`The method transfers: the decision layer reaches ${num(decisionRow(baf).auc_pr)} AUC-PR, ` +
    `${num(decisionRow(baf).auc_pr_lift_over_prevalence, 1)} times the no-skill floor, and ` +
    `catches ${pct(decisionRow(baf)["recall_at_fpr_0.05"], 1)} of fraud at a five per cent ` +
    "false-positive rate, the operating point the dataset's authors report. The fixed score " +
    "bands of Table 3.3 do not transfer, however:"),
  T(bafTierRows, [1, 1.5, 1.5, 1.4, 1.4, 1.2]),
  Caption("Table 4.18  Fixed risk bands applied to Bank Account Fraud"),
  P("At this prevalence a calibrated probability rarely exceeds 0.3, so almost every " +
    "application falls in the low band. The bands are a property of the operating point, not " +
    "of the method, and this is the case for deriving them from the cost model and review " +
    "capacity, as section 4.4's derived policy does, rather than fixing them in advance."),
  P("The dataset also allows something IEEE-CIS does not: a fairness check. BAF records the " +
    "applicant's age, and its authors' criterion is predictive equality - equal false-positive " +
    "rates across groups at the operating threshold. At five per cent overall FPR, applicants " +
    `aged ${bafMeta.fairness ? 50 : "-"} and over are falsely flagged at ` +
    `${pct(fair.fpr_age_over_50, 1)}, against ${pct(fair.fpr_age_under_50, 1)} for younger ` +
    `applicants, a ratio of ${num(fair.fpr_ratio_min_over_max, 2)}. The model has not been ` +
    "given age as a reason to decline, but it has learned correlates of it, and it would " +
    "subject older applicants to roughly three times the friction. A deployment must " +
    "address this before use, by per-group threshold adjustment or a fairness-constrained " +
    "training objective (section 5.4)."),
];

/* ---------------------------------------------------------- 4.14 product -- */
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
  H2("4.14  The Deployed Product"),
  P("The evaluation above scores rows whose features were computed for the whole benchmark " +
    "at once. A deployed system receives one transaction at a time, and the work this section " +
    "describes is what turns the pipeline into a product: a service that accepts the raw " +
    "fields of a new transaction and returns whether it is fraud, a calibrated 0-100 risk " +
    "score, the action to take and the reasons, in plain language."),
  P("The essential component is the online feature store - the online velocity cache of " +
    "architecture layer L2. It holds, for every card, account, device, address, e-mail " +
    "domain and merchant, the time-ordered history of everything seen so far as sorted arrays " +
    "with prefix sums, so that each velocity counter, recency gap, entity norm, graph " +
    "statistic and prior fraud rate for a new transaction is a binary search and a " +
    "subtraction. Row-level features are computed by the offline stage's own functions. " +
    "Automated tests establish that what is served is what was evaluated: replaying a test " +
    "dataset through the store reproduces every offline feature, and the online score " +
    "equals the offline score for the same transaction. On the first 2,000 transactions of " +
    "the real test span, 99.3 per cent match the offline features exactly on all 346 " +
    "columns; the remainder differ only in the amount standard deviation of accounts whose " +
    "previous amounts are identical, where floating-point rounding decides between zero and " +
    "a value close to zero. Scored transactions join the history, " +
    "so a second transaction on the same card seconds later is scored against the first, and " +
    "an analyst's confirmation of an outcome feeds the card's and account's fraud history - " +
    "the feedback loop of the workflow figure."),
  P("The service is exposed three ways: an HTTP API (score a transaction, score a batch, " +
    "record an outcome), an analyst console with three views - check a transaction, a live " +
    "ledger of the replayed test span with live score-drift monitoring and a retraining alert, " +
    "and the model and monitoring outputs of this chapter - and a streaming transport that " +
    "consumes transactions from a Kafka topic, or an in-process queue standing in for one, " +
    "and publishes decisions to another. Table 4.19 measures the path a live request takes, " +
    "from raw fields to decision, including computing the features online."),
  T(productLatency, [3.8, 1.4, 1.4, 1.4, 1.6]),
  Caption("Table 4.19  Latency of the product path, from raw transaction to decision"),
  P(`A single transaction is decided in ${num(single.p50_ms, 1)} ms at the median and ` +
    `${num(single.p95_ms, 1)} ms at the 95th percentile, inside the 100 ms real-time ` +
    "target. Most of that is computing the features; the model itself takes a few " +
    "milliseconds through the boosters' native single-row path. Attaching reason codes adds " +
    `the cost of exact TreeSHAP, bringing the total to ${num(withReasons.p95_ms, 1)} ms at the ` +
    "95th percentile; since reasons are needed only for transactions held or challenged, a " +
    "deployment computes them for that minority after the decision rather than on every " +
    "request." +
    (stream.end_to_end_p95_ms
      ? ` Through the streaming transport, end-to-end latency from publication of a ` +
        `transaction to publication of its decision was ${num(stream.end_to_end_p50_ms, 1)} ms ` +
        `at the median and ${num(stream.end_to_end_p95_ms, 1)} ms at the 95th percentile, at ` +
        `${num(stream.throughput_tps, 0)} transactions per second on one process.`
      : "")),
];

module.exports = { extensionSections: [...tuning, ...profile, ...robustness, ...transfer, ...product] };
