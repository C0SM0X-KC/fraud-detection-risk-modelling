/** Chapter 4 sections 4.2, 4.3 and 4.7: exploratory analysis, statistical tests, segments. */

const R = require("./report_frontmatter.js");

const { D, P, H2, Bullets, T, Figure, Caption, num, int, pct, best } = R;

const A = D.analytics || {};
const conc = A.concentration || {};
const boot = A.bootstrap || {};
const rows = (table, key, value) => (table || []).filter((r) => r[key] === value);
const one = (table, dimension, level) =>
  (table || []).find((r) => r.dimension === dimension && r.level === level) || {};
const seg = (dimension, segment) =>
  (D.segments || []).find((r) => r.dimension === dimension && r.segment === segment) || {};
const pval = (v) => (Number(v) < 0.001 ? "< 0.001" : num(v, 3));
const ratio = (a, b) => num(Number(a) / Number(b), 1);

// 7-day average of the daily fraud rate, for the range quoted against Figure 4.1.
const daily = (D.edaDaily || []).map((r) => Number(r.fraud_rate));
const weekly = daily.map((_, i) => {
  const w = daily.slice(Math.max(0, i - 3), i + 4);
  return w.reduce((a, b) => a + b, 0) / w.length;
});
const weeklyMin = Math.min(...weekly);
const weeklyMax = Math.max(...weekly);
const perDay = (D.edaDaily || []).reduce((a, r) => a + Number(r.transactions), 0) /
  Math.max((D.edaDaily || []).length, 1);

const legit = (D.edaAmount || []).find((r) => r.class === "legitimate") || {};
const fraud = (D.edaAmount || []).find((r) => r.class === "fraud") || {};
const amountTest = (D.statsNumeric || []).find((r) => r.column === "TransactionAmt") || {};
const band = (level) => one(D.edaSegments, "Amount band", level);
const identityYes = one(D.edaSegments, "Identity record present", "yes");
const identityNo = one(D.edaSegments, "Identity record present", "no");

/* ------------------------------------------------------------- 4.2 EDA ---- */
const qualityRows = [
  ["Column family", "Columns", "Mean missing", "Worst column", "Columns > 90 % missing"],
  ...(D.edaQuality || []).map((r) => [
    r.family, r.columns, `${num(r.mean_missing_pct, 1)} %`, `${num(r.max_missing_pct, 1)} %`,
    r.columns_over_90pct_missing]),
];

const amountRows = [
  ["Class", "Transactions", "Mean", "Median", "25th pct", "75th pct", "99th pct", "Total value"],
  ...[legit, fraud].map((r) => [
    r.class === "fraud" ? "Fraud" : "Legitimate", int(r.transactions), num(r.mean, 2),
    num(r.median, 2), num(r.p25, 2), num(r.p75, 2), num(r.p99, 0), int(Math.round(r.total_value))]),
];

const eda = [
  H2("4.2  Exploratory Data Analysis"),
  P("Before any model is fitted, the data are examined to establish what fraud looks like " +
    "in this benchmark, where the data are incomplete and which attributes carry signal. " +
    "Every statistic in this section and the next is computed on the training span alone " +
    `(${int(A.train_rows)} transactions, ${int(A.train_fraud)} of them fraudulent), so that ` +
    "nothing here draws on the periods used to choose or to report the model. Figure 4.1 is " +
    "the one exception: it shows the whole timeline, because its purpose is to describe the " +
    "split itself."),
  ...Figure("reports/figures/eda_timeline.png",
    "Figure 4.1  Daily transaction volume and fraud rate across the timeline"),
  P(`The benchmark averages about ${int(Math.round(perDay / 100) * 100)} transactions a day, ` +
    "with a pronounced surge in the first four weeks. The fraud rate is not constant: its " +
    `seven-day average moves between ${pct(weeklyMin, 1)} and ${pct(weeklyMax, 1)}. A rate ` +
    "that drifts over time is the reason the evaluation protocol cuts on time rather than " +
    "at random: a random split would mix every period into training and hide the drift " +
    "that a deployed model has to cope with."),

  P("Data quality. The merged table has " + `${A.columns ?? "-"} columns. Table 4.3 groups ` +
    "them into families and reports how complete each family is."),
  T(qualityRows, [2.6, 1.2, 1.6, 1.6, 2.2]),
  Caption("Table 4.3  Data-quality profile of the training span by column family"),
  P("Only the count features (C1-C14) are complete. The identity family is the emptiest " +
    `because only ${pct(A.identity_coverage, 1)} of transactions have an identity record at ` +
    `all, and ${A.columns_over_90pct_missing ?? "-"} columns are more than 90 per cent ` +
    "empty. The missing values are not random noise: transactions with an identity record " +
    `are fraudulent at ${pct(identityYes.fraud_rate)}, against ${pct(identityNo.fraud_rate)} ` +
    `for those without, ${ratio(identityYes.fraud_rate, identityNo.fraud_rate)} times as ` +
    "often. Whether a field is filled in is itself evidence, which is why step A.2 of Table " +
    "3.1 leaves missing values unfilled for the tree models."),

  P("Transaction amount. Table 4.4 and Figure 4.2 compare the amounts of fraudulent and " +
    "legitimate transactions."),
  T(amountRows, [1.6, 1.5, 1, 1, 1, 1, 1, 1.6]),
  Caption("Table 4.4  Transaction amount by class, training span"),
  ...Figure("reports/figures/eda_amount.png",
    "Figure 4.2  Distribution of transaction amount, fraud against legitimate"),
  P(`The two distributions are almost the same: the median fraud is ${num(fraud.median, 2)} ` +
    `against ${num(legit.median, 2)} for legitimate payments, and Section 4.3 finds the ` +
    `difference negligible (rank-biserial r = ${num(amountTest.rank_biserial, 3)}). Amount ` +
    "on its own does not separate fraud. It is not irrelevant, however. Broken into bands " +
    `(Figure 4.4), the fraud rate is highest for small payments under 25 ` +
    `(${pct(band("under 25").fraud_rate)}) and again for 250-500 ` +
    `(${pct(band("250-500").fraud_rate)}), and lowest in the middle, 50-100 ` +
    `(${pct(band("50-100").fraud_rate)}). Small payments are commonly associated with testing ` +
    "whether a stolen card works, and larger ones with cashing it out. A U-shape like this " +
    "cannot be captured by a single linear weight, one reason the tree ensembles in Section " +
    "4.4 outperform logistic regression."),

  ...Figure("reports/figures/eda_time_cycles.png",
    "Figure 4.3  Fraud rate and volume by hour of day and day of the weekly cycle"),
  P("Time of day. The dataset's clock is an offset from an undisclosed starting point, so " +
    "the hour labels are relative rather than local time. Volume is lowest at hour " +
    `${A.quietest_hour ?? "-"} and highest at hour ${A.busiest_hour ?? "-"}. The fraud rate ` +
    `peaks at hour ${A.peak_fraud_hour ?? "-"} (${pct(A.peak_fraud_hour_rate, 1)}), close to ` +
    `the quietest hour, and is lowest at hour ${A.lowest_fraud_hour ?? "-"} ` +
    `(${pct(A.lowest_fraud_hour_rate, 1)}). The pattern is consistent with the quiet hours ` +
    "being night-time for most customers: little genuine spending happens then, so fraud " +
    "forms a larger share of what does. The weekly cycle is far weaker."),

  P("Fraud rate by segment. Figure 4.4 breaks the fraud rate down by the categorical " +
    "attributes, each with a 95 per cent Wilson interval."),
  ...Figure("reports/figures/eda_categories.png", "Figure 4.4  Fraud rate by segment"),
  P(`Product C is fraudulent ${num(one(D.edaSegments, "Product code", "C").lift, 1)} times as ` +
    "often as the average transaction, while product W, which carries " +
    `${pct(one(D.edaSegments, "Product code", "W").share_of_volume, 0)} of volume, is below ` +
    `average. Credit cards are defrauded at ${pct(one(D.edaSegments, "Card type", "credit").fraud_rate)} ` +
    `against ${pct(one(D.edaSegments, "Card type", "debit").fraud_rate)} for debit cards, and ` +
    `mobile devices at ${pct(one(D.edaSegments, "Device type", "mobile").fraud_rate)} against ` +
    `${pct(one(D.edaSegments, "Device type", "desktop").fraud_rate)} for desktops. The card ` +
    "network matters little. These one-attribute views overlap - device type is recorded " +
    "only with an identity record - and separating such effects is what the multivariate " +
    "models do."),

  ...Figure("reports/figures/eda_correlation.png",
    "Figure 4.5  Correlation between amount, count (C) and time-delta (D) features"),
  P("Redundancy. Figure 4.5 shows rank correlations among the amount, count and time-delta " +
    `features. ${A.correlation_pairs_above_0_8 ?? "-"} of the ${A.correlation_pairs ?? "-"} ` +
    "pairs have a correlation above 0.8, the strongest being " +
    (A.strongest_correlations || []).map((c) => `${c.pair} (${num(c.rho, 2)})`).join(", ") +
    ". The time-delta features also form a moderately correlated block. A near-duplicate " +
    "column adds little, the same reasoning by which the 339 Vesta columns were pruned " +
    "(Section 4.1)."),

  ...Figure("reports/figures/eda_pareto.png",
    "Figure 4.6  Concentration of fraud value across accounts"),
  P(`Concentration of fraud. Only ${int(conc.accounts_with_fraud)} of the ` +
    `${int(conc.accounts)} accounts in the training span (${pct(conc.share_of_accounts_with_fraud)}) ` +
    "were used for fraud at all, and the loss among them is concentrated: " +
    `${pct(conc.fraud_accounts_for_50pct_value, 1)} of those accounts carry half of the ` +
    `fraud value and ${pct(conc.fraud_accounts_for_80pct_value, 1)} carry 80 per cent ` +
    "(Figure 4.6). Fraud also repeats: " +
    `${pct(conc.fraud_on_account_with_earlier_fraud, 1)} of fraudulent transactions came from ` +
    "an account that had already been used for fraud earlier in the span. This is the " +
    "descriptive fact behind the strongest evidence the model relies on, an account's own " +
    "fraud history (Section 4.8), and behind its sensitivity to label delay (Section 4.11): " +
    "the repetition can be exploited only once the earlier fraud is known."),
];

/* ----------------------------------------------------- 4.3 statistics ---- */
const catRows = [
  ["Attribute", "Levels", "Chi-square", "df", "p (Bonferroni)", "Cramér's V", "Strength",
    "Riskiest level (rate)"],
  ...(D.statsCategorical || []).map((r) => [
    r.variable, r.levels, int(Math.round(r.chi2)), r.dof, pval(r.p_bonferroni),
    num(r.cramers_v, 3), r.strength, `${r.highest_rate_level} (${pct(r.highest_rate, 1)})`]),
];
const numRows = [
  ["Attribute", "Available", "Median, legitimate", "Median, fraud", "p (Bonferroni)",
    "P(fraud higher)", "Rank-biserial r", "Strength"],
  ...(D.statsNumeric || []).map((r) => [
    r.variable, `${num(r.available_pct, 0)} %`, num(r.median_legitimate, 1),
    num(r.median_fraud, 1), pval(r.p_bonferroni), num(r.prob_fraud_higher, 3),
    num(r.rank_biserial, 3), r.strength]),
];
const nTests = (D.statsCategorical || []).length + (D.statsNumeric || []).length;
const nSignificant = [...(D.statsCategorical || []), ...(D.statsNumeric || [])]
  .filter((r) => Number(r.p_bonferroni) < 0.05).length;
const topCat = (D.statsCategorical || [])[0] || {};
const topNum = (D.statsNumeric || [])[0] || {};
const anyMedium = [...(D.statsCategorical || []), ...(D.statsNumeric || [])]
  .some((r) => ["medium", "large"].includes(r.strength));
const notSignificant = [...(D.statsCategorical || []), ...(D.statsNumeric || [])]
  .filter((r) => Number(r.p_bonferroni) >= 0.05).map((r) => r.variable.toLowerCase());
const lowerFirst = (s) => (s ? s[0].toLowerCase() + s.slice(1) : "-");

const statistics = [
  H2("4.3  Statistical Testing of the Fraud Signals"),
  P("The differences seen in Section 4.2 come from a sample, so each is tested formally. " +
    "Every categorical attribute is tested against the fraud label with Pearson's " +
    "chi-square test of independence. Every numeric attribute is compared between fraud " +
    "and legitimate transactions with the Mann-Whitney U test, which compares ranks and so " +
    "does not assume a normal distribution; amounts and counts are heavily skewed. The " +
    `${nTests} p-values are Bonferroni-corrected for multiple testing.`),
  P(`With ${int(A.train_rows)} transactions even a trivial difference is significant, so ` +
    "each test also reports an effect size: Cramér's V for categories, and for numbers the " +
    "rank-biserial correlation r = 2P - 1, where P is the probability that a randomly chosen " +
    "fraud has the larger value. Both are read on Cohen's scale: 0.1 small, 0.3 medium, 0.5 " +
    "large."),
  T(catRows, [2.4, 0.9, 1.3, 0.6, 1.3, 1.1, 1.2, 2.0], { numericFrom: 1 }),
  Caption("Table 4.5  Chi-square tests of categorical attributes against fraud"),
  T(numRows, [2.6, 1.1, 1.3, 1.2, 1.3, 1.2, 1.3, 1.2], { numericFrom: 1 }),
  Caption("Table 4.6  Mann-Whitney U tests of numeric attributes, fraud against legitimate"),
  ...Figure("reports/figures/stats_effect_sizes.png",
    "Figure 4.7  Strength of association with fraud for each attribute"),
  P(`${nSignificant} of the ${nTests} tests reject "no association" after correction` +
    (notSignificant.length ? ` (the exception is ${notSignificant.join(" and ")})` : "") +
    ", which confirms that significance cannot rank attributes at this sample size. The " +
    `effect sizes can. The strongest categorical attribute is ${lowerFirst(topCat.variable)} ` +
    `(V = ${num(topCat.cramers_v, 3)}) and the strongest numeric one is ` +
    `${lowerFirst(topNum.variable)} (|r| = ${num(Math.abs(topNum.rank_biserial), 3)}). ` +
    (anyMedium
      ? "Some attributes reach a medium effect, but most do not."
      : "No attribute reaches even a medium effect: each is individually a weak signal.") +
    " Amount, card network and the weekly cycle are negligible."),
  P("This result explains how the system is built. Since no single attribute separates " +
    "fraud from legitimate payments, a rule on one attribute cannot work. The AUC-PR of " +
    `${num(best?.auc_pr, 2)} reported in Section 4.4 is reached by combining many weak ` +
    "signals with each account's own history, the task multivariate models such as " +
    "gradient-boosted trees are designed for."),
];

/* ---------------------------------------- 4.7 reliability and segments --- */
const bootRows = [
  ["Measure", "Estimate", "95 % interval"],
  ...(D.bootstrap || []).map((r) => {
    const isPct = !r.measure.startsWith("AUC-PR");
    const f = (v) => (isPct ? pct(v, r.measure.includes("false-positive") ? 2 : 1) : num(v, 4));
    return [r.measure.replace(" | ", ": "), f(r.estimate), `${f(r.ci_low)} - ${f(r.ci_high)}`];
  }),
];
const segmentPicksTest = [
  ["Account history", ["first seen", "1-4 earlier", "5-19 earlier", "20+ earlier"]],
  ["Amount band", ["under 25", "25-50", "50-100", "100-250", "250-500", "500+"]],
  ["Product code", ["W", "C", "R", "S", "H"]],
  ["Device type", ["desktop", "mobile", "missing"]],
];
const segRows = [
  ["Segment", "Share of volume", "Fraud rate", "AUC-PR", "Fraud caught", "Precision (high)",
    "False-positive rate"],
  ...segmentPicksTest.flatMap(([dimension, levels]) => levels.map((level) => {
    const r = seg(dimension, level);
    const name = level === "missing" ? "no device data" : level;
    return [`${dimension}: ${name}`, pct(r.share_of_volume, 1), pct(r.fraud_rate, 1),
      num(r.auc_pr, 3), pct(r.fraud_caught_flagged, 1), pct(r.precision_high, 1),
      pct(r.false_positive_rate, 2)];
  })),
];
const first = seg("Account history", "first seen");
const returning = ["1-4 earlier", "5-19 earlier", "20+ earlier"].map((s) => seg("Account history", s));
const caught = returning.map((r) => Number(r.fraud_caught_flagged));
const aucs = returning.map((r) => Number(r.auc_pr));
const small = seg("Amount band", "under 25");
const large = seg("Amount band", "500+");
const productC = seg("Product code", "C");

const segments = [
  H2("4.7  Reliability of the Results and Performance by Segment"),
  P(`The test-span figures are estimates from ${boot.days ?? "-"} days of data, and a ` +
    "different six weeks would give somewhat different numbers. To measure how much, the " +
    `test span was resampled ${int(boot.resamples)} times by drawing whole days with ` +
    "replacement and recomputing every measure; the middle 95 per cent of the results is the " +
    "confidence interval. Whole days are drawn because transactions on the same day share " +
    "conditions, and treating them as independent would make the intervals too narrow."),
  T(bootRows, [5.4, 1.6, 2.4]),
  Caption("Table 4.11  95 per cent confidence intervals for the test-span results " +
    "(day-block bootstrap)"),
  P("The intervals are narrow: the high-risk tier catches between " +
    `${pct(D.bootstrap?.[4]?.ci_low, 1)} and ${pct(D.bootstrap?.[4]?.ci_high, 1)} of fraud at ` +
    `a precision between ${pct(D.bootstrap?.[5]?.ci_low, 1)} and ` +
    `${pct(D.bootstrap?.[5]?.ci_high, 1)}. The intervals of the three models overlap, which ` +
    "on its own would leave open whether the selected blend is genuinely better than " +
    "XGBoost alone. Comparing the two on the same resampled days answers it, since " +
    `variation shared by both cancels out: the blend is ahead by ${num(boot.difference, 4)} ` +
    `AUC-PR (95 per cent interval ${num(boot.ci_low, 4)} to ${num(boot.ci_high, 4)}) and ` +
    `ahead in ${pct(boot.share_of_resamples_better, 1)} of resamples. ` +
    (Number(boot.ci_low) > 0
      ? "The gain is small but not a chance result."
      : "The interval includes zero, so the gain is not distinguishable from chance.")),

  P("A single overall figure can hide segments where the system does badly. Table 4.12 " +
    "and Figure 4.11 break the test-span performance of the deployed decision layer down " +
    "by segment. Fraud caught counts the medium and high tiers together, since both act on " +
    "the transaction; precision is that of the high tier, the manual-review queue."),
  T(segRows, [3.0, 1.2, 1.1, 1.0, 1.2, 1.3, 1.4]),
  Caption("Table 4.12  Detection performance by segment on the held-out test span"),
  ...Figure("reports/figures/segment_detection.png",
    "Figure 4.11  Detection by segment on the held-out test span"),
  P("The clearest weakness is accounts seen for the first time. They make up " +
    `${pct(first.share_of_volume, 1)} of test volume, and the system catches only ` +
    `${pct(first.fraud_caught_flagged, 1)} of their fraud (AUC-PR ${num(first.auc_pr, 3)}), ` +
    `against ${pct(Math.min(...caught), 1)} to ${pct(Math.max(...caught), 1)} (AUC-PR ` +
    `${num(Math.min(...aucs), 3)} to ${num(Math.max(...aucs), 3)}) for accounts with ` +
    "earlier transactions. The model's strongest evidence is an account's own history " +
    "(Section 4.8), so it is weakest exactly where no history exists. The same mechanism " +
    "underlies the behavioural weakness found by probing the live service (Section 4.13)."),
  P("Detection also falls as the amount rises: " +
    `${pct(small.fraud_caught_flagged, 1)} of fraud under 25 is caught but only ` +
    `${pct(large.fraud_caught_flagged, 1)} above 500, where the precision is highest ` +
    `(${pct(large.precision_high, 1)}). Expensive frauds are therefore missed more often than ` +
    "cheap ones, and only " +
    `${pct(large.fraud_value_caught, 1)} of fraud value above 500 is caught. Product C, the ` +
    "riskiest product in Section 4.2, has the highest false-positive rate of the five " +
    `products (${pct(productC.false_positive_rate, 2)}), consistent with its high base rate ` +
    "raising the scores of legitimate C payments as well."),
  P("Three operational recommendations follow from these measurements:"),
  ...Bullets([
    "For accounts with no history, the decision should lean on signals that do not need " +
    "history, such as device, e-mail domain and velocity across the card, or route them to " +
    "step-up authentication at a lower score.",
    "For high-value transactions, a threshold that falls as the amount at risk rises, as " +
    "the cost model of Section 4.6 implies, would target the value that a flat threshold " +
    "misses.",
    "Score bands should be set per product, so that product C does not consume a " +
    "disproportionate share of the review queue.",
  ]),
];

module.exports = { analyticsEda: [...eda, ...statistics], analyticsSegments: segments };
