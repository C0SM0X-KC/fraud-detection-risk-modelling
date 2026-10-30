/** Shared helpers, data loading and front matter for the Phase-3 report. */

const fs = require("fs");
const path = require("path");
const {
  AlignmentType, BorderStyle, Document, Footer, HeadingLevel, ImageRun,
  LevelFormat, PageBreak, PageNumber, Packer, Paragraph, ShadingType, Table,
  TableCell, TableRow, TableOfContents, TextRun, VerticalAlign, WidthType,
} = require("docx");

const ROOT = path.resolve(__dirname, "..");
const OUT = path.join(ROOT, "reports", "Phase-3_Fraud_Detection_Report.docx");

// A4 minus 1-inch margins, in DXA.
const CONTENT_WIDTH = 9026;
const FIG_MAX_PX = 480;

/* ------------------------------------------------------------------ data -- */
const readJson = (rel) => {
  const p = path.join(ROOT, rel);
  if (!fs.existsSync(p)) { console.warn(`  missing ${rel}`); return null; }
  return JSON.parse(fs.readFileSync(p, "utf8"));
};

// RFC 4180 fields: some labels contain commas, e.g. "decision layer (xgboost, calibrated)".
const parseCsvLine = (line) => {
  const cells = [];
  let cell = "";
  let quoted = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (quoted) {
      if (ch === '"' && line[i + 1] === '"') { cell += '"'; i++; }
      else if (ch === '"') quoted = false;
      else cell += ch;
    } else if (ch === '"') quoted = true;
    else if (ch === ",") { cells.push(cell); cell = ""; }
    else cell += ch;
  }
  cells.push(cell);
  return cells;
};

const readCsv = (rel) => {
  const p = path.join(ROOT, rel);
  if (!fs.existsSync(p)) { console.warn(`  missing ${rel}`); return null; }
  const lines = fs.readFileSync(p, "utf8").trim().split(/\r?\n/);
  const head = parseCsvLine(lines[0]);
  return lines.slice(1).map((line) => {
    const cells = parseCsvLine(line);
    if (cells.length !== head.length) {
      throw new Error(`${rel}: row has ${cells.length} cells, header has ${head.length}`);
    }
    return Object.fromEntries(head.map((h, i) => [h, cells[i]]));
  });
};

const D = {
  evaluation: readJson("reports/metrics/evaluation_summary.json"),
  models: readJson("reports/metrics/model_metrics.json"),
  split: readJson("reports/metrics/split_summary.json"),
  fusion: readJson("reports/metrics/fusion_summary.json"),
  fits: readJson("reports/metrics/base_model_fits.json"),
  spec: readJson("data/processed/feature_spec.json"),
  comparison: readCsv("reports/tables/model_comparison.csv"),
  tiersConfig: readCsv("reports/tables/risk_tiers_config.csv"),
  tiersDerived: readCsv("reports/tables/risk_tiers_derived.csv"),
  shap: readCsv("reports/tables/shap_global_importance.csv"),
  psi: readCsv("reports/tables/psi_score_drift.csv"),
  latency: readCsv("reports/tables/latency.csv"),
  ablation: readCsv("reports/ablation_label_delay/results.csv"),
  tuning: readCsv("reports/tuning/deep_tier_search.csv"),
  tuningBest: readJson("reports/tuning/best_config.json"),
  // Added with the product, robustness, transfer and tuning work.
  latencyProduct: readCsv("reports/tables/latency_product.csv"),
  groups: readCsv("reports/tables/shap_group_importance.csv"),
  faith: readCsv("reports/tables/explanation_faithfulness.csv"),
  reasons: readCsv("reports/tables/alert_reason_codes.csv"),
  robustness: readCsv("reports/tables/robustness.csv"),
  robustnessMeta: readJson("reports/metrics/robustness.json"),
  baf: readCsv("reports/tables/baf_transfer.csv"),
  bafTiers: readCsv("reports/tables/baf_tiers.csv"),
  bafMeta: readJson("reports/metrics/baf_transfer.json"),
  optuna: readJson("reports/tuned_run/tuned_params.json"),
  tunedComparison: readCsv("reports/tuned_run/model_comparison.csv"),
  handsetComparison: readCsv("reports/baseline_handset/model_comparison.csv"),
  handsetFusion: readJson("reports/baseline_handset/fusion_summary.json"),
  analytics: readJson("reports/metrics/analytics_summary.json"),
  edaQuality: readCsv("reports/tables/eda_data_quality.csv"),
  edaAmount: readCsv("reports/tables/eda_amount_by_class.csv"),
  edaSegments: readCsv("reports/tables/eda_fraud_by_segment.csv"),
  edaDaily: readCsv("reports/tables/eda_daily.csv"),
  statsCategorical: readCsv("reports/tables/stats_categorical_tests.csv"),
  statsNumeric: readCsv("reports/tables/stats_numeric_tests.csv"),
  bootstrap: readCsv("reports/tables/stats_bootstrap_ci.csv"),
  segments: readCsv("reports/tables/segment_performance.csv"),
};

const num = (v, dp = 4) => (v === undefined || v === null || v === "" || Number.isNaN(Number(v)))
  ? "-" : Number(v).toFixed(dp);
const int = (v) => (v === undefined || v === null || v === "") ? "-" : Number(v).toLocaleString("en-US");
const pct = (v, dp = 2) => (v === undefined || v === null) ? "-" : `${(Number(v) * 100).toFixed(dp)} %`;

/** Internal model keys read badly in a thesis table; give them proper names. */
const MODEL_LABELS = {
  logistic_regression: "Logistic regression",
  random_forest: "Random forest",
  xgboost: "XGBoost",
  lightgbm: "LightGBM",
  lstm: "LSTM (sequential)",
  tcn: "TCN (sequential)",
  hetero_gnn: "Heterogeneous GNN",
  tabpfn: "TabPFN v2",
  catboost: "CatBoost",
  blend_top2: "Blend of top two (logit mean)",
  blend_top3: "Blend of top three (logit mean)",
  stacked: "Stacked ensemble",
  "stacked+calibrated": "Stacked + calibrated",
};
const label = (key) => MODEL_LABELS[key] || String(key);

/* --------------------------------------------------------------- helpers -- */
const P = (text, opts = {}) => new Paragraph({
  alignment: opts.align || AlignmentType.JUSTIFIED,
  spacing: { after: opts.after ?? 160, line: opts.line ?? 300 },
  indent: opts.indent,
  children: [new TextRun({
    text, bold: opts.bold, italics: opts.italics,
    size: opts.size ?? 22, font: "Times New Roman",
  })],
});

const Runs = (runs, opts = {}) => new Paragraph({
  alignment: opts.align || AlignmentType.JUSTIFIED,
  spacing: { after: opts.after ?? 160, line: 300 },
  children: runs.map((r) => new TextRun({
    text: r.text, bold: r.bold, italics: r.italics,
    size: opts.size ?? 22, font: "Times New Roman",
  })),
});

const H1 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_1,
  keepNext: true,
  alignment: AlignmentType.CENTER,
  spacing: { before: 240, after: 240 },
  children: [new TextRun({ text, bold: true, size: 32, font: "Times New Roman" })],
});

const H2 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_2,
  keepNext: true,
  spacing: { before: 280, after: 140 },
  children: [new TextRun({ text, bold: true, size: 26, font: "Times New Roman" })],
});

const Bullets = (items) => items.map((t) => new Paragraph({
  numbering: { reference: "bullets", level: 0 },
  spacing: { after: 90, line: 280 },
  alignment: AlignmentType.JUSTIFIED,
  children: [new TextRun({ text: t, size: 22, font: "Times New Roman" })],
}));

const Caption = (text) => new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { before: 100, after: 240 },
  children: [new TextRun({ text, size: 20, italics: true, font: "Times New Roman" })],
});

const cell = (text, { bold = false, width, align = AlignmentType.LEFT, shade, keepNext = false } = {}) =>
  new TableCell({
    width: { size: width, type: WidthType.DXA },
    verticalAlign: VerticalAlign.CENTER,
    shading: shade ? { type: ShadingType.CLEAR, fill: shade, color: "auto" } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({
      alignment: align,
      keepNext,
      spacing: { after: 0, line: 260 },
      children: [new TextRun({ text: String(text), bold, size: 19, font: "Times New Roman" })],
    })],
  });

/** rows[0] is the header. `weights` are relative column widths. */
const T = (rows, weights, { numericFrom = 1 } = {}) => {
  const total = weights.reduce((a, b) => a + b, 0);
  const widths = weights.map((w) => Math.round((w / total) * CONTENT_WIDTH));
  widths[widths.length - 1] = CONTENT_WIDTH - widths.slice(0, -1).reduce((a, b) => a + b, 0);

  return new Table({
    columnWidths: widths,
    width: { size: CONTENT_WIDTH, type: WidthType.DXA },
    rows: rows.map((row, r) => new TableRow({
      tableHeader: r === 0,
      children: row.map((text, c) => cell(text, {
        bold: r === 0,
        width: widths[c],
        shade: r === 0 ? "E8E8E8" : undefined,
        keepNext: r === rows.length - 1,
        align: r === 0 ? AlignmentType.CENTER
          : (c >= numericFrom ? AlignmentType.CENTER : AlignmentType.LEFT),
      })),
    })),
  });
};

/** Reads intrinsic pixel dimensions so figures keep their aspect ratio. */
const imageSize = (buf) => {
  if (buf[0] === 0x89 && buf.toString("ascii", 1, 4) === "PNG") {
    return { w: buf.readUInt32BE(16), h: buf.readUInt32BE(20), type: "png" };
  }
  // Minimal JPEG SOF scan.
  let i = 2;
  while (i < buf.length) {
    if (buf[i] !== 0xff) { i += 1; continue; }
    const marker = buf[i + 1];
    if (marker >= 0xc0 && marker <= 0xcf && ![0xc4, 0xc8, 0xcc].includes(marker)) {
      return { h: buf.readUInt16BE(i + 5), w: buf.readUInt16BE(i + 7), type: "jpg" };
    }
    i += 2 + buf.readUInt16BE(i + 2);
  }
  return { w: FIG_MAX_PX, h: Math.round(FIG_MAX_PX * 0.6), type: "jpg" };
};

const Figure = (relPath, caption, maxPx = FIG_MAX_PX) => {
  const p = path.join(ROOT, relPath);
  if (!fs.existsSync(p)) {
    console.warn(`  missing figure ${relPath}`);
    return [P(`[FIGURE NOT GENERATED: ${relPath}]`, { align: AlignmentType.CENTER, bold: true }),
      Caption(caption)];
  }
  const buf = fs.readFileSync(p);
  const { w, h, type } = imageSize(buf);
  const width = Math.min(maxPx, w);
  const height = Math.round((h / w) * width);
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      keepNext: true,
      spacing: { before: 160, after: 0 },
      children: [new ImageRun({ data: buf, type, transformation: { width, height } })],
    }),
    Caption(caption),
  ];
};

const Break = () => new Paragraph({ children: [new PageBreak()] });

/* ------------------------------------------------------- derived numbers -- */
const modelByLabel = Object.fromEntries((D.models || []).map((m) => [m.label, m]));
const best = D.comparison ? D.comparison[0] : null;
const testPrev = modelByLabel.xgboost ? modelByLabel.xgboost.prevalence : 0.0344;
const tierRow = (rows, tier) => (rows || []).find((r) => r.tier === tier) || {};
const latRow = (mode) => (D.latency || []).find((r) => r.mode === mode) || {};
const abl = (variant, model) =>
  (D.ablation || []).find((r) => r.variant === variant && r.model === model) || {};

const worstPsi = (D.psi || []).reduce(
  (m, r) => Math.max(m, Number(r.psi_vs_window0 || 0)), 0);

const describeInputs = () => {
  console.log("building report from:");
console.log(`  models compared : ${(D.comparison || []).length}`);
console.log(`  best AUC-PR     : ${best ? best.auc_pr : "?"} (${best ? best.label : "?"})`);
console.log(`  ablation rows   : ${(D.ablation || []).length}`);
};
describeInputs();

/* ------------------------------------------------------------- document -- */
const titlePage = [
  P("FINANCIAL TRANSACTION FRAUD DETECTION AND RISK MODELLING",
    { align: AlignmentType.CENTER, bold: true, size: 32, after: 120 }),
  P("A hybrid graph-temporal learning framework with calibrated, explainable risk scoring",
    { align: AlignmentType.CENTER, italics: true, size: 24, after: 480 }),
  P("A Minor Project Report submitted to", { align: AlignmentType.CENTER, after: 0 }),
  P("Manipal Academy of Higher Education", { align: AlignmentType.CENTER, bold: true, after: 0 }),
  P("in partial fulfilment of the requirement for the award of the degree of",
    { align: AlignmentType.CENTER, after: 240 }),
  P("BACHELOR OF TECHNOLOGY", { align: AlignmentType.CENTER, bold: true, size: 26, after: 0 }),
  P("in", { align: AlignmentType.CENTER, after: 0 }),
  P("COMPUTER SCIENCE AND ENGINEERING",
    { align: AlignmentType.CENTER, bold: true, size: 26, after: 400 }),
  P("Submitted by", { align: AlignmentType.CENTER, after: 0 }),
  P("Nandurkar Shreyas Vivek  (235805446)",
    { align: AlignmentType.CENTER, bold: true, after: 300 }),
  P("Under the guidance of", { align: AlignmentType.CENTER, after: 0 }),
  P("Dr. Usha M", { align: AlignmentType.CENTER, bold: true, after: 0 }),
  P("Associate Professor, Department of Computer Science and Engineering",
    { align: AlignmentType.CENTER, after: 0 }),
  P("Manipal Institute of Technology, Bengaluru", { align: AlignmentType.CENTER, after: 600 }),
  P("DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING",
    { align: AlignmentType.CENTER, bold: true, after: 0 }),
  P("SEPTEMBER 2026", { align: AlignmentType.CENTER, bold: true, after: 0 }),
  Break(),
];

const certificate = [
  P("DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING",
    { align: AlignmentType.CENTER, bold: true, size: 26, after: 0 }),
  P("Bengaluru", { align: AlignmentType.CENTER, after: 400 }),
  H1("CERTIFICATE"),
  P("This is to certify that the minor project titled FINANCIAL TRANSACTION FRAUD " +
    "DETECTION AND RISK MODELLING is a record of the bonafide work done by KABIR CHHABRA " +
    "(Reg. No. 235805300), MAYANK (Reg. No. 235805360) and NANDURKAR SHREYAS VIVEK " +
    "(Reg. No. 235805446), submitted in partial fulfilment of the " +
    "requirements for the award of the Degree of Bachelor of Technology (B.Tech.) in " +
    "COMPUTER SCIENCE AND ENGINEERING of Manipal Institute of Technology Bengaluru, " +
    "Karnataka (A Constituent Institute of Manipal Academy of Higher Education), " +
    "during the academic year 2025-2026.", { after: 800 }),
  P("Dr. Usha M", { bold: true, after: 0 }),
  P("Associate Professor, CSE Dept.", { after: 0 }),
  P("M.I.T, BENGALURU", { after: 500 }),
  P("Dr. Dayananda P", { bold: true, after: 0 }),
  P("HOD, CSE Dept.", { after: 0 }),
  P("M.I.T, BENGALURU", { after: 0 }),
  Break(),
];

const acknowledgement = [
  H1("ACKNOWLEDGEMENT"),
  P("We express our sincere gratitude to our project guide, Dr. Usha M, Department of " +
    "Computer Science and Engineering, Manipal Institute of Technology Bengaluru, for her " +
    "guidance through every stage of this work. Her observations at the earlier reviews - " +
    "on the need for an explicit end-to-end workflow, on updating the modelling approach " +
    "to reflect current literature, and on presenting the survey findings in tabular form " +
    "- shaped the specification that this phase implements."),
  P("We thank the Head of the Department of Computer Science and Engineering and the " +
    "Director, Manipal Institute of Technology Bengaluru, for the academic environment and " +
    "the computational facilities that made this work possible. We also acknowledge the " +
    "faculty members and laboratory staff of the department whose assistance was sought " +
    "during the course of the project."),
  P("Kabir Chhabra (235805300)", { bold: true, after: 0 }),
  P("Mayank (235805360)", { bold: true, after: 0 }),
  P("Nandurkar Shreyas Vivek (235805446)", { bold: true, after: 0 }),
  Break(),
];

const abstract = [
  H1("ABSTRACT"),
  P("Digital payments have moved onto real-time settlement rails, and card-not-present " +
    "fraud has moved with them. Static rule engines and single tabular classifiers detect " +
    "previously observed patterns but degrade against organised fraud rings, evolving " +
    "attacker tactics and severe class imbalance, and they offer no defensible explanation " +
    "for the decisions they produce. A fraud system must therefore do two things that the " +
    "literature usually treats as one: decide whether a transaction is fraudulent, and " +
    "quantify how risky it is on a scale an operation can act upon."),
  P("This project follows the data-analytics process from the description of the data to " +
    "a prescribed action, implemented as a five-stage pipeline with a feedback loop. " +
    "Behavioural, " +
    "velocity and relational features are engineered under a strict causality rule, so that " +
    "no aggregate may be computed from a transaction later than the one being scored. Four " +
    "model families are benchmarked under one identical protocol: logistic regression and " +
    "random forest as baselines, XGBoost and LightGBM as the tabular tier, an LSTM over " +
    "per-entity transaction sequences, and a temporal heterogeneous graph neural network " +
    "with attention over the card-merchant-device graph. A decision layer chooses on " +
    "validation between a stacking meta-learner and simple averages of the base scores, " +
    "selecting the average of XGBoost and LightGBM; isotonic regression calibrates the " +
    "result, and the calibrated probability is mapped to " +
    "a 0-100 risk score with cost-sensitive operational tiers. Evaluation uses a strictly " +
    "chronological split of the IEEE-CIS Fraud Detection benchmark, comprising 590,540 " +
    "transactions at 3.50 per cent fraud prevalence over 182 days. Exploratory analysis and " +
    "statistical tests with effect sizes characterise the data before any model is fitted, " +
    "and the results are qualified with bootstrap confidence intervals and a breakdown by " +
    "segment."),
  P(`On the held-out final span of ${int(D.split?.test?.rows)} transactions, the strongest ` +
    `model attains an AUC-PR of ${num(best?.auc_pr)} against a no-skill floor equal to the ` +
    `prevalence of ${num(testPrev)}, a lift of ` +
    `${num((modelByLabel[best?.label] || modelByLabel.xgboost)?.auc_pr_lift_over_prevalence, 1)} times. Isotonic ` +
    `calibration reduces expected calibration error from ${num(D.fusion?.ece_before)} to ` +
    `${num(D.fusion?.ece_after)}. The high-risk tier isolates ` +
    `${pct(tierRow(D.tiersConfig, "high").share_of_all_fraud)} of all fraud in ` +
    `${pct(tierRow(D.tiersConfig, "high").share_of_volume)} of transaction volume at a ` +
    `precision of ${pct(tierRow(D.tiersConfig, "high").precision_in_tier)}, reducing modelled ` +
    `loss by ${num(D.evaluation?.cost?.config?.cost_reduction_pct, 1)} per cent against ` +
    `taking no action. A controlled ablation shows that withholding fraud outcomes for 30 ` +
    `days, as a real chargeback cycle would, lowers AUC-PR to ` +
    `${num(abl("delay_30", "xgboost").auc_pr)} - a sensitivity none of the surveyed studies ` +
    `reports. The system is delivered as a product that decides a single raw transaction ` +
    `in ${num(((D.latencyProduct || []).find((r) => r.mode === "single_transaction") || {}).p50_ms, 0)} ` +
    `ms at the median with plain-language reasons whose faithfulness is tested, and its ` +
    `robustness to evasion and its transfer to the Bank Account Fraud dataset are measured.`),
  P("The work concludes that a calibrated, cost-anchored decision layer is what converts a " +
    "classifier into a usable fraud control, and that the labelling assumptions embedded in " +
    "history-based features matter more to reported performance than the choice of model " +
    "architecture. The implementation uses Python with scikit-learn, XGBoost, LightGBM, " +
    "CatBoost, PyTorch, SHAP, FastAPI and Streamlit, and is released as a configuration-" +
    "driven, tested and containerised pipeline that reproduces every figure and table in " +
    "this report."),
  Break(),
];

const frontTables = [
  H1("TABLE OF CONTENTS"),
  new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }),
  Break(),
  H1("LIST OF TABLES"),
  T([
    ["Table", "Title"],
    ["1.1", "Project work schedule"],
    ["2.1", "Survey of recent work on financial fraud detection (2023-2026)"],
    ["2.2", "Performance metrics reported in the reviewed literature"],
    ["2.3", "Research gaps identified from the literature review"],
    ["3.1", "Data preparation and model development steps"],
    ["3.2", "Model stack and rationale for inclusion"],
    ["3.3", "Risk tiers, score bands and corresponding actions"],
    ["3.4", "Benchmark dataset specification"],
    ["3.5", "Evaluation metrics and their justification"],
    ["3.6", "Tools and technologies used"],
    ["4.1", "Chronological split of the benchmark"],
    ["4.2", "Composition of the engineered feature set"],
    ["4.3", "Data-quality profile of the training span by column family"],
    ["4.4", "Transaction amount by class, training span"],
    ["4.5", "Chi-square tests of categorical attributes against fraud"],
    ["4.6", "Mann-Whitney U tests of numeric attributes, fraud against legitimate"],
    ["4.7", "Model comparison on the held-out test span"],
    ["4.8", "Effect of isotonic calibration"],
    ["4.9", "Risk-tier performance under the specified score bands"],
    ["4.10", "Modelled cost of the decision policy"],
    ["4.11", "95 per cent confidence intervals for the test-span results (day-block bootstrap)"],
    ["4.12", "Detection performance by segment on the held-out test span"],
    ["4.13", "Attribution by evidence family over high-risk alerts"],
    ["4.14", "Explanation faithfulness: deletion test on high-risk alerts"],
    ["4.15", "Inference latency of the complete benchmark stack"],
    ["4.16", "Label-delay ablation"],
    ["4.17", "Further modelling experiments and the adoption decision"],
    ["4.18", "Evasion of the deployed model under attacker-controlled features"],
    ["4.19", "The same protocol on Bank Account Fraud (test month)"],
    ["4.20", "Fixed risk bands applied to Bank Account Fraud"],
    ["4.21", "Latency of the product path, from raw transaction to decision"],
  ], [1, 7], { numericFrom: 9 }),
  Break(),
  H1("LIST OF FIGURES"),
  T([
    ["Figure", "Title"],
    ["3.1", "Overall workflow of the proposed system"],
    ["3.2", "Layered architecture of the proposed system"],
    ["4.1", "Daily transaction volume and fraud rate across the timeline"],
    ["4.2", "Distribution of transaction amount, fraud against legitimate"],
    ["4.3", "Fraud rate and volume by hour of day and day of the weekly cycle"],
    ["4.4", "Fraud rate by segment"],
    ["4.5", "Correlation between amount, count (C) and time-delta (D) features"],
    ["4.6", "Concentration of fraud value across accounts"],
    ["4.7", "Strength of association with fraud for each attribute"],
    ["4.8", "Precision-recall curves on the test span"],
    ["4.9", "Reliability of the calibrated risk score"],
    ["4.10", "Risk-score distribution and tier boundaries"],
    ["4.11", "Detection by segment on the held-out test span"],
    ["4.12", "Global SHAP attribution over high-risk alerts"],
    ["4.13", "Score drift across the test timeline"],
  ], [1, 7], { numericFrom: 9 }),
  Break(),
];

module.exports = {
  D, P, Runs, H1, H2, Bullets, Caption, T, Figure, Break,
  num, int, pct, label, modelByLabel, best, testPrev, tierRow, latRow, abl, worstPsi,
  titlePage, certificate, acknowledgement, abstract, frontTables,
  CONTENT_WIDTH, OUT, ROOT,
};
