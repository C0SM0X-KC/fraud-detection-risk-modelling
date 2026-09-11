/**
 * Chapter bodies for the Phase-3 report.
 *
 * Chapters 1-3 carry forward the specification agreed at the earlier reviews.
 * Chapter 4 is generated from the metric files the pipeline writes, so the
 * document cannot drift away from the run that produced it. Chapter 5 draws
 * the conclusions those results support.
 *
 * Entry point is scripts/build_report.js.
 */

const R = require("./report_frontmatter.js");
const {
  D, P, H1, H2, Bullets, T, Figure, Break,
  num, int, pct, modelByLabel, best, testPrev, tierRow, latRow, abl, worstPsi, OUT,
} = R;

/* =================================================== CHAPTER 1 ============ */
const chapter1 = [
  H1("CHAPTER 1  INTRODUCTION"),
  P("This chapter introduces the problem of financial transaction fraud detection, " +
    "describes the present-day scenario that motivates the work, states the shortcomings " +
    "in existing approaches that the project sets out to address, lists the objectives, " +
    "and explains how the remainder of the report is organised."),

  H2("1.1  Introduction to the Area of Work"),
  P("Financial fraud detection is the task of deciding, for each incoming transaction, " +
    "whether it is legitimate or fraudulent, and doing so quickly enough that the decision " +
    "can still influence the outcome of the transaction. The problem sits at the " +
    "intersection of machine learning, data engineering and risk management, and it is " +
    "distinguished from a conventional classification problem by four properties."),
  ...Bullets([
    "The positive class is rare. In most realistic benchmarks fraudulent transactions " +
    "account for between one and four per cent of all records, so a model that predicts " +
    "the majority class for every input already scores above ninety-five per cent accuracy " +
    "while catching nothing at all.",
    "The cost of the two error types is asymmetric. A missed fraud costs the value of the " +
    "transaction; a false decline costs review effort and customer goodwill. The two are " +
    "not interchangeable and are not equal in magnitude.",
    "The data generating process is non-stationary. Attackers observe the deployed decision " +
    "boundary and adapt to it, so a model that performs well at deployment degrades over " +
    "the following months.",
    "The decision must be explainable. Financial institutions operate under model-governance " +
    "expectations that require a reason for every adverse action.",
  ]),
  P("Risk modelling is the complementary half of the problem. A classifier answers a binary " +
    "question, but an operational fraud system does not act on a binary answer. It needs a " +
    "graded, calibrated measure of how risky a transaction is, so that low-risk traffic can " +
    "clear without friction, borderline traffic can be challenged with an additional " +
    "authentication factor, and genuinely high-risk traffic can be held for a human " +
    "analyst. Converting a model output into such a measure requires calibration, threshold " +
    "selection driven by cost rather than convention, and an operational tier structure. " +
    "This project addresses both halves."),

  H2("1.2  Present-Day Scenario"),
  P("The state of practice has changed substantially between 2023 and 2026, and four " +
    "developments are directly relevant to this work."),
  ...Bullets([
    "Streaming detection has become routine. Stream-processing pipelines built on Apache " +
    "Kafka and Apache Flink now score card transactions at millisecond-level latency under " +
    "loads of the order of ten thousand transactions per second.",
    "Relational learning has displaced purely row-wise modelling for organised fraud. Graph " +
    "neural networks applied over graphs of cards, merchants, devices and addresses expose " +
    "coordinated multi-account behaviour that a tabular classifier examining one row at a " +
    "time cannot see.",
    "Tabular foundation models have emerged as a serious alternative to gradient boosting. " +
    "TabPFN v2, published in Nature in 2025, performs in-context inference over a learned " +
    "prior and matches or exceeds tuned gradient-boosted trees on small to medium tables in " +
    "a single forward pass.",
    "The open problems have shifted from raw accuracy to robustness and accountability. " +
    "Concept drift, adversarial robustness and explainability are now the recurring " +
    "limitations named in survey papers.",
  ]),

  H2("1.3  Motivation and Shortcomings in Existing Work"),
  P("The literature reviewed in Chapter 2 reports strong headline metrics, but three " +
    "recurring methodological choices make those metrics difficult to trust and impossible " +
    "to compare. Most studies split their data at random, which allows a card's future " +
    "behaviour to inform its own past score. Most report accuracy or AUC-ROC, both of which " +
    "are optimistic under severe imbalance. And most stop at classification, leaving the " +
    "conversion from a score into an action unspecified, which is precisely the part a " +
    "fraud operation consumes."),
  P("The motivation for this work is therefore not to propose a new architecture. It is to " +
    "build one pipeline in which several model families are compared under a single honest " +
    "protocol, in which the output is calibrated and priced against the actual costs of the " +
    "two error types, and in which the assumptions that inflate reported performance are " +
    "measured rather than left implicit."),

  H2("1.4  Objectives of the Work"),
  P("The objectives below are stated so that each is verifiable, and each addresses one or " +
    "more of the research gaps identified in Table 2.3."),
  ...Bullets([
    "To build a reproducible fraud detection pipeline on the IEEE-CIS Fraud Detection " +
    "benchmark, comprising 590,540 transactions at 3.50 per cent fraud prevalence, using a " +
    "strictly chronological train and test split so that no future information is available " +
    "at training time. (Addresses G1.)",
    "To engineer a feature set covering transaction velocity per card and per device, " +
    "rolling aggregate statistics of transaction amount, recency measures, device and " +
    "geolocation mismatch indicators, and to construct a heterogeneous card-merchant-device " +
    "transaction graph from the same data. (Addresses G2.)",
    "To benchmark five model families - classical classifiers, gradient boosting, " +
    "sequential models, a heterogeneous graph neural network and a tabular foundation " +
    "model - under one identical evaluation protocol. (Addresses G2.)",
    "To fuse the base model scores through a stacking meta-learner, calibrate the fused " +
    "output with isotonic regression, and convert the calibrated probability into a bounded " +
    "0-100 risk score with cost-sensitive Low, Medium and High tiers mapped to defined " +
    "operational actions. (Addresses G3.)",
    "To attach SHAP-based reason codes to every high-risk alert, so that each adverse " +
    "decision is auditable and can be explained to a reviewer. (Addresses G4.)",
    "To quantify concept drift using the Population Stability Index across successive " +
    "temporal windows, and to measure per-transaction inference latency at the median and " +
    "ninety-fifth percentile, establishing real-time feasibility rather than assuming it. " +
    "(Addresses G5 and G6.)",
  ]),

  H2("1.5  Project Work Schedule"),
  P("The project is carried out in phases in accordance with the minor specialization " +
    "mini-project guidelines, with a separate review presentation and report for each " +
    "phase. The earlier phases established the problem, surveyed the literature and " +
    "specified the workflow, architecture and evaluation protocol. The present report " +
    "covers the implementation of that specification and the analysis of its measured " +
    "results."),
  T([
    ["Phase", "Activity", "Outcome"],
    ["Phase 1", "Domain study, topic finalisation, literature survey of work published " +
      "between 2023 and 2026, statement of objectives", "Review presentation and report"],
    ["Phase 2", "Specification of the workflow, layered system architecture, model stack " +
      "and evaluation protocol", "Design report and synopsis"],
    ["Phase 3", "Implementation of the specified pipeline, experimental evaluation on the " +
      "full benchmark, and analysis of results", "This report and review presentation"],
    ["Phase 4", "Serving layer, analyst dashboard and project demonstration",
      "Final demonstration and viva"],
  ], [1.2, 5, 3], { numericFrom: 9 }),
  R.Caption("Table 1.1  Project work schedule"),

  H2("1.6  Organization of the Report"),
  P("Chapter 2 presents the background theory and the literature survey in tabular form, " +
    "together with the performance figures reported by the reviewed authors, and summarises " +
    "the outcome of that review as a set of research gaps. Chapter 3 presents the " +
    "methodology: the overall workflow, the step-wise data preparation and model " +
    "development procedure, the assumptions made, the layered system architecture, the " +
    "model stack, the risk modelling and decision layer, the dataset specification, the " +
    "evaluation metrics and the implementation. Chapter 4 analyses the results obtained on " +
    "the benchmark. Chapter 5 concludes and sets out the future scope. The references " +
    "follow in IEEE format."),
  Break(),
];

/* =================================================== CHAPTER 2 ============ */
const chapter2 = [
  H1("CHAPTER 2  BACKGROUND THEORY AND LITERATURE REVIEW"),
  P("This chapter introduces the project title in specific terms, sets out the background " +
    "theory underlying the techniques used, presents a survey of work published between " +
    "2023 and 2026 together with the performance the authors report, and converts the " +
    "outcome of that review into the set of research gaps this project addresses."),

  H2("2.1  Introduction to the Project Title"),
  P("The title of this work, Financial Transaction Fraud Detection and Risk Modelling, " +
    "names two deliverables. Fraud detection refers to the construction and comparative " +
    "evaluation of models that assign a fraud probability to an individual payment " +
    "transaction on the basis of its own attributes, the recent behaviour of the card or " +
    "account that produced it, and the position of that card or account in a wider " +
    "transaction graph. Risk modelling refers to the conversion of that probability into a " +
    "calibrated, bounded and interpretable risk score, and to the selection of thresholds " +
    "on that score which reflect the actual costs of the two error types and the review " +
    "capacity available to the institution. The literature is therefore examined along both " +
    "axes: what each study achieves in detection, and whether it produces anything usable " +
    "as a risk measure."),

  H2("2.2  Background Theory"),
  P("Four families of technique recur across the reviewed work. Ensemble methods over " +
    "decision trees, principally random forests, gradient boosting machines and their " +
    "modern implementations XGBoost and LightGBM, remain the strongest single family on " +
    "tabular transaction data because they capture higher-order interactions among " +
    "engineered features without requiring feature scaling. Stacking generalises this by " +
    "training a meta-learner on the out-of-fold predictions of several base models, so that " +
    "the strengths of models with different inductive biases combine."),
  P("Sequence models, principally long short-term memory networks and temporal " +
    "convolutional networks, treat a card's transaction history as an ordered sequence and " +
    "detect fraud as a break in that sequence rather than as a property of an isolated row. " +
    "Graph neural networks operate on a heterogeneous graph in which transactions, cards, " +
    "merchants and devices are distinct node types; message passing over this graph " +
    "propagates suspicion from a known fraudulent node to its neighbours, which is the " +
    "mechanism by which organised rings are exposed. Tabular foundation models such as " +
    "TabPFN v2 reframe supervised learning as in-context inference: the model is pretrained " +
    "on millions of synthetic tabular tasks and, at inference, conditions on the training " +
    "set supplied as context rather than updating any parameters."),
  P("Two auxiliary techniques are used throughout. Class imbalance is addressed either at " +
    "the data level, using synthetic minority oversampling such as SMOTE or ADASYN, or at " +
    "the algorithm level, using class weights and cost-sensitive loss functions. " +
    "Explainability is addressed principally through SHAP, which attributes a model's " +
    "output for an individual prediction to its input features using a game-theoretic " +
    "allocation, and which is therefore able to produce a per-alert reason code."),

  H2("2.3  Literature Survey"),
  P("Eight studies published between 2023 and 2026 were reviewed, six of them from 2024 " +
    "onwards. They were selected to span the four technique families described above and to " +
    "include both primary experimental work and recent systematic reviews."),
  T([
    ["Author / Year", "Technique", "Dataset", "Key Result", "Limitation"],
    ["Alhashmi et al. (2023) [1]", "Stacking ensemble of voting classifier, RF, GBM, LR, MLP",
      "Bank Account Fraud (1M records)", "Accuracy 0.98, precision 0.85, recall 0.90, F1 0.87",
      "Random split; no drift, latency or explainability study"],
    ["Motie and Raahemi (2024) [3]", "Systematic review of GNNs for financial fraud",
      "Multiple benchmarks", "Graph structure exposes organised fraud rings",
      "Scalability and real-time inference unresolved"],
    ["Zhao et al. (2024) [8]", "Improved LightGBM for extremely imbalanced data",
      "Credit-card transactions", "Strong recall at low computational cost",
      "Purely tabular; ignores relational signal"],
    ["Alarfaj and Shahzadi (2024) [7]", "GNN combined with autoencoders", "Real-time card data",
      "Improved real-time fraud prevention", "Sensitive to the reconstruction threshold chosen"],
    ["Hollmann et al. (2025) [4]", "TabPFN v2 tabular foundation model",
      "Broad tabular benchmark suite", "Exceeds tuned GBDTs in a single forward pass",
      "Practical limit near 10,000 rows; untested at fraud scale"],
    ["Sha et al. (2025) [5]", "Heterogeneous GNN with graph attention", "IEEE-CIS Fraud Detection",
      "Outperforms GCN, GAT and GraphSAGE on AUC-ROC", "High inference cost; static graph snapshot"],
    ["Hafez et al. (2025) [6]", "Systematic review of AI in card fraud detection", "2015-2024 corpus",
      "Identifies XAI and drift as the top open problems", "Review only; no new model contributed"],
    ["Pandey et al. (2026) [2]", "Thematic synthesis of AI fraud frameworks",
      "Eight peer-reviewed sources", "Kafka-Flink under 50 ms; neuro-symbolic hybrid 98.5 % F1",
      "Secondary research; no implementation or validation"],
  ], [2.2, 2.6, 2.0, 2.6, 2.6], { numericFrom: 99 }),
  R.Caption("Table 2.1  Survey of recent work on financial fraud detection (2023-2026)"),

  H2("2.4  Reported Performance of Existing Models"),
  P("Table 2.2 collects the quantitative results reported by the reviewed authors. One " +
    "caveat must be stated explicitly: the values are reported on different datasets, under " +
    "different train and test protocols, and against different metrics, so they are not " +
    "directly comparable across rows. This incomparability is itself a motivation for the " +
    "present work, in which every model family is re-benchmarked under one identical " +
    "protocol."),
  T([
    ["Model / Technique", "Dataset / Context", "Metric", "Value"],
    ["Logistic Regression", "Bank Account Fraud", "F1-score", "0.68"],
    ["Random Forest", "Bank Account Fraud", "F1-score", "0.77"],
    ["Gradient Boosting", "Bank Account Fraud", "F1-score", "0.73"],
    ["Neural Network (MLP)", "Bank Account Fraud", "F1-score", "0.73"],
    ["Voting Classifier", "Bank Account Fraud", "F1-score", "0.83"],
    ["Stacking Ensemble", "Bank Account Fraud", "F1-score", "0.87"],
    ["Random Forests / SVM", "Institutional datasets", "Precision", "95 %"],
    ["XGBoost + Neural Network", "Banking simulations", "F1-score", "0.97"],
    ["LSTM", "Card transaction streams", "RMSE", "0.12"],
    ["Autoencoder", "Synthetic fraud datasets", "Recall", "98 %"],
    ["Neuro-symbolic + GraphSAGE", "Banking hybrids", "F1-score", "98.5 %"],
    ["Any model under PGD attack", "Adversarial testing", "Accuracy drop", "~20 %"],
  ], [3, 3, 2, 1.4]),
  R.Caption("Table 2.2  Performance metrics reported in the reviewed literature " +
    "(compiled from [1] and [2])"),
  P("The pattern these figures expose is the central methodological point of this project. " +
    "Across the six models evaluated by Alhashmi et al. [1], accuracy is essentially flat, " +
    "varying only between 0.96 and 0.98 including for the weakest model, while the F1-score " +
    "over the same six models ranges from 0.68 to 0.87. A metric that does not separate a " +
    "poor model from a strong one is not a useful basis for comparison. This is why the " +
    "protocol adopted in Chapter 3 is anchored on AUC-PR and recall at a fixed " +
    "false-positive rate."),

  H2("2.5  Summarised Outcome of the Literature Review"),
  P("The review supports three positive conclusions. Ensemble methods, and stacking in " +
    "particular, consistently outperform any single constituent model. Relational and " +
    "sequential signal carries information that row-wise tabular models cannot access. " +
    "Streaming infrastructure has made real-time scoring practical at production " +
    "transaction volumes. It also exposes six gaps, which together define the scope of this " +
    "work."),
  T([
    ["Gap", "Description", "Why it matters"],
    ["G1", "Evaluation without time",
      "Most studies use random train and test splits, which leak future information into " +
      "training. Fraud is a temporal process, and performance measured under a " +
      "chronological split is materially lower and more honest."],
    ["G2", "Tabular-only modelling",
      "Row-wise classifiers cannot observe the card-merchant-device graph, so coordinated " +
      "multi-account fraud rings remain invisible to them."],
    ["G3", "Detection treated as the end point",
      "Papers report classification metrics but rarely produce a calibrated, actionable " +
      "risk score with defined operational tiers, which is what a fraud operation consumes."],
    ["G4", "Opaque decisions",
      "Ensembles and deep models supply no reason code, which conflicts with " +
      "model-governance and audit expectations in banking."],
    ["G5", "Drift and adversarial decay",
      "Concept drift and adversarial perturbation are acknowledged in discussion sections " +
      "but seldom measured."],
    ["G6", "Unreported cost of serving",
      "Inference latency and throughput are rarely measured, so claims of real-time " +
      "feasibility remain unverified."],
  ], [0.8, 2.6, 6.4], { numericFrom: 99 }),
  R.Caption("Table 2.3  Research gaps identified from the literature review"),
  Break(),
];



module.exports = { chapter1, chapter2 };
