"""Data analytics: exploratory analysis, statistical tests and performance by segment.

    python -m frds analytics
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score

from . import plots
from .config import Config
from .data import load as data_load
from .data.split import load_split
from .features.build import entity_key_text
from .fusion.risk import policy_from_config, risk_score
from .models.train import load_scores
from .utils import get_logger, read_json, set_seed, timed, write_json

log = get_logger("frds.analytics")

SECONDS_PER_DAY = 86_400

# Column families of the merged IEEE-CIS frame, for the data-quality profile.
FAMILIES = [
    ("Card", "card attributes card1-card6", r"card\d"),
    ("Address", "billing region and country addr1, addr2", r"addr\d"),
    ("Distance", "address distances dist1, dist2", r"dist\d"),
    ("E-mail", "purchaser and recipient e-mail domains", r"[PR]_emaildomain"),
    ("C (counts)", "Vesta counting features C1-C14", r"C\d+"),
    ("D (time deltas)", "Vesta time-delta features D1-D15", r"D\d+"),
    ("M (match flags)", "Vesta match flags M1-M9", r"M\d"),
    ("V (Vesta)", "Vesta engineered features V1-V339", r"V\d+"),
    ("Identity", "identity table id_01-id_38, DeviceType, DeviceInfo",
     r"id_\d+|DeviceType|DeviceInfo"),
]

CATEGORICAL_TESTS = [
    ("ProductCD", "Product code"),
    ("card4", "Card network"),
    ("card6", "Card type"),
    ("DeviceType", "Device type"),
    ("P_emaildomain", "Purchaser e-mail domain"),
    ("R_emaildomain", "Recipient e-mail domain"),
    ("M4", "Match flag M4"),
    ("M6", "Match flag M6"),
    ("has_identity", "Identity record present"),
    ("hour", "Hour of day"),
    ("weekday", "Day of the weekly cycle"),
    ("amount_band", "Amount band"),
]

NUMERIC_TESTS = [
    ("TransactionAmt", "Transaction amount"),
    ("dist1", "Distance between addresses (dist1)"),
    ("C1", "Count feature C1"),
    ("C2", "Count feature C2"),
    ("C5", "Count feature C5"),
    ("C13", "Count feature C13"),
    ("C14", "Count feature C14"),
    ("D1", "Time delta D1"),
    ("D4", "Time delta D4"),
    ("D10", "Time delta D10"),
    ("D15", "Time delta D15"),
    ("id_02", "Identity feature id_02"),
]

CORRELATION_COLUMNS = ["TransactionAmt", "dist1"] + [f"C{i}" for i in range(1, 15)] + [
    "D1", "D2", "D3", "D4", "D10", "D11", "D15"]

AMOUNT_BANDS = [0, 25, 50, 100, 250, 500, np.inf]
AMOUNT_LABELS = ["under 25", "25-50", "50-100", "100-250", "250-500", "500+"]
HISTORY_BANDS = [-1, 0, 4, 19, np.inf]
HISTORY_LABELS = ["first seen", "1-4 earlier", "5-19 earlier", "20+ earlier"]
SEGMENT_DIMENSIONS = [
    ("account_history", "Account history"),
    ("amount_band", "Amount band"),
    ("ProductCD", "Product code"),
    ("card6", "Card type"),
    ("card4", "Card network"),
    ("DeviceType", "Device type"),
    ("email", "Purchaser e-mail domain"),
]


# ------------------------------------------------------------- helpers ----
def wilson_interval(k, n, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    """95% Wilson interval for a proportion k/n (stays inside 0-1 for rare events)."""
    k = np.asarray(k, dtype=np.float64)
    n = np.maximum(np.asarray(n, dtype=np.float64), 1.0)
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return np.clip(centre - half, 0, 1), np.clip(centre + half, 0, 1)


def strength(effect: float) -> str:
    """Conventional reading of Cramer's V (binary outcome) and rank-biserial r."""
    a = abs(effect)
    if not np.isfinite(a):
        return "-"
    return "negligible" if a < 0.1 else "small" if a < 0.3 else "medium" if a < 0.5 else "large"


def levels(series: pd.Series, top: int | None = None, min_count: int = 100) -> pd.Series:
    """Categorical levels as text: missing is its own level, rare levels pool as other."""
    text = series.astype("object").where(series.notna(), "missing").astype(str)
    counts = text.value_counts()
    keep = set(counts.index[counts >= min_count])
    if top is not None:
        ranked = [v for v in counts.index if v != "missing"][:top]
        keep &= set(ranked) | {"missing"}
    return text.where(text.isin(keep), "other")


def rate_table(frame: pd.DataFrame, column: str, dimension: str) -> pd.DataFrame:
    """Volume, fraud count and fraud rate (with a 95% interval) for each level."""
    overall = float(frame["isFraud"].mean())
    grouped = frame.groupby(column, observed=True)["isFraud"].agg(["size", "sum"])
    grouped = grouped.rename(columns={"size": "transactions", "sum": "fraud"})
    lo, hi = wilson_interval(grouped["fraud"], grouped["transactions"])
    out = pd.DataFrame({
        "dimension": dimension,
        "level": grouped.index.astype(str),
        "transactions": grouped["transactions"].to_numpy(),
        "share_of_volume": grouped["transactions"].to_numpy() / len(frame),
        "fraud": grouped["fraud"].to_numpy(),
        "fraud_rate": (grouped["fraud"] / grouped["transactions"]).to_numpy(),
        "ci_low": lo, "ci_high": hi,
    })
    out["lift"] = out["fraud_rate"] / overall
    return out.reset_index(drop=True)


# ------------------------------------------------------- the analysis frame
def analysis_frame(df: pd.DataFrame, cfg: Config, split) -> pd.DataFrame:
    """One slim row per transaction with the derived fields every analysis uses."""
    t = df[cfg.require("data.time_col")].to_numpy(np.int64)
    amount = df[cfg.require("data.amount_col")].astype(np.float64)
    frame = pd.DataFrame({
        "day": t // SECONDS_PER_DAY,
        "hour": (t // 3600) % 24,
        "weekday": (t // SECONDS_PER_DAY) % 7,
        "TransactionAmt": amount.to_numpy(),
        "isFraud": df[cfg.require("data.target_col")].astype(np.int8).to_numpy(),
    })
    span = np.full(len(df), "train", dtype=object)
    span[split.valid] = "validation"
    span[split.test] = "test"
    frame["split"] = span

    identity_cols = [c for c in df.columns if re.fullmatch(r"id_\d+|DeviceType|DeviceInfo", c)]
    frame["has_identity"] = np.where(df[identity_cols].notna().any(axis=1), "yes", "no")
    frame["amount_band"] = pd.cut(amount, AMOUNT_BANDS, right=False,
                                  labels=AMOUNT_LABELS).astype(str).to_numpy()

    train_mask = frame["split"].eq("train").to_numpy()
    for col in ("ProductCD", "card4", "card6", "DeviceType", "M4", "M6"):
        frame[col] = levels(df[col]).to_numpy()
    for col in ("P_emaildomain", "R_emaildomain"):
        frame[col] = levels(df[col], top=15).to_numpy()
    # The six purchaser domains with the most training volume, for segments and filters.
    top_email = df.loc[train_mask, "P_emaildomain"].value_counts().index[:6]
    email = df["P_emaildomain"].astype("object")
    frame["email"] = np.where(email.isna(), "missing",
                              np.where(email.isin(top_email), email, "other"))

    for col, _ in NUMERIC_TESTS:
        frame[col] = df[col].astype(np.float64).to_numpy()
    for col in CORRELATION_COLUMNS:
        if col not in frame:
            frame[col] = df[col].astype(np.float64).to_numpy()

    # The synthetic account the features use (card1 + addr1 + e-mail + first-seen day).
    uid, _ = pd.factorize(entity_key_text(df, cfg)["uid"])
    frame["uid"] = uid
    prior = frame.groupby("uid").cumcount()
    frame["account_history"] = pd.cut(prior, HISTORY_BANDS, labels=HISTORY_LABELS).astype(str)
    return frame


# ------------------------------------------------------------ descriptive
def data_quality(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    missing = df.isna().mean()
    for name, description, pattern in FAMILIES:
        cols = [c for c in df.columns if re.fullmatch(pattern, c)]
        share = missing[cols]
        rows.append({
            "family": name, "description": description, "columns": len(cols),
            "mean_missing_pct": 100 * float(share.mean()),
            "max_missing_pct": 100 * float(share.max()),
            "columns_over_90pct_missing": int((share > 0.9).sum()),
            "columns_complete": int((share == 0).sum()),
        })
    return pd.DataFrame(rows)


def amount_by_class(train: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, part in (("legitimate", train[train.isFraud == 0]),
                        ("fraud", train[train.isFraud == 1])):
        a = part["TransactionAmt"]
        rows.append({
            "class": label, "transactions": len(a), "mean": a.mean(), "median": a.median(),
            "p25": a.quantile(0.25), "p75": a.quantile(0.75), "p90": a.quantile(0.90),
            "p99": a.quantile(0.99), "max": a.max(), "total_value": a.sum(),
        })
    return pd.DataFrame(rows)


def daily_series(frame: pd.DataFrame) -> pd.DataFrame:
    daily = frame.groupby("day").agg(
        transactions=("isFraud", "size"), fraud=("isFraud", "sum"),
        value=("TransactionAmt", "sum"), split=("split", "first"))
    daily["fraud_rate"] = daily["fraud"] / daily["transactions"]
    return daily.reset_index()


def fraud_concentration(train: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Pareto view of fraud value by account, and how often fraud repeats on one."""
    fraud = train[train.isFraud == 1]
    value = fraud.groupby("uid")["TransactionAmt"].sum().sort_values(ascending=False)
    curve = pd.DataFrame({
        "share_of_fraud_accounts": np.arange(1, len(value) + 1) / len(value),
        "share_of_fraud_value": value.cumsum().to_numpy() / value.sum(),
    })

    def accounts_for(share: float) -> float:
        k = int(np.searchsorted(curve["share_of_fraud_value"].to_numpy(), share))
        return float(curve["share_of_fraud_accounts"].iloc[min(k, len(curve) - 1)])

    prior_fraud = train.groupby("uid")["isFraud"].cumsum() - train["isFraud"]
    repeat = float((prior_fraud[train.isFraud == 1] > 0).mean())
    accounts = int(train["uid"].nunique())
    summary = {
        "accounts": accounts,
        "accounts_with_fraud": int(len(value)),
        "share_of_accounts_with_fraud": len(value) / accounts,
        "fraud_accounts_for_50pct_value": accounts_for(0.5),
        "fraud_accounts_for_80pct_value": accounts_for(0.8),
        "top_10pct_fraud_accounts_value_share":
            float(curve["share_of_fraud_value"].iloc[max(int(0.1 * len(curve)) - 1, 0)]),
        "fraud_on_account_with_earlier_fraud": repeat,
    }
    return curve, summary


# ------------------------------------------------------------ inferential
def categorical_tests(train: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, label in CATEGORICAL_TESTS:
        table = pd.crosstab(train[col], train["isFraud"])
        chi2, p, dof, _ = stats.chi2_contingency(table, correction=False)
        v = float(np.sqrt(chi2 / table.to_numpy().sum()))
        rates = table[1] / table.sum(axis=1)
        rows.append({
            "variable": label, "column": col, "levels": len(table), "chi2": chi2, "dof": dof,
            "p_value": p, "cramers_v": v, "strength": strength(v),
            "highest_rate_level": str(rates.idxmax()), "highest_rate": float(rates.max()),
        })
    out = pd.DataFrame(rows)
    out["p_bonferroni"] = np.minimum(out["p_value"] * len(out), 1.0)
    return out.sort_values("cramers_v", ascending=False).reset_index(drop=True)


def numeric_tests(train: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, label in NUMERIC_TESTS:
        fraud = train.loc[train.isFraud == 1, col].dropna().to_numpy()
        legit = train.loc[train.isFraud == 0, col].dropna().to_numpy()
        u, p = stats.mannwhitneyu(fraud, legit, alternative="two-sided", method="asymptotic")
        auc = float(u) / (len(fraud) * len(legit))
        r = 2 * auc - 1
        rows.append({
            "variable": label, "column": col,
            "available_pct": 100 * float(train[col].notna().mean()),
            "median_legitimate": float(np.median(legit)), "median_fraud": float(np.median(fraud)),
            "u_statistic": float(u), "p_value": float(p),
            "prob_fraud_higher": auc, "rank_biserial": r, "strength": strength(r),
        })
    out = pd.DataFrame(rows)
    out["p_bonferroni"] = np.minimum(out["p_value"] * len(out), 1.0)
    return out.reindex(out["rank_biserial"].abs().sort_values(ascending=False).index) \
        .reset_index(drop=True)


def bootstrap_intervals(y: np.ndarray, score_map: dict[str, np.ndarray], tiers: np.ndarray,
                        days: np.ndarray, reference: tuple[str, str], n_boot: int,
                        seed: int) -> tuple[pd.DataFrame, dict]:
    """95% intervals by resampling whole days, since one day's transactions are related."""
    rng = np.random.default_rng(seed)
    day_rows = [np.flatnonzero(days == d) for d in np.unique(days)]
    flagged = tiers != "low"
    high = tiers == "high"

    def measures(idx: np.ndarray) -> dict[str, float]:
        yy = y[idx]
        out = {f"AUC-PR | {name}": average_precision_score(yy, s[idx])
               for name, s in score_map.items()}
        fraud = yy == 1
        out["High tier: share of fraud caught"] = high[idx][fraud].mean()
        out["High tier: precision"] = yy[high[idx]].mean()
        out["Medium + high: share of fraud caught"] = flagged[idx][fraud].mean()
        out["Medium + high: false-positive rate"] = flagged[idx][~fraud].mean()
        return out

    point = measures(np.arange(len(y)))
    draws = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(day_rows), len(day_rows))
        draws.append(measures(np.concatenate([day_rows[k] for k in pick])))
    draws = pd.DataFrame(draws)

    rows = [{"measure": m, "estimate": point[m],
             "ci_low": float(draws[m].quantile(0.025)),
             "ci_high": float(draws[m].quantile(0.975))} for m in point]
    a, b = (f"AUC-PR | {name}" for name in reference)
    diff = draws[a] - draws[b]
    comparison = {
        "better": reference[0], "baseline": reference[1],
        "difference": point[a] - point[b],
        "ci_low": float(diff.quantile(0.025)), "ci_high": float(diff.quantile(0.975)),
        "share_of_resamples_better": float((diff > 0).mean()),
        "resamples": n_boot, "days": len(day_rows),
    }
    rows.append({"measure": f"AUC-PR difference | {reference[0]} minus {reference[1]}",
                 "estimate": comparison["difference"], "ci_low": comparison["ci_low"],
                 "ci_high": comparison["ci_high"]})
    return pd.DataFrame(rows), comparison


def segment_performance(test: pd.DataFrame, y: np.ndarray, p: np.ndarray,
                        tiers: np.ndarray) -> pd.DataFrame:
    rows = []
    amount = test["TransactionAmt"].to_numpy()
    for col, label in SEGMENT_DIMENSIONS:
        values = test[col].to_numpy()
        for level in pd.Series(values).value_counts().index:
            m = values == level
            yy, tt, fraud = y[m], tiers[m], y[m] == 1
            flagged = tt != "low"
            n_fraud = int(fraud.sum())
            rows.append({
                "dimension": label, "segment": str(level), "transactions": int(m.sum()),
                "share_of_volume": float(m.mean()), "fraud": n_fraud,
                "fraud_rate": float(yy.mean()),
                "auc_pr": (float(average_precision_score(yy, p[m]))
                           if n_fraud >= 20 and n_fraud < m.sum() else np.nan),
                "fraud_caught_high": float((tt[fraud] == "high").mean()) if n_fraud else np.nan,
                "precision_high": float(yy[tt == "high"].mean()) if (tt == "high").any() else np.nan,
                "fraud_caught_flagged": float(flagged[fraud].mean()) if n_fraud else np.nan,
                "false_positive_rate": float(flagged[~fraud].mean()) if (~fraud).any() else np.nan,
                "fraud_value_caught": (float(amount[m][fraud & flagged].sum()
                                             / amount[m][fraud].sum()) if n_fraud else np.nan),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- stage ----
def run(cfg: Config, n_boot: int = 1000) -> dict:
    seed = int(cfg.require("project.seed"))
    set_seed(seed)
    with timed("analytics", log):
        tables = cfg.path("reports") / "tables"
        figures = cfg.path("reports") / "figures"
        df = data_load.load_prepared(cfg)
        split = load_split(cfg)

        # Exploratory analysis and tests use the training span only, so nothing
        # here looks at the periods used to choose or to report the model.
        quality = data_quality(df.iloc[split.train])
        frame = analysis_frame(df, cfg, split)
        n_columns = int(df.shape[1])
        del df
        train = frame.iloc[split.train]
        overall = float(train["isFraud"].mean())

        quality.to_csv(tables / "eda_data_quality.csv", index=False)
        amounts = amount_by_class(train)
        amounts.to_csv(tables / "eda_amount_by_class.csv", index=False)
        daily = daily_series(frame)
        daily.to_csv(tables / "eda_daily.csv", index=False)

        rates = pd.concat([rate_table(train, col, label) for col, label in [
            ("ProductCD", "Product code"), ("card4", "Card network"), ("card6", "Card type"),
            ("DeviceType", "Device type"), ("email", "Purchaser e-mail domain"),
            ("has_identity", "Identity record present"), ("amount_band", "Amount band"),
        ]], ignore_index=True)
        rates.to_csv(tables / "eda_fraud_by_segment.csv", index=False)
        hourly = rate_table(train, "hour", "Hour of day")
        weekly = rate_table(train, "weekday", "Day of the weekly cycle")
        pd.concat([hourly, weekly]).to_csv(tables / "eda_time_cycles.csv", index=False)

        sample = train[CORRELATION_COLUMNS].sample(min(len(train), 100_000), random_state=seed)
        corr = sample.corr(method="spearman")
        corr.to_csv(tables / "eda_correlation.csv")
        upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack().dropna()

        curve, concentration = fraud_concentration(train)

        cat = categorical_tests(train)
        cat.to_csv(tables / "stats_categorical_tests.csv", index=False)
        num = numeric_tests(train)
        num.to_csv(tables / "stats_numeric_tests.csv", index=False)

        plots.eda_timeline(daily, figures / "eda_timeline.png")
        plots.eda_missingness(quality, figures / "eda_missingness.png")
        plots.eda_amount(train, figures / "eda_amount.png")
        plots.eda_time_cycles(hourly, weekly, overall, figures / "eda_time_cycles.png")
        plots.eda_categories(rates, overall, figures / "eda_categories.png")
        plots.eda_correlation(corr, figures / "eda_correlation.png")
        plots.eda_pareto(curve, figures / "eda_pareto.png")
        plots.effect_sizes(cat, num, figures / "stats_effect_sizes.png")

        # -- test span: how reliable the results are, and where the system is weak
        scores = load_scores(cfg)
        fused = np.load(cfg.path("artifacts") / "fused_scores.npz")
        fusion = read_json(cfg.path("reports") / "metrics" / "fusion_summary.json")
        test_pos = scores["test_positions"]
        test = frame.iloc[test_pos].reset_index(drop=True)
        y = scores["y_test"].astype(int)
        p_cal = fused["p_test_calibrated"]
        risk = risk_score(p_cal, cfg)
        tiers = policy_from_config(cfg).assign(risk)

        selected = fusion.get("selected_score", "stacked")
        members = [m.replace("xgboost", "XGBoost").replace("lightgbm", "LightGBM")
                   for m in fusion.get("blend_members", [])]
        selected_label = f"Blend ({' + '.join(members)})" if members else selected
        score_map = {selected_label: fused["p_test_raw"],
                     "XGBoost": scores["test__xgboost"], "LightGBM": scores["test__lightgbm"],
                     "Decision layer (calibrated)": p_cal}
        boot, comparison = bootstrap_intervals(
            y, score_map, tiers, test["day"].to_numpy(), (selected_label, "XGBoost"),
            n_boot, seed)
        boot.to_csv(tables / "stats_bootstrap_ci.csv", index=False)

        segments = segment_performance(test, y, p_cal, tiers)
        segments.to_csv(tables / "segment_performance.csv", index=False)
        plots.segment_detection(segments, figures / "segment_detection.png")

        # The console's analytics page filters this file; derived from the
        # competition data, so it stays under data/ (not redistributed).
        slim = frame[["day", "hour", "weekday", "split", "TransactionAmt", "ProductCD",
                      "card4", "card6", "DeviceType", "email", "has_identity",
                      "amount_band", "account_history", "isFraud"]].copy()
        slim["risk_score"] = np.int16(-1)
        slim.loc[test_pos, "risk_score"] = risk
        slim["tier"] = ""
        slim.loc[test_pos, "tier"] = tiers
        for col in ("split", "ProductCD", "card4", "card6", "DeviceType", "email",
                    "has_identity", "amount_band", "account_history", "tier"):
            slim[col] = slim[col].astype("category")
        slim.to_parquet(cfg.path("processed") / "analytics_transactions.parquet", index=False)

        identity = rates[rates.dimension == "Identity record present"].set_index("level")
        legit, fraud = amounts.set_index("class").loc["legitimate"], \
            amounts.set_index("class").loc["fraud"]
        summary = {
            "train_rows": int(len(train)), "train_fraud": int(train["isFraud"].sum()),
            "train_prevalence": overall, "columns": n_columns,
            "columns_over_90pct_missing": int(quality["columns_over_90pct_missing"].sum()),
            "identity_coverage": float(identity.loc["yes", "share_of_volume"]),
            "fraud_rate_with_identity": float(identity.loc["yes", "fraud_rate"]),
            "fraud_rate_without_identity": float(identity.loc["no", "fraud_rate"]),
            "median_amount_legitimate": float(legit["median"]),
            "median_amount_fraud": float(fraud["median"]),
            "mean_amount_legitimate": float(legit["mean"]),
            "mean_amount_fraud": float(fraud["mean"]),
            "peak_fraud_hour": int(hourly.loc[hourly.fraud_rate.idxmax(), "level"]),
            "peak_fraud_hour_rate": float(hourly.fraud_rate.max()),
            "lowest_fraud_hour": int(hourly.loc[hourly.fraud_rate.idxmin(), "level"]),
            "lowest_fraud_hour_rate": float(hourly.fraud_rate.min()),
            "busiest_hour": int(hourly.loc[hourly.transactions.idxmax(), "level"]),
            "quietest_hour": int(hourly.loc[hourly.transactions.idxmin(), "level"]),
            "correlation_pairs": int(len(upper)),
            "correlation_pairs_above_0_8": int((upper.abs() > 0.8).sum()),
            "strongest_correlations": [
                {"pair": f"{a} and {b}", "rho": float(v)}
                for (a, b), v in upper.abs().sort_values(ascending=False).head(3).items()],
            "concentration": concentration,
            "bootstrap": comparison,
            "test_rows": int(len(test)),
        }
        write_json(summary, cfg.path("reports") / "metrics" / "analytics_summary.json")
        log.info("categorical tests\n%s", cat[["variable", "cramers_v", "strength"]]
                 .to_string(index=False))
        log.info("numeric tests\n%s", num[["variable", "rank_biserial", "strength"]]
                 .to_string(index=False))
        log.info("bootstrap\n%s", boot.to_string(index=False))
        return summary
