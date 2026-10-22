"""Governance layer (L5): adversarial robustness (report Table 4.2, robustness tier)."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ..config import Config
from ..data.split import load_split
from ..features.build import load_features
from ..product import DeployedModel
from ..utils import get_logger, set_seed, write_json

log = get_logger("frds.evaluation.robustness")


def mutable_columns(columns: list[str], patterns: list[str]) -> list[str]:
    compiled = [re.compile(p) for p in patterns]
    return [c for c in columns if any(p.search(c) for p in compiled)]


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class _Scaler:
    """Median imputation and standardisation, fitted on training rows."""

    def __init__(self, X: pd.DataFrame) -> None:
        values = X.to_numpy(np.float64)
        self.median = np.nan_to_num(np.nanmedian(values, axis=0))
        filled = np.where(np.isnan(values), self.median, values)
        self.mean = filled.mean(axis=0)
        self.std = filled.std(axis=0)
        self.std[self.std < 1e-9] = 1.0
        self.lo = (np.nanmin(np.where(np.isnan(values), np.inf, values), axis=0) - self.mean) / self.std
        self.hi = (np.nanmax(np.where(np.isnan(values), -np.inf, values), axis=0) - self.mean) / self.std
        self.integral = np.all(np.isnan(values) | (values == np.round(values)), axis=0)

    def to_z(self, X: pd.DataFrame) -> np.ndarray:
        values = X.to_numpy(np.float64)
        return (np.where(np.isnan(values), self.median, values) - self.mean) / self.std

    def from_z(self, z: np.ndarray, original: pd.DataFrame) -> pd.DataFrame:
        values = z * self.std + self.mean
        values[:, self.integral] = np.round(values[:, self.integral])
        raw = original.to_numpy(np.float64)
        values = np.where(np.isnan(raw), np.nan, values)      # missing stays missing
        frame = pd.DataFrame(values, columns=original.columns, index=original.index)
        return frame.astype(original.dtypes.to_dict(), errors="ignore")


def _surrogate(z: np.ndarray, target: np.ndarray, seed: int, epochs: int = 6):
    import torch
    from torch import nn

    torch.manual_seed(seed)
    net = nn.Sequential(
        nn.Linear(z.shape[1], 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, 1),
    )
    optimiser = torch.optim.Adam(net.parameters(), lr=1e-3)
    zt = torch.tensor(z, dtype=torch.float32)
    tt = torch.tensor(target, dtype=torch.float32).unsqueeze(1)
    n = len(zt)
    for epoch in range(epochs):
        order = torch.randperm(n)
        total = 0.0
        for k in range(0, n, 2048):
            idx = order[k: k + 2048]
            optimiser.zero_grad()
            loss = nn.functional.mse_loss(net(zt[idx]), tt[idx])
            loss.backward()
            optimiser.step()
            total += loss.item() * len(idx)
        log.info("surrogate epoch %d: distillation MSE %.4f", epoch + 1, total / n)
    net.eval()
    return net


def _pgd(net, z: np.ndarray, mask: np.ndarray, eps: float, lo: np.ndarray, hi: np.ndarray,
         steps: int, seed: int) -> np.ndarray:
    import torch

    rng = np.random.default_rng(seed)
    z0 = torch.tensor(z, dtype=torch.float32)
    m = torch.tensor(mask, dtype=torch.float32)
    delta = torch.tensor(rng.uniform(-eps, eps, z.shape), dtype=torch.float32) * m
    alpha = 2.5 * eps / steps
    lo_t = torch.tensor(lo, dtype=torch.float32)
    hi_t = torch.tensor(hi, dtype=torch.float32)
    for _ in range(steps):
        delta.requires_grad_(True)
        score = net(z0 + delta).sum()           # push the fraud score down
        (grad,) = torch.autograd.grad(score, delta)
        with torch.no_grad():
            delta = delta - alpha * grad.sign() * m
            delta = delta.clamp(-eps, eps)
            delta = torch.max(torch.min(z0 + delta, hi_t), lo_t) - z0
    return (z0 + delta).detach().numpy().astype(np.float64)


def run(cfg: Config) -> pd.DataFrame:
    seed = int(cfg.require("project.seed"))
    set_seed(seed)
    epsilons = [float(e) for e in cfg.get("robustness.epsilons", [0.1, 0.25, 0.5, 1.0])]
    steps = int(cfg.get("robustness.pgd_steps", 30))
    max_attacked = int(cfg.get("robustness.max_attacked", 2000))
    patterns = list(cfg.require("robustness.mutable_features"))

    model = DeployedModel(cfg)
    X, y, _, _ = load_features(cfg)
    split = load_split(cfg)
    y_arr = y.to_numpy()
    mutable = mutable_columns(list(X.columns), patterns)
    if not mutable:
        raise RuntimeError("no feature matches robustness.mutable_features")
    log.info("attacker controls %d of %d features", len(mutable), X.shape[1])

    rng = np.random.default_rng(seed)
    train = split.train
    sample = np.sort(np.concatenate([
        train[y_arr[train] == 1],
        rng.choice(train[y_arr[train] == 0], min(150_000, int((y_arr[train] == 0).sum())),
                   replace=False),
    ]))
    scaler = _Scaler(X.iloc[sample])
    target = _logit(model.raw(X.iloc[sample]))
    net = _surrogate(scaler.to_z(X.iloc[sample]), target, seed)

    test = split.test
    X_test = X.iloc[test]
    base_risk = model.risk(X_test)
    tiers = model.policy.assign(base_risk)
    fraud = y_arr[test] == 1

    import torch
    from scipy.stats import spearmanr

    probe = rng.choice(len(test), min(20_000, len(test)), replace=False)
    with torch.no_grad():
        surrogate_scores = net(torch.tensor(scaler.to_z(X_test.iloc[probe]),
                                            dtype=torch.float32)).numpy().ravel()
    fidelity = float(spearmanr(surrogate_scores, _logit(model.raw(X_test.iloc[probe])))[0])
    log.info("surrogate fidelity on test (Spearman): %.3f", fidelity)

    caught = np.flatnonzero(fraud & (tiers != "low"))
    attacked = np.sort(rng.choice(caught, min(max_attacked, len(caught)), replace=False))
    X_att = X_test.iloc[attacked]
    z_att = scaler.to_z(X_att)
    mask = np.zeros_like(z_att)
    cols = [X.columns.get_loc(c) for c in mutable]
    mask[:, cols] = 1.0
    mask = mask * ~np.isnan(X_att.to_numpy(np.float64))  # absent fields stay absent

    total_fraud = int(fraud.sum())
    detected_before = int((fraud & (tiers != "low")).sum())
    high_before = base_risk[attacked] >= model.policy.high_lo
    rows = []
    for eps in epsilons:
        for attack in ("pgd", "random"):
            if attack == "pgd":
                z_adv = _pgd(net, z_att, mask, eps, scaler.lo, scaler.hi, steps, seed)
            else:
                signs = rng.choice([-1.0, 1.0], size=z_att.shape)
                z_adv = np.clip(z_att + eps * signs * mask, scaler.lo, scaler.hi)
            risk_after = model.risk(scaler.from_z(z_adv, X_att))
            evaded = risk_after < model.policy.medium_lo
            detected_after = detected_before - int(evaded.sum())
            rows.append({
                "epsilon_std": eps,
                "attack": attack,
                "frauds_attacked": len(attacked),
                "evasion_rate": round(float(evaded.mean()), 4),
                "high_tier_kept": round(float(np.mean(
                    risk_after[high_before] >= model.policy.high_lo)) if high_before.any() else np.nan, 4),
                "mean_risk_before": round(float(base_risk[attacked].mean()), 2),
                "mean_risk_after": round(float(risk_after.mean()), 2),
                "fraud_recall_before": round(detected_before / total_fraud, 4),
                "fraud_recall_after": round(detected_after / total_fraud, 4),
            })
            log.info("eps %.2f %-6s evasion %.1f%%, recall %.3f -> %.3f", eps, attack,
                     100 * rows[-1]["evasion_rate"], rows[-1]["fraud_recall_before"],
                     rows[-1]["fraud_recall_after"])

    frame = pd.DataFrame(rows)
    frame["recall_drop_relative"] = (
        1 - frame["fraud_recall_after"] / frame["fraud_recall_before"]).round(4)
    reports = cfg.path("reports")
    frame.to_csv(reports / "tables" / "robustness.csv", index=False)
    write_json({
        "threat_model": "attacker edits only features derived from their own transaction's fields",
        "mutable_features": mutable,
        "surrogate_fidelity_spearman": fidelity,
        "pgd_steps": steps,
        "caveat": "feature-space perturbation; derived features may move without their "
                  "source, so this is an upper bound on a real attacker",
        "results": frame.to_dict(orient="records"),
    }, reports / "metrics" / "robustness.json")

    from .. import plots

    plots.robustness(frame, reports / "figures" / "robustness.png")
    return frame
