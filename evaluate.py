#!/usr/bin/env python3
"""
evaluate.py
===========
Reproduce every metric, 95% confidence interval and significance test reported
in the paper from the released per-sample score files (scores/).

Metrics
-------
F1      F1-score of the WuW class.
BA      Device-balanced accuracy: mean of the per-device accuracies.
WDM     Mean of the pairwise Wasserstein-1 distances between the per-device
        distributions of the WuW score.
WDS     Standard deviation of those pairwise distances.
A sample is predicted as WuW when score_wuw > score_non_wuw. F1 and BA are
computed on WuW Device Diversity; WDM and WDS on both test sets.

Uncertainty
-----------
One-stage cluster bootstrap: each resample draws clusters with replacement and
keeps every sample of each drawn cluster.
  - WuW Device Diversity: cluster = speaker (24).
  - Parallel TAU:         cluster = audio clip (78), recorded on devices a/b/c.
Every system is evaluated on the same resamples, so the difference to BL is
paired. CIs are percentile intervals. The two-sided p-value of a difference is
    p = min(1, 2 · min(#{Δ* ≤ 0} + 1, #{Δ* ≥ 0} + 1) / (B + 1)),
so p < α exactly when the CI of the difference excludes 0. No correction for
multiple comparisons is applied.

Outputs (in --output-dir)
-------------------------
ci_results.csv     per system and metric: point estimate, CI, whether the CI
                   contains the point estimate, mean over resamples
significance.csv   per system and metric: difference to BL, its CI, p-value,
                   significant (p < α)
threshold_optimization.csv
                   BL and DA-2: leave-one-device-out threshold optimisation,
                   re-run on every WuW Device Diversity resample. Final
                   threshold; BA and BF1 (mean per-device WuW F1) at 0.5 and at
                   the final threshold, with CIs; optimised − default, its CI,
                   p-value, significant (paired within each system)

Usage
-----
python evaluate.py [--bootstraps 10000] [--seed 42] [--alpha 5] [--workers 8]
                   [--output-dir results] [--exclude-devices DEV [DEV ...]]

--exclude-devices drops WuW Device Diversity devices before resampling (the
sensitivity analysis drops HUAWEI_BXX-W09 and samsung_SM-S721B).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

HERE = Path(__file__).resolve().parent
SCORES = HERE / "scores"
SYSTEMS = pd.read_csv(HERE / "experiments.tsv", sep="\t")["id"].tolist()

DD_METRICS = ["F1", "BA", "WDM_DD", "WDS_DD"]
TAU_METRICS = ["WDM_TAU", "WDS_TAU"]
METRICS = DD_METRICS + TAU_METRICS

THRESHOLD_SYSTEMS = ["BL", "DA-2"]
THRESHOLD_GRID = np.round(np.arange(0.01, 1.0, 0.01), 2)
THRESHOLD_METRICS = ["BA", "BF1"]


def f1_wuw(is_wuw: np.ndarray, pred_wuw: np.ndarray) -> float:
    tp = np.sum(is_wuw & pred_wuw)
    fp = np.sum(~is_wuw & pred_wuw)
    fn = np.sum(is_wuw & ~pred_wuw)
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom else 0.0


def device_balanced_accuracy(is_wuw: np.ndarray, pred_wuw: np.ndarray, devices: np.ndarray) -> float:
    return float(np.mean([np.mean(is_wuw[devices == d] == pred_wuw[devices == d])
                          for d in np.unique(devices)]))


def wdm_wds(scores: np.ndarray, devices: np.ndarray) -> tuple[float, float]:
    groups = {d: scores[devices == d] for d in np.unique(devices)}
    pw = [wasserstein_distance(groups[a], groups[b]) for a, b in combinations(sorted(groups), 2)]
    return float(np.mean(pw)), float(np.std(pw))


def _per_device(is_wuw: np.ndarray, pred: np.ndarray, devices: np.ndarray,
                devs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-device accuracy and WuW F1 for each column of pred (n, T). Both (K, T)."""
    acc, f1 = [], []
    for d in devs:
        w, p = is_wuw[devices == d, None], pred[devices == d]
        tp, fp, fn = (p & w).sum(0), (p & ~w).sum(0), (~p & w).sum(0)
        denom = 2 * tp + fp + fn
        acc.append((p == w).mean(0))
        f1.append(np.divide(2 * tp, denom, out=np.zeros(len(denom)), where=denom > 0))
    return np.array(acc), np.array(f1)


def threshold_optimization(is_wuw: np.ndarray, scores: np.ndarray,
                           devices: np.ndarray) -> tuple[float, ...]:
    """Each round picks the grid threshold maximising BA on the other K−1
    devices; the final threshold is the median of the K picks. Returns
    (threshold, BA and BF1 at 0.5, BA and BF1 at the final threshold), with
    every device evaluated at the final threshold."""
    devs = np.unique(devices)
    acc, _ = _per_device(is_wuw, scores[:, None] >= THRESHOLD_GRID, devices, devs)
    picks = [THRESHOLD_GRID[np.argmax(np.delete(acc, k, axis=0).mean(0))] for k in range(len(devs))]
    thr = float(np.median(picks))
    acc, f1 = _per_device(is_wuw, np.stack([scores >= 0.5, scores >= thr], axis=1), devices, devs)
    (ba0, ba1), (bf0, bf1) = acc.mean(0), f1.mean(0)
    return thr, ba0, bf0, ba1, bf1


def load(test_set: str, system: str, exclude: tuple[str, ...] = ()) -> pd.DataFrame:
    df = pd.read_csv(SCORES / test_set / f"{system}.tsv", sep="\t")
    df = df[~df["device"].isin(exclude)]
    return df.sort_values("sample_id").reset_index(drop=True)


def cluster_rows(clusters: pd.Series) -> tuple[list[str], list[np.ndarray]]:
    names = sorted(clusters.unique())
    return names, [np.flatnonzero(clusters.to_numpy() == c) for c in names]


def make_draws(n_clusters: int, n_boot: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n_clusters, size=(n_boot, n_clusters))


def two_sided_p(delta: np.ndarray) -> float:
    tail = min(np.sum(delta <= 0) + 1, np.sum(delta >= 0) + 1)
    return float(min(1.0, 2 * tail / (len(delta) + 1)))


def bootstrap_system(args: tuple) -> dict:
    system, dd_draws, tau_draws, exclude = args
    out: dict = {"system": system}

    df = load("wuw_device_diversity", system, exclude)
    _, rows = cluster_rows(df["speaker_id"])
    is_wuw = df["is_wuw"].to_numpy().astype(bool)
    pred = (df["score_wuw"] > df["score_non_wuw"]).to_numpy()
    sc, dv = df["score_wuw"].to_numpy(), df["device"].to_numpy()

    def dd_stats(idx: np.ndarray) -> tuple[float, ...]:
        return (f1_wuw(is_wuw[idx], pred[idx]),
                device_balanced_accuracy(is_wuw[idx], pred[idx], dv[idx]),
                *wdm_wds(sc[idx], dv[idx]))

    out["dd_point"] = np.array(dd_stats(np.arange(len(df))))
    out["dd_boot"] = np.array([dd_stats(np.concatenate([rows[c] for c in d])) for d in dd_draws])

    tau = load("parallel_tau", system)
    _, rows = cluster_rows(tau["clip_id"])
    sc, dv = tau["score_wuw"].to_numpy(), tau["device"].to_numpy()
    out["tau_point"] = np.array(wdm_wds(sc, dv))
    out["tau_boot"] = np.array([wdm_wds(sc[i], dv[i]) for i in
                                (np.concatenate([rows[c] for c in d]) for d in tau_draws)])
    return out


def bootstrap_threshold(args: tuple) -> dict:
    system, dd_draws, exclude = args
    df = load("wuw_device_diversity", system, exclude)
    _, rows = cluster_rows(df["speaker_id"])
    w = df["is_wuw"].to_numpy().astype(bool)
    sc, dv = df["score_wuw"].to_numpy(), df["device"].to_numpy()
    return {"system": system,
            "point": np.array(threshold_optimization(w, sc, dv)),
            "boot": np.array([threshold_optimization(w[i], sc[i], dv[i]) for i in
                              (np.concatenate([rows[c] for c in d]) for d in dd_draws)])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bootstraps", type=int, default=10000, help="Bootstrap resamples (default: 10000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--alpha", type=float, default=5, help="Level in %% (default: 5 → 95%% CI)")
    parser.add_argument("--workers", type=int, default=8, help="Parallel processes (default: 8)")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results", help="Output directory (default: results/)")
    parser.add_argument("--exclude-devices", nargs="+", default=[], help="WuW Device Diversity devices to drop")
    args = parser.parse_args()
    B, A, exclude = args.bootstraps, args.alpha, tuple(args.exclude_devices)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dd_clusters, _ = cluster_rows(load("wuw_device_diversity", "BL", exclude)["speaker_id"])
    tau_clusters, _ = cluster_rows(load("parallel_tau", "BL")["clip_id"])
    dd_draws = make_draws(len(dd_clusters), B, args.seed)
    tau_draws = make_draws(len(tau_clusters), B, args.seed + 1)
    print(f"B={B}, seed={args.seed}, α={A}% | WuW DD: {len(dd_clusters)} speakers | "
          f"Parallel TAU: {len(tau_clusters)} clips" + (f" | excluded: {', '.join(exclude)}" if exclude else ""))

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        res = {r["system"]: r for r in ex.map(bootstrap_system,
                                              [(s, dd_draws, tau_draws, exclude) for s in SYSTEMS])}
        thr_res = list(ex.map(bootstrap_threshold, [(s, dd_draws, exclude) for s in THRESHOLD_SYSTEMS]))

    def point_and_boot(r: dict) -> tuple[dict, dict]:
        pts = dict(zip(DD_METRICS, r["dd_point"])) | dict(zip(TAU_METRICS, r["tau_point"]))
        boots = ({m: r["dd_boot"][:, i] for i, m in enumerate(DD_METRICS)}
                 | {m: r["tau_boot"][:, i] for i, m in enumerate(TAU_METRICS)})
        return pts, boots

    bl_pts, bl_boots = point_and_boot(res["BL"])
    ci_rows, sig_rows = [], []
    for system in SYSTEMS:
        pts, boots = point_and_boot(res[system])
        ci, sig = {"Label": system}, {"Label": system}
        for m in METRICS:
            lo, hi = np.percentile(boots[m], [A / 2, 100 - A / 2])
            ci |= {m: pts[m], f"{m}_lo": lo, f"{m}_hi": hi,
                   f"{m}_ci_valid": bool(lo <= pts[m] <= hi), f"{m}_boot_mean": boots[m].mean()}
            if system != "BL":
                delta = boots[m] - bl_boots[m]
                dlo, dhi = np.percentile(delta, [A / 2, 100 - A / 2])
                p = two_sided_p(delta)
                sig |= {f"{m}_diff": pts[m] - bl_pts[m], f"{m}_lo": dlo, f"{m}_hi": dhi,
                        f"{m}_p": p, f"{m}_sig": p < A / 100}
        ci_rows.append(ci)
        if system != "BL":
            sig_rows.append(sig)

    pd.DataFrame(ci_rows).to_csv(args.output_dir / "ci_results.csv", index=False)
    pd.DataFrame(sig_rows).to_csv(args.output_dir / "significance.csv", index=False)
    print(f"Saved: {args.output_dir / 'ci_results.csv'}\nSaved: {args.output_dir / 'significance.csv'}")

    thr_rows = []
    for r in thr_res:
        pt, bt = r["point"], r["boot"]
        row = {"Label": r["system"], "Threshold": pt[0]}
        for j, m in enumerate(THRESHOLD_METRICS):
            for k, op in ((1 + j, "default"), (3 + j, "opt")):
                lo, hi = np.percentile(bt[:, k], [A / 2, 100 - A / 2])
                row |= {f"{m}_{op}": pt[k], f"{m}_{op}_lo": lo, f"{m}_{op}_hi": hi}
            delta = bt[:, 3 + j] - bt[:, 1 + j]
            dlo, dhi = np.percentile(delta, [A / 2, 100 - A / 2])
            p = two_sided_p(delta)
            row |= {f"{m}_diff": pt[3 + j] - pt[1 + j], f"{m}_diff_lo": dlo, f"{m}_diff_hi": dhi,
                    f"{m}_p": p, f"{m}_sig": p < A / 100}
        thr_rows.append(row)
    pd.DataFrame(thr_rows).to_csv(args.output_dir / "threshold_optimization.csv", index=False)
    print(f"Saved: {args.output_dir / 'threshold_optimization.csv'}")


if __name__ == "__main__":
    main()
