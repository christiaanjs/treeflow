"""Summarise treeflow_dta_hmc posterior vs simulation truth.

Reads samples.csv produced by treeflow_dta_hmc and compares posterior
mean / 95% CI / ESS for each π and R parameter against the fixed truth
encoded in dta-validation.lphy.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Truth from dta-validation.lphy. State order: "0","1","2","3"
# (lexicographic, matches DiscreteTraitData default).
# Rate order: row-major upper triangle — (0,1),(0,2),(0,3),(1,2),(1,3),(2,3).
TRUTH_PI = [0.4, 0.3, 0.2, 0.1]
TRUTH_R = [0.25, 0.15, 0.10, 0.20, 0.10, 0.20]
RATE_LABELS = ["(0,1)", "(0,2)", "(0,3)", "(1,2)", "(1,3)", "(2,3)"]


def ess_ar1(x: np.ndarray, max_lag: int = 500, cutoff: float = 0.05) -> float:
    """ESS via autocorrelation sum, cut when |r_k| drops below `cutoff`."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    n = len(x)
    denom = float((x * x).sum())
    if denom == 0.0:
        return float(n)
    r = np.correlate(x, x, mode="full")[n - 1:] / denom
    s = 1.0
    for k in range(1, min(n, max_lag)):
        if abs(r[k]) < cutoff:
            break
        s += 2 * r[k]
    return float(n) / max(s, 1.0)


def summarise(samples_csv: Path) -> None:
    df = pd.read_csv(samples_csv)
    print(f"Loaded {len(df)} samples from {samples_csv}")
    print(f"Columns: {list(df.columns)}\n")

    header = f"{'param':<10}{'label':<10}{'truth':>8}{'mean':>10}{'q025':>10}{'q975':>10}{'cover':>8}{'ess':>10}"
    print(header)
    print("-" * len(header))

    rows: list[tuple[str, str, float, float, float, float, bool, float]] = []
    for i, t in enumerate(TRUTH_PI):
        s = df[f"frequencies_{i}"].to_numpy()
        lo, hi = float(np.quantile(s, 0.025)), float(np.quantile(s, 0.975))
        rows.append((f"pi_{i}", str(i), t, float(np.mean(s)), lo, hi, lo <= t <= hi, ess_ar1(s)))
    for i, t in enumerate(TRUTH_R):
        s = df[f"rates_{i}"].to_numpy()
        lo, hi = float(np.quantile(s, 0.025)), float(np.quantile(s, 0.975))
        rows.append((f"R_{i}", RATE_LABELS[i], t, float(np.mean(s)), lo, hi, lo <= t <= hi, ess_ar1(s)))

    for name, label, truth, mean, lo, hi, cov, ess in rows:
        cov_mark = "Y" if cov else "N"
        print(f"{name:<10}{label:<10}{truth:>8.3f}{mean:>10.3f}{lo:>10.3f}{hi:>10.3f}{cov_mark:>8}{ess:>10.1f}")

    n_cov = sum(1 for r in rows if r[6])
    print(f"\nCoverage: {n_cov}/{len(rows)} parameters inside 95% CI")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--samples",
        type=Path,
        default=Path(__file__).parent / "samples.csv",
    )
    args = ap.parse_args()
    summarise(args.samples)


if __name__ == "__main__":
    main()
