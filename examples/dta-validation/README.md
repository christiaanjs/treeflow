# DTA recovery validation for `treeflow_dta_hmc`

End-to-end sanity check for the discrete-trait (phylogeography / DTA)
NUTS CLI added in this branch: simulate a K-state trait under a known
Q on a known tree, fit with `treeflow_dta_hmc`, and verify that the
posterior mean / 95% CI cover the simulation truth.

## Truth (see `dta-validation.lphy`)

- `K = 4` states, labelled `"0","1","2","3"` (lexicographic order)
- `π = [0.4, 0.3, 0.2, 0.1]`
- `R = [0.25, 0.15, 0.10, 0.20, 0.10, 0.20]`  (row-major upper triangle)
- `μ = 0.2`
- `N = 400` tips, Yule(λ=1)

## Requirements

- [LinguaPhylo](https://github.com/LinguaPhylo/linguaPhylo) built
  locally for simulation — default path
  `~/Git/linguaPhylo/lphy-studio/target/lphy-studio-1.7.0`
- `treeflow_dta_hmc` on `$PATH` (installed with this repo's extras)

## Run

```sh
./run.sh
```

Runs: `slphy` → `convert.py` (nexus → `traits.csv` + `tree.nwk`) →
`treeflow_dta_hmc` (500 burn-in + 1000 samples, default NUTS) →
`summarize.py` (posterior vs. truth).

On a MacBook this takes ≈ 51 min (NUTS is the bottleneck, the first
~10 s is LPhy).

## Expected outcome

Posterior covers 9/10 parameters on the default seed. The one miss
(π₁) is a finite-tree-depth artefact — observed state-1 count is
well below the stationary expectation.

For reference, the Stan port at
[`phylogeo-stan`](https://github.com/alexeid/phylogeo-stan) runs the
same model on the same dataset in ≈ 97 s (32× faster) with the same
9/10 coverage.
