# Native likelihood: why the speedup shrinks with more sites

This note explains the scaling pattern seen in
`examples/native_likelihood_benchmark.ipynb` — the native C++ op's speedup over
the TensorFlow reference **grows with the number of taxa but shrinks with the
number of sites** — and what we can do to keep the speedup at large site counts.

## The observation

From the benchmark sweep (double precision, 4 states, CPU):

| leaves | sites | ref ms | native ms | **likelihood speedup** | ref grad ms | nat grad ms | **grad speedup** |
|-------:|------:|-------:|----------:|-----------------------:|------------:|------------:|-----------------:|
| 8      | 500   | 0.86   | 0.39      | 2.2x  | 1.36    | 0.38  | 3.6x  |
| 16     | 500   | 1.02   | 0.32      | 3.2x  | 1.93    | 0.44  | 4.4x  |
| 32     | 1000  | 8.89   | 0.84      | 10.6x | 28.54   | 0.85  | 33.7x |
| 64     | 1000  | 11.49  | 0.65      | 17.7x | 32.69   | 0.92  | 35.6x |
| 128    | 1000  | 20.60  | 0.60      | 34.1x | 65.47   | 1.36  | 48.2x |
| 128    | 2000  | 29.17  | 0.98      | 29.9x | 118.04  | 2.23  | 53.0x |
| 256    | 2000  | 58.36  | 1.55      | 37.7x | 238.92  | 4.17  | 57.3x |
| 512    | 2000  | 118.51 | 3.72      | 31.9x | 483.25  | 10.21 | 47.3x |
| 1024   | 2000  | 247.35 | 7.67      | 32.3x | 1025.32 | 20.05 | 51.1x |
| 1024   | 4000  | 396.36 | 13.23     | 30.0x | 1399.44 | 38.92 | 36.0x |

Holding sites fixed, the speedup climbs steeply with taxa (2x → 34x). Holding
taxa fixed and doubling the sites, it erodes: 128 taxa 34x → 30x (1000 → 2000
sites); 1024 taxa 32x → 30x (2000 → 4000 sites). The gradient erodes harder:
1024 taxa 51x → 36x.

## Why the speedup grows with taxa

The TensorFlow reference
(`treeflow/traversal/phylo_likelihood.py`) runs a graph-level postorder loop:
**one iteration per internal node** (`L − 1` of them), each issuing a handful of
separate TF ops (`gather`, broadcast-multiply, `reduce_sum`, `reduce_prod`,
`TensorArray.write`). Every op carries a fixed dispatch/launch overhead that is
independent of problem size, so the reference has a cost term proportional to

```
(number of nodes) × (per-op overhead)
```

The native op runs the whole traversal inside a single compiled kernel with no
per-node dispatch. As the tree grows, the reference pays more and more fixed
overhead that the native op simply does not have — so the speedup grows with
taxa, essentially without bound. This is the native op's structural advantage.

## Why the speedup shrinks with more sites

Sites are the batch dimension `B`. Both implementations spread work over sites,
but in very different ways:

- **The reference vectorises across sites.** Each node's computation is a few
  Eigen/BLAS-backed tensor ops over the whole `B`-site batch. Its per-op
  overhead is fixed, so as `B` grows that overhead is amortised over more sites
  and the reference's *per-site* cost falls toward an efficient, vectorised
  steady state. More sites make the reference **relatively more efficient** —
  the overhead-dominated regime that favours the native op fades.

- **The native op uses a hand-written scalar inner loop.** Each site is walked
  by a scalar triple loop — the matvec
  `for s: for j: g += P[s,j] * partial_child[j]`
  (`phylo_likelihood_op.cc`). Two things hurt it as `B` grows:

  1. **It becomes memory-bandwidth bound.** The forward op allocates and writes
     the full `node_partials` buffer of shape `[B, Nn, S]` (it must, to feed the
     analytic gradient). At 1024 taxa × 4000 sites that is
     `4000 × 2047 × 4 × 8 B ≈ 262 MB` — far beyond any cache. The arithmetic per
     node is tiny (`S² = 16` multiply-adds for nucleotides), so the kernel is
     dominated by streaming that buffer to and from DRAM. Once the working set
     blows past cache, the op is bandwidth-limited and its compute advantage
     stops mattering.

  2. **The inner loop is not vectorised.** The hot kernel is a 4×4 matvec with a
     length-4 reduction — too short to vectorise over `j`, and because sites are
     processed one at a time the compiler cannot vectorise across sites either.
     So the native op runs essentially scalar per element, while the reference's
     per-site work goes through vectorised Eigen kernels. In the large-`B`
     regime — where per-site throughput dominates total time — the reference's
     better per-element efficiency narrows the gap.

The **gradient** erodes faster (1024 taxa: 51x → 36x) because the backward op
reads that same multi-hundred-MB partials buffer *and* writes gradient
accumulators, roughly doubling the memory traffic, so it hits the bandwidth wall
sooner.

In short: **taxa scaling is won on fixed per-op overhead — the native op's
structural, unbounded advantage. Site scaling is a per-element throughput
contest, and the native op currently loses ground there because its inner kernel
is scalar and memory-bound.**

## What we can do to keep the speedup at large site counts

The fix is to make the per-site arithmetic vectorised and cache-friendly —
essentially the BEAGLE strategy. In rough priority order:

1. **Block sites and vectorise across them (highest impact).** Restructure so
   that for each node we apply its `S×S` transition matrix to a *panel* of sites
   at once. With partials laid out state-major / sites-innermost (`[S, block]`),
   the core operation becomes a small matrix–matrix product
   `P[S,S] × Partials[S, block]`, which (a) vectorises cleanly over the site
   block with SIMD and (b) loads each `P` matrix once per node per block instead
   of once per node per site, cutting memory traffic dramatically. This attacks
   both the vectorisation and the bandwidth problem at once.

2. **Cache-tile over sites.** Process sites in tiles sized to fit L2 (a few
   hundred to ~1k sites) so a tile's partials and the transition matrices stay
   resident while the traversal runs, instead of streaming a 100+ MB buffer once
   per node. Combined with (1), the traversal reuses hot data and the kernel
   stops being DRAM-bound.

3. **Don't materialise all partials when the gradient isn't needed.** The
   forward op always allocates `[B, Nn, S]` to hand to the backward pass. For a
   pure likelihood evaluation (no `GradientTape`) that is ~262 MB of pointless
   allocation and writes at the large-`B` end. A forward-only variant that keeps
   only the running partials (or recomputes them in the backward pass) removes a
   large chunk of the bandwidth cost from the likelihood-only path.

4. **Let the vectoriser do its job.** The build uses `-O3 -mavx2` but avoids
   `-march=native` (an AVX512-FP16 bug in the bundled Eigen headers — see
   `build.sh`). The wins must come from vectorising the long *site* dimension,
   not the length-4 state loops, so the layout change in (1) is the prerequisite
   — compiler flags alone will not help while the loop is site-at-a-time.

5. **Lower priority: narrower dtype and rate-category handling.** A float32 path
   halves the bandwidth (the rescaling threshold logic already accounts for the
   reduced stability). And the discrete-Gamma rate mixture multiplies the
   effective node count, so the blocked inner loop should treat rate categories
   as part of the vectorised panel to keep them efficient too.

Expected outcome: with site-blocking plus tiling, the per-site kernel becomes
SIMD-vectorised and compute-bound, so the native op's per-site throughput
advantage is preserved as the site count grows. The speedup should stay roughly
flat (or keep climbing) with sites instead of eroding — while the taxa-scaling
advantage is untouched.
