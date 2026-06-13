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

## Follow-up: what we actually tried, and the results

All timings below are medians on a 4-core Xeon container, double precision,
nucleotide data (S=4) — the regime where the original speedup eroded with sites.

### 1. Site-blocked SIMD (shipped, opt-in via `block_size`)

Implemented exactly as proposed: process `block_size` sites at a time in a
transposed `[node, state, block]` scratch so the per-node matrix products become
AXPY loops over the contiguous site dimension. Forward stays bit-identical.

Result: **performance-neutral, occasionally a slight gradient win, never a clear
forward win.** Representative value+grad medians (256 taxa × 4000 sites):
`block_size` 1 → 41.5 ms, 8 → 31.7, 16 → 31.1, 32 → 30.4, 64 → 36.6; the forward
alone was ~flat (≈10 ms) across block sizes and if anything slightly *worse* than
the direct-to-buffer per-site path. Even at codon-scale state counts (S=60),
where the S² matrix work should dominate, blocking moved nothing measurable.

Conclusion: at these sizes the kernel is **memory-movement bound**, not compute
bound, so vectorising the arithmetic doesn't change throughput — the compiler
was already vectorising the scalar `sum_j` reduction well enough, and the extra
transpose traffic offsets any gain. It is therefore shipped **off by default**
(`block_size=1` = the original per-site traversal) and available to opt into on
hardware/state-counts where compute dominates. See
`native_block_size_benchmark.ipynb`.

### 2. Forward-mode (JVP) for the gradient — not applicable

A JVP (forward-mode AD) computes one directional derivative `(∂L/∂P)·v` per
pass. To assemble the full gradient over all `Bt·M·S·S` entries of the
transition matrices you would need that many JVP passes. Reverse-mode (VJP) gets
the entire gradient of the scalar log-likelihood in **one** backward sweep, which
is why it is used. Forward-mode wins only when outputs ≫ inputs — the opposite of
our case (one scalar loss, many parameters). So JVP does not reduce gradient cost
here; it would be orders of magnitude worse.

### 3. Recompute / checkpointing the partials (prototyped, neutral)

The partials buffer exists only because reverse-mode needs the forward
intermediates. The classic lever is to **not store `[B, Nn, S]`** at all:
recompute the forward partials block-by-block in cache inside the backward pass
(plus a forward-only value op that skips the partials write). That removes the
~262 MB write + read round-trip of the partials tensor, at the cost of one extra
in-cache forward traversal.

Prototyped as two ops (`PhyloLikelihoodValue` + `PhyloLikelihoodValueGrad`) wired
with `tf.custom_gradient`. Correctness was exact (gradient matched the
save-partials path to ~1e-14). Performance:

| taxa × sites | save-partials v+grad | recompute v+grad | speedup |
|---|---|---|---|
| 128 × 2000  (block=16) |   7.2 ms |   7.3 ms | 0.99× |
| 256 × 4000  (block=16) |  27.4 ms |  27.5 ms | 1.00× |
| 512 × 8000  (block=16) | 109.8 ms | 108.2 ms | 1.01× |

i.e. **neutral**: the extra forward recompute costs about as much as the partials
I/O it eliminates on this memory subsystem (and at `block_size=1` it was a net
loss, ~0.75×). The prototype was therefore not merged.

### Where this leaves us

On the tested hardware the original scalar kernel is already close to optimal for
the memory-bound, low-state regime, and none of the three levers moved the needle
there. The remaining ideas most likely to help are the ones that genuinely cut
DRAM traffic *and* whose recompute/overhead is cheap relative to it — most
plausibly on machines with much higher core counts and memory bandwidth (where
the partials round-trip is a larger share of the time), or for large state
spaces / float32 (which change the compute-vs-bandwidth balance). Re-running
`native_block_size_benchmark.ipynb` on the target hardware is the way to decide.

## Follow-up 2: the rate-category mixture tiling (fixed)

The biggest *practical* inefficiency turned out not to be the inner kernel at all,
but how the batch was being assembled upstream for a **discrete rate-category
mixture** (discrete-Gamma / Weibull site rates — the common case in real
analyses).

### The problem

With `M` rate categories, `get_sequence_distribution` wraps `LeafCTMC` in a
`DiscreteParameterMixture`, which adds a category batch dimension. The transition
matrices then have batch shape `[…, M]` — they vary only across categories, *not*
across sites. But `_canonicalize_batch` flattened the full batch to `B = sites · M`
and **tiled the transition matrices to `[sites · M, node, S, S]`**, replicating
each category's matrices once per site. For 128 taxa × 2000 sites × 4 categories
that is a **261 MB** transition tensor (and the gradient then had to reduce a
261 MB `grad_probs` back down via an autodiff broadcast-sum). The op was forced
onto its slowest path (`Bt == B`: a distinct per-element matrix set, no broadcast
reuse).

### The fix: gather indices instead of tiling

The op now takes, per (flattened) batch element `b`, a **gather index** into the
transition-probs / frequencies batch (`probs_index[b]`, `freqs_index[b]`), and
reads `transition_probs[probs_index[b]]` directly. The transition matrices stay
at their own size `Bt` (= `M` for a rate mixture, `1` for the common broadcast
case, `B` for genuinely per-element), and `_canonicalize_batch` builds the index
with a cheap broadcast of `arange(Bt)` — `B` ints (tens of KB) instead of a
multi-hundred-MB tile. The backward op scatters the gradient back into the `Bt`
sets, reducing across the elements that share a set (thread-local accumulators +
a final locked reduction, exactly as the old `Bt == 1` broadcast path did, now
generalised to any `Bt < B`).

This is deliberately a **general** mechanism, not a rate-mixture special case: the
gather index expresses *any* broadcasting pattern (leading sample dims,
interleaved VI parameter-sample / category / site dims, …), so it also covers the
VI case where parameter draws and categories are both batch dims. `Bt == 1`
(broadcast) and `Bt == B` (per-site) remain fast special paths in the kernel; the
new middle ground (`1 < Bt < B`) is what the rate mixture needs.

### Result

Value + gradient, double precision, 4 states, 4-core container (medians):

| taxa × sites × cats | before (tiled, `Bt = B`) | after (gather, `Bt = M`) | speedup | probs tile removed |
|---|---:|---:|---:|---:|
| 64 × 1000 × 4  | 35.8 ms | 14.6 ms | 2.5× | 65 MB |
| 128 × 2000 × 4 | 125.8 ms | 44.3 ms | 2.8× | 261 MB |
| 64 × 1000 × 8  | 76.9 ms | 25.1 ms | 3.1× | 130 MB |

Correctness is covered by `test_native_shared_batch_*` (value, gradient, and
rescaled gradient vs. the reference; plus a bit-identical check against the
explicitly tiled `Bt == B` path) and an end-to-end discrete-Gamma mixture test
through `LeafCTMC` / `DiscreteParameterMixture`.

### Should we instead specialise a kernel?

Two specialised kernels were considered; the gather-index fix above makes the
first unnecessary and clarifies when the second is worth it.

1. **A dedicated rate-mixture kernel** (an explicit category axis, looping
   categories while reusing each site's transposed leaf partials). This would
   additionally avoid replicating the *leaf data* across categories (the
   remaining `[sites · M, L, S]` broadcast that happens one layer up, in
   `LeafCTMC._broadcast_for_likelihood`). But leaf data is `O(sites · M · L · S)`
   — far smaller than the transition tile we just eliminated, and read
   sequentially — so the payoff is small and it bakes the mixture structure into
   the op. **Not pursued:** the general gather index already removes the dominant
   cost with no special-casing.

2. **An eigendecomposition kernel that never materialises `P`.** This is the
   genuinely different axis and the recommended next step if more is needed. For
   an `EigendecompositionSubstitutionModel` the transition matrices are
   `P(t) = U · diag(exp(λ·t)) · U⁻¹`. Today we materialise every `P` (shape
   `[…, node, S, S]`, for a rate mixture `[M, node, S, S]`) and differentiate
   w.r.t. those entries. A kernel taking `U, U⁻¹, λ` and the (rate-scaled) branch
   lengths could form each `P` on the fly per node/category, so:
   - the `[…, node, S, S]` transition tensor is never built or stored (a real
     bandwidth win at large `node · M`), and
   - the gradient flows directly to **branch lengths and the rate matrix**, which
     is what callers actually optimise — removing the separate
     `get_transition_probabilities` matrix-exp + its backward graph.

   The trade-off is a larger, substitution-model-aware kernel and recomputing
   `exp(λ·t)` (cheap: `S` exponentials per branch) inside both the forward and
   backward passes. This is worth doing when the transition-matrix construction
   /storage — not the pruning traversal — is the bottleneck (large `S`, many
   categories), and is left as future work.
