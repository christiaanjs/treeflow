# Native tree-traversal ops

Compiled TensorFlow custom ops for treeflow's two performance-critical tree
traversals, each a drop-in replacement for a pure-TensorFlow `tf.TensorArray`
reference with an **analytic** reverse-mode gradient:

* the **phylogenetic likelihood** (Felsenstein pruning, a *postorder*
  traversal) — see below;
* the **node-height ratio transform** (a *preorder* traversal) — see
  [Node-height ratio transform op](#node-height-ratio-transform-op).

Both ops describe the topology to the kernel as integer index tensors and share
the host-side index helpers in
[`cc/tree_traversal.h`](cc/tree_traversal.h).

# Native phylogenetic likelihood op

A compiled TensorFlow custom op implementing Felsenstein's pruning algorithm
(the per-site phylogenetic likelihood) and its **analytic** reverse-mode
gradient, as a drop-in replacement for the pure-TensorFlow reference in
[`treeflow/traversal/phylo_likelihood.py`](../../traversal/phylo_likelihood.py).

## Why

The reference implementation runs the postorder traversal as a Python-level
`tf.TensorArray` loop. The native op runs the whole traversal in compiled C++ as
a single TensorFlow op, parallelised over alignment sites, giving roughly a
**5–15× speedup** for both the likelihood and its gradient (see the benchmark
notebook, [`examples/native_likelihood_benchmark.ipynb`](../../../examples/native_likelihood_benchmark.ipynb)).

## How it hooks into TensorFlow

* The forward op `PhyloLikelihood` takes the tree topology index vectors
  (`postorder_indices`, `child_indices`) and the per-branch transition
  probability matrices, and outputs **both** the per-site likelihoods and the
  partial likelihood vectors at every node.
* The backward op `PhyloLikelihoodGrad` consumes those saved partials to compute
  the exact gradient with respect to the transition probabilities (and the root
  frequencies) — it does **not** recompute the forward traversal, exactly like a
  hand-written BEAGLE-style implementation.
* The gradient is wired into TensorFlow autodiff with
  `@tf.RegisterGradient("PhyloLikelihood")`, so the op works transparently
  inside `tf.GradientTape` and `tf.function`.

It is exact in `float64` (the project default) and also supports `float32`.

## Rescaling (numerical stability on large trees)

On large/deep trees the partial likelihoods underflow to zero (around ~300
taxa for `float64`, ~36 for `float32`), so the linear likelihood becomes `0`
and its log `-inf`. The **rescaled** variant (`PhyloLikelihoodRescaled` /
`native_phylogenetic_log_likelihood_rescaled`) divides the partials at every
internal node by their per-site maximum, accumulates the log of the scale
factors, and returns the per-site **log** likelihood, which stays finite. Its
analytic gradient reuses the saved scaled partials and scale factors exactly
like the unrescaled one.

There is a matching pure-TensorFlow rescaled implementation,
`treeflow.traversal.phylo_likelihood.phylogenetic_log_likelihood_rescaled`.

### Choosing rescaled vs. unrescaled

Rescaling costs a little extra per node, so it is wasteful on small trees.
`treeflow.traversal.phylo_likelihood_dispatch.phylogenetic_log_likelihood`
returns the per-site log likelihood and chooses for you via `rescaling=`:

* `False` — never rescale (fastest; may underflow);
* `True` — always rescale (most stable);
* `"auto"` (default) — pick statically from the leaf count and dtype
  (`default_rescaling_threshold`), with no runtime overhead;
* `"adaptive"` — compute the unscaled likelihood and fall back to the rescaled
  one (via `tf.cond`) only if it is not finite.

Pass `use_native=True` to route through the native ops.

## Node-height ratio transform op

The node-height ratio transform maps the per-internal-node height ratios used by
inference to the actual node heights of a time tree. It is a **preorder**
(root-to-leaves) traversal: the root height is read directly, and every other
node's height is placed a fraction (its ratio) of the way between its anchor
height and its parent's height. This is the compiled counterpart of the
reference `tf.TensorArray` loop in
[`treeflow/traversal/ratio_transform.py`](../../traversal/ratio_transform.py).

* The forward op `NodeHeightRatio` takes the preorder/parent index vectors, the
  ratios and the anchor heights, and outputs the node heights.
* The backward op `NodeHeightRatioGrad` reuses those saved heights and walks the
  nodes in **reverse preorder** (children before parents) to accumulate the
  exact gradient with respect to both the ratios and the anchor heights — no
  recomputation of the forward traversal.
* It is wired into autodiff with `@tf.RegisterGradient("NodeHeightRatio")` and
  supports arbitrary leading (sample/site) batch dimensions, `float32`/`float64`.

It is consumed through `NodeHeightRatioBijector(..., use_native=...)`
(`False` by default; `True` or `"auto"` to route the forward transform through
the native op — the inverse and log-det-Jacobian stay pure TensorFlow), and the
`treeflow_profile` CLI reports its speedup alongside the likelihood's.

```python
from treeflow.acceleration.native import native_ratios_to_node_heights

heights = native_ratios_to_node_heights(
    topology.preorder_node_indices - topology.taxon_count,  # internal-node space
    topology.parent_indices[topology.taxon_count:] - topology.taxon_count,
    ratios,          # [..., internal_node]
    anchor_heights,  # [..., internal_node]
)
```

## Subsplit Bayesian network sampler op

The **SBN topology sampler** (`SbnSample` / `treeflow.acceleration.native.sbn`)
draws whole rooted tree topologies from a subsplit Bayesian network /
conditional clade distribution — the variational family VBPI places over
topologies (see [`treeflow/vbpi`](../../vbpi)). Unlike the other two ops it has
**no gradient**: topologies are discrete, so the SBN's parameter gradients flow
through the differentiable `log_prob` (pure TensorFlow), and only the forward
draw is compiled.

Sampling a topology is an inherently *sequential* traversal of the SBN's
pointer-array (CSR) support: start at the root clade, draw one of its candidate
child subsplits from the pre-normalised conditional probabilities, and recurse
into each non-leaf child clade. That sequential walk is a poor fit for
vectorised TensorFlow but a natural fit for a compiled kernel (it shards the
independent per-sample draws across threads), exactly like the other traversal
ops here. It is the compiled counterpart of the NumPy reference walk in
`SubsplitBayesianNetwork._sample_numpy`, and is selected via
`sample_topologies(..., use_native=True)` (or `"auto"`).

```python
from treeflow.vbpi import SubsplitBayesianNetwork, SubsplitSupport

sbn = SubsplitBayesianNetwork(support)
samples = sbn.sample_topologies(1000, seed=0, use_native="auto")
# samples.parent_indices  [n, 2n-2]   treeflow topologies
# samples.candidate_indices [n, n-1]  chosen SBN parameter per internal node
# samples.node_clade_ids  [n, 2n-1]   per-node clade id (for branch params)
```

## Building

```bash
bash treeflow/acceleration/native/build.sh                  # all ops
bash treeflow/acceleration/native/build.sh node_height_ratio_op  # just one
bash treeflow/acceleration/native/build.sh sbn_op                # the SBN sampler
# or
python -m treeflow.acceleration.native.build
```

This compiles each `cc/<op>.cc` into the matching `_<op>.so` (e.g.
`_phylo_likelihood_op.so`, `_node_height_ratio_op.so`) using the compile/link
flags reported by the installed TensorFlow (so the C++ ABI matches the running
runtime). The `.so` files are intentionally git-ignored — they are
environment-specific and must be built against the local TensorFlow.

### Instruction-set baseline

`build.sh` picks its `-march`/`-mavx*` flags from `TREEFLOW_NATIVE_ARCH`:

* `portable` (default when calling `build.sh` directly, e.g. from the
  `Dockerfile` or a CI job) — a baseline shared by every supported x86_64 host
  (`-mavx2`; no flag, i.e. the compiler default, on other architectures). Use
  this whenever the `.so` might run on a different machine than the one that
  compiled it.
* `native` (the default when going through `python -m
  treeflow.acceleration.native.build`, e.g. local development or the
  test-fixture auto-build fallback) — `-march=native`, tuned for the exact CPU
  doing the compiling. Only safe when the machine that builds the op is also
  the machine that runs it.

**Why this matters:** `-march=native` bakes in whatever instruction-set
extensions (AVX2, AVX-512, ...) the *compiling* machine's CPU happens to
support. If the resulting `.so` is later executed on a different CPU that
lacks one of those extensions, the process crashes with `Fatal Python error:
Illegal instruction` (SIGILL) the moment the op runs — not a build failure,
so it surfaces as a mysterious runtime crash. This bit us in CI: the `build`
and `pytest` jobs run on separate GitHub Actions runners, and Docker's
`cache-from: type=gha` layer cache could replay a `.so` compiled on one
runner's CPU into a `pytest` job running on a different runner, causing an
intermittent SIGILL depending on whether the two runners' CPUs happened to
match. The `Dockerfile` (and therefore the published image) uses the
`portable` default for exactly this reason — the image is built once and run
on arbitrary hardware, so it can never assume build-CPU == run-CPU.

If you want `-march=native`'s extra performance for a build you know will
only ever run on the machine that compiled it, set
`TREEFLOW_NATIVE_ARCH=native` before calling `build.sh` directly, or just use
the Python entry point, which already defaults to it.

### Docker

The `Dockerfile` builds the op as part of the image: it installs a C++ compiler
in a single layer, runs `build.sh` (portable instruction-set baseline — see
above, since this image is published and run on machines other than the one
that built it), then removes the compiler again, and `pip install .` copies
the resulting `.so` into site-packages via `package_data`. The runtime image
therefore ships the native op with no build toolchain. The GitHub Actions
`pytest` workflow builds the `test` image and runs the suite, so the native op
(and its tests) are exercised in CI.

## Usage

```python
from treeflow.acceleration.native import native_phylogenetic_likelihood

site_likelihoods = native_phylogenetic_likelihood(
    sequences_onehot,        # [..., leaf, state]
    transition_probs,        # [..., node, state, state]
    frequencies,             # [..., state]
    topology.postorder_node_indices,
    topology.node_child_indices,
)
```

Or via the distribution. By default `LeafCTMC` **auto-detects** the native op
and enables **adaptive** rescaling, so you get the fast native path with safe
fallbacks out of the box:

```python
from treeflow.distributions.leaf_ctmc import LeafCTMC
# Defaults: use_native="auto" (native if built, else TensorFlow),
#           rescaling="adaptive" (rescale only if the unscaled value underflows).
dist = LeafCTMC(transition_probs_tree, frequencies)

# Override explicitly if desired:
#   use_native: "auto" | True | False
#   rescaling:  False | True | "auto" | "adaptive"
dist = LeafCTMC(transition_probs_tree, frequencies, use_native=True, rescaling="auto")
```

`treeflow.distributions.leaf_ctmc.native_acceleration_available()` reports
whether the native op can be loaded.

## Tests

```bash
pytest -m native
```

The tests check the native op against the reference implementation (forward
values and autodiff gradients), against finite differences, and against the
known HKY log-likelihood of the `hello` dataset. They auto-skip if the op cannot
be built (e.g. no C++ compiler available).
