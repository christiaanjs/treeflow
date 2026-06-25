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

## Conditional clade topology ops

Compiled ops for the conditional clade (subsplit Bayesian network) distribution
over rooted topologies in
[`treeflow/conditional_clade/`](../../conditional_clade/), the drop-in
counterparts of the pure-TensorFlow `tf.while_loop` reference in
[`treeflow/conditional_clade/tensor_ops.py`](../../conditional_clade/tensor_ops.py).

The recursive, data-dependent structure of a topology — which clades get
expanded depends on which subsplits were sampled — is awkward to express with
`tf.while_loop` but natural as ordinary C++ recursion, so these kernels are both
simpler and faster than the graph-mode reference while producing identical
results.

* `ConditionalCladeSample` — samples `parent_indices` (one independent topology
  per seed row) from the per-subsplit logits, via a recursive expansion that
  assigns internal node ids in post-order.
* `ConditionalCladeLogProb` — the log-probability of a topology, plus the chosen
  flat subsplit indices saved for the backward pass; `ConditionalCladeLogProbGrad`
  is the analytic (scatter-add) gradient w.r.t. the conditional log-probabilities,
  wired in with `@tf.RegisterGradient("ConditionalCladeLogProb")`.
* `ParentIndicesToChildIndices` / `ChildIndicesToPreorder` — the topology index
  transforms (deriving `child_indices` and the pre-order traversal).

They are consumed through `ConditionalCladeTreeDistribution(..., use_native=...)`
(`"auto"` by default: native if the library can be loaded, else the
pure-TensorFlow path), so `sample` and `log_prob` transparently use the compiled
ops inside `tf.function`. The lower-level wrappers are in
[`conditional_clade.py`](conditional_clade.py)
(`native_sample_parent_indices`, `native_topology_log_prob`,
`native_parent_indices_to_child_indices`, `native_child_indices_to_preorder`).

## Building

```bash
bash treeflow/acceleration/native/build.sh                  # all ops
bash treeflow/acceleration/native/build.sh node_height_ratio_op  # just one
bash treeflow/acceleration/native/build.sh conditional_clade_op  # the clade ops
# or
python -m treeflow.acceleration.native.build
```

This compiles each `cc/<op>.cc` into the matching `_<op>.so` (e.g.
`_phylo_likelihood_op.so`, `_node_height_ratio_op.so`) using the compile/link
flags reported by the installed TensorFlow (so the C++ ABI matches the running
runtime). The `.so` files are intentionally git-ignored — they are
environment-specific and must be built against the local TensorFlow.

### Docker

The `Dockerfile` builds the op as part of the image: it installs a C++ compiler
in a single layer, runs `build.sh`, then removes the compiler again, and
`pip install .` copies the resulting `.so` into site-packages via
`package_data`. The runtime image therefore ships the native op with no build
toolchain. The GitHub Actions `pytest` workflow builds the `test` image and runs
the suite, so the native op (and its tests) are exercised in CI.

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
