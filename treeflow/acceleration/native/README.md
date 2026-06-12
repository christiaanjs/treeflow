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

## Building

```bash
bash treeflow/acceleration/native/build.sh
# or
python -m treeflow.acceleration.native.build
```

This compiles `cc/phylo_likelihood_op.cc` into `_phylo_likelihood_op.so` using
the compile/link flags reported by the installed TensorFlow (so the C++ ABI
matches the running runtime). The `.so` is intentionally git-ignored — it is
environment-specific and must be built against the local TensorFlow.

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

Or via the distribution:

```python
from treeflow.distributions.leaf_ctmc import LeafCTMC
# use_native: native C++ op; rescaling: False | True | "auto" | "adaptive"
dist = LeafCTMC(transition_probs_tree, frequencies, use_native=True, rescaling="auto")
```

## Tests

```bash
pytest -m native
```

The tests check the native op against the reference implementation (forward
values and autodiff gradients), against finite differences, and against the
known HKY log-likelihood of the `hello` dataset. They auto-skip if the op cannot
be built (e.g. no C++ compiler available).
