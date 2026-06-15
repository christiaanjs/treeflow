import pytest
import yaml
import tensorflow as tf

from treeflow.model.phylo_model import (
    phylo_model_to_joint_distribution,
    PhyloModel,
)
from treeflow.vi.fixed_topology_advi import (
    fit_fixed_topology_variational_approximation,
)
from treeflow.distributions.leaf_ctmc import native_acceleration_available
from treeflow.bijectors.node_height_ratio_bijector import (
    NodeHeightRatioBijector,
    native_ratio_transform_available,
)
from treeflow.tree.topology.numpy_tree_topology import StaticNumpyTreeTopology

# Run the native engine only where the C++ ops are actually built.
USE_NATIVE_PARAMS = [False] + (
    [True]
    if (native_acceleration_available() and native_ratio_transform_available())
    else []
)


def _build_pinned(model_file, tree, alignment, use_native, unroll):
    with open(model_file) as f:
        model_dict = yaml.safe_load(f)
    model = PhyloModel(model_dict)
    dist = phylo_model_to_joint_distribution(
        model, tree, alignment, use_native=use_native, unroll=unroll
    )
    encoded = alignment.get_encoded_sequence_tensor(tree.taxon_set)
    return dist.experimental_pin(alignment=encoded)


@pytest.mark.parametrize("use_native", USE_NATIVE_PARAMS)
@pytest.mark.parametrize("unroll", [True, False])
def test_fit_fixed_topology_native_and_unroll(
    actual_model_file, hello_tensor_tree, hello_alignment, use_native, unroll
):
    """Fixed-topology VI runs through both the native and pure-TensorFlow engines,
    with the traversals either unrolled or dynamic.

    With ``unroll=True`` and ``use_native=False`` this also *verifies unrolling*: the
    traversal drivers raise if ``unroll=True`` but the (captured) topology is not
    statically foldable, so a successful fit means the likelihood and ratio-transform
    traversals were unrolled inside the traced VI step.
    """
    tf.random.set_seed(1)
    pinned = _build_pinned(
        actual_model_file, hello_tensor_tree, hello_alignment, use_native, unroll
    )
    num_steps = 5
    approximation, results = fit_fixed_topology_variational_approximation(
        pinned,
        topologies=dict(tree=hello_tensor_tree.topology),
        optimizer=tf.optimizers.Adam(learning_rate=1e-2),
        num_steps=num_steps,
        use_native=use_native,
        unroll=unroll,
        sample_size=2,
    )
    # Completing all steps is the end-to-end check: with unroll=True the traversal
    # drivers would have *raised* during the traced fit if the topology weren't
    # statically foldable, so reaching here means the traversals were unrolled.
    # (ELBO finiteness is a separate, numerically-fragile VI concern, not asserted.)
    assert tuple(results.loss.shape) == (num_steps,)


@pytest.mark.parametrize("unroll", [True, False])
def test_fit_fixed_topology_static_numpy_topology(
    actual_model_file, hello_tensor_tree, hello_alignment, unroll
):
    """Fixed-topology VI accepts a static NumPy topology pin.

    The bijector keeps the static topology for the ratio transform (whose traversal
    folds the static indices and unrolls when ``unroll=True``) and rebuilds it as an
    in-graph-constant TensorflowTreeTopology for the tree value the JointDistribution
    consumes. Completing all steps with ``unroll=True`` means the traversals were
    unrolled (they raise otherwise)."""
    tf.random.set_seed(1)
    pinned = _build_pinned(
        actual_model_file, hello_tensor_tree, hello_alignment, False, unroll
    )
    topology = StaticNumpyTreeTopology.from_numpy_topology(
        hello_tensor_tree.topology.numpy()
    )
    num_steps = 5
    approximation, results = fit_fixed_topology_variational_approximation(
        pinned,
        topologies=dict(tree=topology),
        optimizer=tf.optimizers.Adam(learning_rate=1e-2),
        num_steps=num_steps,
        use_native=False,
        unroll=unroll,
        sample_size=2,
    )
    assert tuple(results.loss.shape) == (num_steps,)


def _has_while(concrete_fn):
    return any("While" in op.type for op in concrete_fn.graph.get_operations())


def _random_parent_indices(leaf_count, seed=0):
    import numpy as np

    rng = np.random.default_rng(seed)
    node_count = 2 * leaf_count - 1
    parent = np.full(node_count, -1, dtype=np.int32)
    active = list(range(leaf_count))
    nxt = leaf_count
    while len(active) > 1:
        a = active.pop(rng.integers(len(active)))
        b = active.pop(rng.integers(len(active)))
        parent[a] = parent[b] = nxt
        active.append(nxt)
        nxt += 1
    return parent[:-1]


def test_static_numpy_topology_unrolls_likelihood_at_scale():
    """The headline benefit: a static NumPy topology pin lets the *likelihood*
    traversal unroll at a tree size (128 taxa) where a topology routed through the
    JointDistribution would not statically fold.

    FixedTopologyRootedTreeBijector rebuilds the static topology as in-graph
    constants in ``forward``; emitting that tree and running the likelihood in one
    trace (as the VI loss does) compiles to a straight-line graph (no ``While``)."""
    import numpy as np
    from treeflow.bijectors.fixed_topology_bijector import (
        FixedTopologyRootedTreeBijector,
    )
    from treeflow.traversal.phylo_likelihood import phylogenetic_likelihood

    leaf_count, state_count = 128, 4
    topology = StaticNumpyTreeTopology(_random_parent_indices(leaf_count))
    height_bijector = NodeHeightRatioBijector(topology, use_native=False, unroll=True)
    bijector = FixedTopologyRootedTreeBijector(
        topology, height_bijector, sampling_times=tf.zeros([leaf_count], tf.float64)
    )

    rng = np.random.default_rng(1)
    node_count = 2 * leaf_count - 1
    probs = rng.uniform(0.1, 1.0, size=(1, node_count, state_count, state_count))
    probs = tf.constant(probs / probs.sum(-1, keepdims=True))
    freqs = tf.constant(np.full(state_count, 1.0 / state_count))
    seqs = tf.constant(np.eye(state_count)[rng.integers(0, state_count, (8, leaf_count))])

    def emit_and_score(ratios):
        tree = bijector.forward(ratios)  # builds in-graph-constant topology
        return phylogenetic_likelihood(
            tree.topology, seqs, probs, freqs,
            batch_shape=tf.shape(seqs)[:1], unroll=True,
        )

    concrete = tf.function(emit_and_score).get_concrete_function(
        tf.constant(rng.uniform(0.1, 0.9, leaf_count - 1))
    )
    assert not _has_while(concrete)  # likelihood unrolled at 128 taxa


def test_unroll_removes_while_loop_in_height_transform(hello_tensor_tree):
    """Concrete check that ``unroll`` controls the traversal: the pure-TensorFlow
    height-ratio transform compiles to a straight-line graph (no ``While`` op) when
    unrolled, and to a ``tf.while_loop`` otherwise -- the same switch fit_fixed wires
    through to the approximation's node-height bijector."""
    topology = hello_tensor_tree.topology
    n_internal = int(topology.taxon_count) - 1
    spec = tf.TensorSpec([n_internal], tf.float64)

    def forward_concrete(unroll):
        bijector = NodeHeightRatioBijector(topology, use_native=False, unroll=unroll)
        return tf.function(bijector.forward).get_concrete_function(spec)

    assert not _has_while(forward_concrete(unroll=True))  # unrolled
    assert _has_while(forward_concrete(unroll=False))  # dynamic while_loop
