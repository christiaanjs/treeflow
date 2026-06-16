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
@pytest.mark.parametrize("unroll", ["unrolled", "while_loop"])
def test_fit_fixed_topology_native_and_unroll(
    actual_model_file, hello_tensor_tree, hello_alignment, use_native, unroll
):
    """Fixed-topology VI runs through both the native and pure-TensorFlow engines,
    with the traversals either fully unrolled or run as a dynamic while_loop.

    With ``unroll="unrolled"`` and ``use_native=False`` this also *verifies unrolling*:
    that mode raises if the (captured) topology index values are not statically
    foldable, so a successful fit means the likelihood and ratio-transform traversals
    were unrolled inside the traced VI step.
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
    # Completing all steps is the end-to-end check: with unroll="unrolled" the
    # traversal drivers would have *raised* during the traced fit if the topology
    # weren't statically foldable, so reaching here means the traversals were unrolled.
    # (ELBO finiteness is a separate, numerically-fragile VI concern, not asserted.)
    assert tuple(results.loss.shape) == (num_steps,)


@pytest.mark.parametrize("unroll", ["unrolled", "while_loop"])
def test_fit_fixed_topology_static_numpy_topology(
    actual_model_file, hello_tensor_tree, hello_alignment, unroll
):
    """Fixed-topology VI accepts a static NumPy topology pin.

    The bijector keeps the static topology for the ratio transform (whose traversal
    folds the static indices and unrolls when ``unroll="unrolled"``) and rebuilds it
    as an in-graph-constant TensorflowTreeTopology for the tree value the
    JointDistribution consumes. Completing all steps with ``unroll="unrolled"`` means
    the traversals were unrolled (it raises otherwise)."""
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
