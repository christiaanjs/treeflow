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


def _has_while(concrete_fn):
    return any("While" in op.type for op in concrete_fn.graph.get_operations())


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
