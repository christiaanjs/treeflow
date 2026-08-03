"""Tests for the tree normalising flow bijector.

Covers the properties the flow's usefulness rests on: it is exactly the
identity at initialisation, it is invertible with an exact log-det-Jacobian,
its dependence structure really is tree-wide in both directions (which is what
the traversal sandwich buys over either affine map alone), and it responds to
its auxiliary and per-node conditioning inputs.
"""
import numpy as np
import pytest
import tensorflow as tf
from numpy.testing import assert_allclose

from treeflow import DEFAULT_FLOAT_DTYPE_TF
from treeflow.bijectors.elementwise_node_flow import NONLINEARITIES
from treeflow.bijectors.tree_normalizing_flow import (
    ORDERS,
    TreeFlowParameters,
    TreeNormalizingFlowBijector,
)
from treeflow.tree.topology.numpy_tree_topology import NumpyTreeTopology
from treeflow.tree.topology.tensorflow_tree_topology import numpy_topology_to_tensor

TAXON_COUNT = 8
NODE_COUNT = TAXON_COUNT - 1


def _random_parent_indices(taxon_count, rng):
    node_count = 2 * taxon_count - 1
    parent = np.full(node_count, -1, dtype=np.int32)
    active = list(range(taxon_count))
    next_node = taxon_count
    while len(active) > 1:
        a = active.pop(rng.integers(len(active)))
        b = active.pop(rng.integers(len(active)))
        parent[a] = next_node
        parent[b] = next_node
        active.append(next_node)
        next_node += 1
    return parent[:-1]


def constant(x):
    return tf.constant(np.asarray(x), dtype=DEFAULT_FLOAT_DTYPE_TF)


@pytest.fixture
def topology():
    return numpy_topology_to_tensor(
        NumpyTreeTopology(
            parent_indices=_random_parent_indices(
                TAXON_COUNT, np.random.default_rng(0)
            )
        )
    )


@pytest.fixture
def x():
    return constant(np.random.default_rng(1).normal(size=(4, NODE_COUNT)))


def perturb(bijector, scale=0.3, seed=2):
    """Move the flow away from its identity initialisation."""
    rng = np.random.default_rng(seed)
    for variable in bijector.trainable_variables:
        variable.assign(variable + constant(rng.normal(scale=scale, size=variable.shape)))
    return bijector


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("nonlinearity", NONLINEARITIES)
@pytest.mark.parametrize("num_layers", [1, 3])
def test_identity_at_initialisation(topology, x, num_layers, nonlinearity):
    """A freshly built flow is the identity, so it can be dropped into an
    existing approximation without changing it."""
    bijector = TreeNormalizingFlowBijector(
        topology, num_layers=num_layers, nonlinearity=nonlinearity
    )
    assert_allclose(bijector.forward(x).numpy(), x.numpy(), rtol=1e-8, atol=1e-8)
    assert_allclose(
        bijector.forward_log_det_jacobian(x, event_ndims=1).numpy(),
        np.zeros(x.shape[0]),
        atol=1e-8,
    )


def test_identity_at_initialisation_with_conditioning(topology, x):
    """Conditioning does not break the identity initialisation: the
    conditioners' output weights start at zero."""
    auxiliary = constant(np.random.default_rng(3).normal(size=(4, 3)))
    node_input = constant(np.random.default_rng(4).normal(size=(4, NODE_COUNT, 2)))
    bijector = TreeNormalizingFlowBijector(
        topology, auxiliary_input=auxiliary, node_input=node_input
    )
    assert_allclose(bijector.forward(x).numpy(), x.numpy(), rtol=1e-8, atol=1e-8)


# ---------------------------------------------------------------------------
# Bijectivity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("nonlinearity", NONLINEARITIES)
@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("num_layers", [1, 2])
def test_roundtrip(topology, x, order, num_layers, nonlinearity):
    bijector = perturb(
        TreeNormalizingFlowBijector(
            topology, num_layers=num_layers, order=order, nonlinearity=nonlinearity
        )
    )
    y = bijector.forward(x)
    assert_allclose(bijector.inverse(y).numpy(), x.numpy(), rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("nonlinearity", NONLINEARITIES)
@pytest.mark.parametrize("order", ORDERS)
def test_log_det_jacobian_matches_determinant(topology, order, nonlinearity):
    bijector = perturb(
        TreeNormalizingFlowBijector(
            topology, num_layers=2, order=order, nonlinearity=nonlinearity
        )
    )
    x = constant(np.random.default_rng(5).normal(size=(NODE_COUNT,)))
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = bijector.forward(x)
    jacobian = tape.jacobian(y, x).numpy()
    sign, log_abs_det = np.linalg.slogdet(jacobian)
    assert sign > 0
    assert_allclose(
        log_abs_det,
        bijector.forward_log_det_jacobian(x, event_ndims=1).numpy(),
        rtol=1e-9,
    )


def test_forward_and_inverse_log_det_jacobians_agree(topology, x):
    bijector = perturb(TreeNormalizingFlowBijector(topology, num_layers=2))
    y = bijector.forward(x)
    assert_allclose(
        bijector.forward_log_det_jacobian(x, event_ndims=1).numpy(),
        -bijector.inverse_log_det_jacobian(y, event_ndims=1).numpy(),
        rtol=1e-9,
    )


def test_extreme_parameters_stay_finite_and_invertible(topology, x):
    """The bounded recursion weights keep a badly conditioned flow usable."""
    bijector = perturb(
        TreeNormalizingFlowBijector(topology, num_layers=2), scale=5.0, seed=6
    )
    y = bijector.forward(x)
    assert np.isfinite(y.numpy()).all()
    assert_allclose(bijector.inverse(y).numpy(), x.numpy(), rtol=1e-6, atol=1e-6)
    assert np.isfinite(
        bijector.forward_log_det_jacobian(x, event_ndims=1).numpy()
    ).all()


# ---------------------------------------------------------------------------
# Dependence structure
# ---------------------------------------------------------------------------


def _jacobian(bijector, seed=7):
    x = constant(np.random.default_rng(seed).normal(size=(NODE_COUNT,)))
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = bijector.forward(x)
    return tape.jacobian(y, x).numpy()


def _ancestors_and_descendants(topology):
    """Boolean matrix: node pairs on a common root-to-tip path."""
    parent = topology.parent_indices.numpy()
    taxon_count = int(topology.taxon_count)
    related = np.eye(NODE_COUNT, dtype=bool)
    for i in range(NODE_COUNT):
        node = i + taxon_count
        while node != 2 * taxon_count - 2:  # walk up to the root
            node = parent[node]
            related[i, node - taxon_count] = True
            related[node - taxon_count, i] = True
    return related


def test_sandwich_dependence_structure(topology):
    """The traversal order decides which pairs of nodes the sandwich couples.

    Postorder-first (up then down, as in belief propagation's
    collect/distribute) couples every pair in one layer; preorder-first couples
    only ancestor-descendant pairs until a second layer is stacked on. Neither
    affine map alone can do either.
    """
    postorder_first = _jacobian(
        perturb(TreeNormalizingFlowBijector(topology, order="postorder_first"))
    )
    assert (np.abs(postorder_first) > 1e-12).all()

    preorder_first = _jacobian(
        perturb(TreeNormalizingFlowBijector(topology, order="preorder_first"))
    )
    related = _ancestors_and_descendants(topology)
    assert (np.abs(preorder_first[related]) > 1e-12).all()
    assert (preorder_first[~related] == 0).all()
    assert not related.all()  # the tree does have cousins for this to be about

    two_layers = _jacobian(
        perturb(
            TreeNormalizingFlowBijector(
                topology, order="preorder_first", num_layers=2
            )
        )
    )
    assert (np.abs(two_layers) > 1e-12).all()


def _jacobian_at(bijector, value):
    x = constant(np.full(NODE_COUNT, value))
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = bijector.forward(x)
    return tape.jacobian(y, x).numpy()


@pytest.mark.parametrize("nonlinearity", ["spline", "sinh_arcsinh"])
def test_flow_is_nonlinear(topology, nonlinearity):
    """The elementwise layer makes the map genuinely nonlinear -- the Jacobian
    depends on where it is evaluated."""
    bijector = perturb(
        TreeNormalizingFlowBijector(topology, nonlinearity=nonlinearity)
    )
    assert (
        np.abs(_jacobian_at(bijector, -1.0) - _jacobian_at(bijector, 1.0)).max()
        > 1e-6
    )


def test_affine_nonlinearity_is_a_structured_gaussian_map(topology, x):
    """``nonlinearity="affine"`` drops the nonlinearity: the flow becomes a
    linear map (constant Jacobian, constant log-det-Jacobian) that still couples
    every pair of nodes -- a tree-structured Gaussian, and the ablation against
    which the spline flow's nonlinearity is measured."""
    bijector = perturb(TreeNormalizingFlowBijector(topology, nonlinearity="affine"))
    assert_allclose(
        _jacobian_at(bijector, -2.0), _jacobian_at(bijector, 2.0), rtol=1e-11
    )
    assert (np.abs(_jacobian_at(bijector, 0.0)) > 1e-12).all()
    # A linear map's log-det-Jacobian does not depend on the coordinates.
    assert_allclose(
        bijector.forward_log_det_jacobian(x, event_ndims=1).numpy(),
        bijector.forward_log_det_jacobian(x + 5.0, event_ndims=1).numpy(),
        rtol=1e-11,
    )


def test_orders_differ(topology, x):
    """Pre-order-first and post-order-first are different bijections (neither
    is a special case of the other)."""
    parameters = TreeFlowParameters(
        node_count=NODE_COUNT, dtype=DEFAULT_FLOAT_DTYPE_TF
    )
    rng = np.random.default_rng(8)
    for variable in parameters.trainable_variables:
        variable.assign(variable + constant(rng.normal(scale=0.3, size=variable.shape)))
    outputs = [
        TreeNormalizingFlowBijector(
            topology, flow_parameters=parameters, order=order
        ).forward(x)
        for order in ORDERS
    ]
    assert np.abs(outputs[0].numpy() - outputs[1].numpy()).max() > 1e-6


# ---------------------------------------------------------------------------
# Conditioning
# ---------------------------------------------------------------------------


def test_auxiliary_and_node_conditioning(topology, x):
    auxiliary = constant(np.random.default_rng(9).normal(size=(4, 3)))
    node_input = constant(np.random.default_rng(10).normal(size=(4, NODE_COUNT, 2)))
    bijector = perturb(
        TreeNormalizingFlowBijector(
            topology, auxiliary_input=auxiliary, node_input=node_input
        )
    )
    baseline = bijector.forward(x)

    # Different conditioning values give a different (but still invertible) map.
    other = bijector.copy_with_inputs(
        auxiliary_input=auxiliary + 1.0, node_input=node_input
    )
    assert other.flow_parameters is bijector.flow_parameters
    assert np.abs(other.forward(x).numpy() - baseline.numpy()).max() > 1e-6
    assert_allclose(
        other.inverse(other.forward(x)).numpy(), x.numpy(), rtol=1e-9, atol=1e-9
    )

    node_perturbed = bijector.copy_with_inputs(
        auxiliary_input=auxiliary, node_input=node_input + 1.0
    )
    assert np.abs(node_perturbed.forward(x).numpy() - baseline.numpy()).max() > 1e-6


def test_conditioning_requires_matching_parameters(topology, x):
    bijector = TreeNormalizingFlowBijector(topology)
    with pytest.raises(ValueError, match="auxiliary conditioner"):
        bijector.copy_with_inputs(auxiliary_input=constant(np.zeros((4, 3)))).forward(x)
    with pytest.raises(ValueError, match="per-node conditioner"):
        bijector.copy_with_inputs(
            node_input=constant(np.zeros((4, NODE_COUNT, 2)))
        ).forward(x)


def test_invalid_order(topology):
    with pytest.raises(ValueError, match="order must be"):
        TreeNormalizingFlowBijector(topology, order="inorder")


# ---------------------------------------------------------------------------
# Trainability
# ---------------------------------------------------------------------------


def test_gradients_reach_every_variable(topology, x):
    auxiliary = constant(np.random.default_rng(11).normal(size=(4, 3)))
    node_input = constant(np.random.default_rng(12).normal(size=(4, NODE_COUNT, 2)))
    bijector = perturb(
        TreeNormalizingFlowBijector(
            topology, auxiliary_input=auxiliary, node_input=node_input, num_layers=2
        )
    )
    variables = bijector.trainable_variables
    # base + 3 spline + 3 auxiliary conditioner + 3 per-node conditioner
    assert len(variables) == 10
    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(bijector.forward(x)) + tf.reduce_sum(
            bijector.forward_log_det_jacobian(x, event_ndims=1)
        )
    gradients = tape.gradient(loss, variables)
    assert all(gradient is not None for gradient in gradients)
    assert all(np.isfinite(gradient.numpy()).all() for gradient in gradients)


def test_transformed_distribution_density(topology):
    """Used as a variational family: sampling and its density agree with an
    independent evaluation of ``log_prob``."""
    import tensorflow_probability.python.distributions as tfd

    bijector = perturb(TreeNormalizingFlowBijector(topology, num_layers=2))
    base = tfd.Sample(
        tfd.Normal(
            tf.constant(0.0, DEFAULT_FLOAT_DTYPE_TF),
            tf.constant(1.0, DEFAULT_FLOAT_DTYPE_TF),
        ),
        NODE_COUNT,
    )
    distribution = tfd.TransformedDistribution(base, bijector)
    sample, log_prob = distribution.experimental_sample_and_log_prob(
        5, seed=(13, 14)
    )
    assert_allclose(
        log_prob.numpy(), distribution.log_prob(sample).numpy(), rtol=1e-9
    )
    assert np.isfinite(log_prob.numpy()).all()
