"""Tests for the tree normalising flow variational approximation."""
import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability.python.distributions as tfd
from numpy.testing import assert_allclose

from treeflow import DEFAULT_FLOAT_DTYPE_TF
from treeflow.distributions.tree.coalescent.constant_coalescent import (
    ConstantCoalescent,
)
from treeflow.model.approximation import get_fixed_topology_tree_flow_approximation
from treeflow.tree.io import parse_newick
from treeflow.tree.rooted.tensorflow_rooted_tree import convert_tree_to_tensor

TREE_NAME = "tree_dist_name"

_constant = lambda x: tf.constant(x, dtype=DEFAULT_FLOAT_DTYPE_TF)


@pytest.fixture
def tree(hello_newick_file):
    return convert_tree_to_tensor(parse_newick(hello_newick_file))


@pytest.fixture
def model(tree):
    """A coalescent tree with a global parameter and a per-branch parameter."""
    branch_count = tree.branch_lengths.shape[-1]
    return tfd.JointDistributionNamed(
        dict(
            pop_size=tfd.LogNormal(_constant(0.0), _constant(1.0)),
            rates=tfd.Sample(
                tfd.LogNormal(_constant(0.0), _constant(1.0)), branch_count
            ),
            tree=lambda pop_size: ConstantCoalescent(
                tree.taxon_count,
                pop_size,
                tree.sampling_times,
                tree_name=TREE_NAME,
            ),
            obs=lambda tree, rates: tfd.Normal(
                tf.reduce_sum(tree.branch_lengths * rates, axis=-1), _constant(1.0)
            ),
        )
    ).experimental_pin(obs=_constant(10.0))


def build(model, tree, **kwargs):
    kwargs.setdefault("topology_pins", {TREE_NAME: tree.topology})
    return get_fixed_topology_tree_flow_approximation(
        model, dtype=DEFAULT_FLOAT_DTYPE_TF, **kwargs
    )


@pytest.mark.parametrize(
    "parameter_approximation", ["mean_field", "full_rank", "iaf"]
)
def test_sample_and_log_prob(model, tree, parameter_approximation):
    approximation, variables_dict = build(
        model,
        tree,
        init_loc=dict(tree=tree),
        parameter_approximation=parameter_approximation,
        seed=(1, 2),
    )
    sample = approximation.sample(4, seed=(3, 4))
    assert (
        tf.reduce_all(
            sample["tree"].topology.parent_indices == tree.topology.parent_indices
        )
        .numpy()
        .item()
    )
    assert_allclose(
        sample["tree"].sampling_times.numpy(),
        tree.sampling_times.numpy(),
        atol=1e-12,
    )
    model_log_prob = model.unnormalized_log_prob(sample)
    approximation_log_prob = approximation.log_prob(sample)
    assert np.isfinite(model_log_prob.numpy()).all()
    assert np.isfinite(approximation_log_prob.numpy()).all()
    assert variables_dict


@pytest.mark.parametrize("nonlinearity", ["spline", "sinh_arcsinh", "affine"])
def test_nonlinearity_choices(model, tree, nonlinearity):
    """Every nonlinearity gives a usable approximation, including "affine",
    which makes the tree block a tree-structured Gaussian."""
    approximation, variables_dict = build(
        model, tree, init_loc=dict(tree=tree), nonlinearity=nonlinearity
    )
    sample, log_prob = approximation.experimental_sample_and_log_prob(
        4, seed=(18, 19)
    )
    assert np.isfinite(log_prob.numpy()).all()
    assert np.isfinite(model.unnormalized_log_prob(sample).numpy()).all()
    shared = [name for name in variables_dict if "nonlinearity" in name]
    assert len(shared) == (0 if nonlinearity == "affine" else (3 if nonlinearity == "spline" else 2))


def test_sample_and_log_prob_consistency(model, tree):
    approximation, _ = build(model, tree, init_loc=dict(tree=tree), num_layers=2)
    sample, log_prob = approximation.experimental_sample_and_log_prob(
        4, seed=(5, 6)
    )
    assert_allclose(
        log_prob.numpy(), approximation.log_prob(sample).numpy(), rtol=1e-9
    )


def test_identity_at_initialisation(model, tree):
    """The flow starts as the identity, so at initialisation the approximation
    does not depend on how deep the flow is -- with the same base draw, a
    one-layer and a three-layer flow must produce the same sample and the same
    density (and likewise whichever family the parameter block uses, since
    those coincide at initialisation too)."""
    seed = (7, 8)
    samples, log_probs = [], []
    for kwargs in (
        dict(num_layers=1),
        dict(num_layers=3),
        dict(num_layers=1, parameter_approximation="full_rank"),
    ):
        approximation, _ = build(model, tree, init_loc=dict(tree=tree), **kwargs)
        sample, log_prob = approximation.experimental_sample_and_log_prob(
            8, seed=seed
        )
        samples.append(sample)
        log_probs.append(log_prob)
    for sample, log_prob in zip(samples[1:], log_probs[1:]):
        assert_allclose(
            sample["tree"].node_heights.numpy(),
            samples[0]["tree"].node_heights.numpy(),
            rtol=1e-6,
        )
        assert_allclose(sample["pop_size"].numpy(), samples[0]["pop_size"].numpy())
        assert_allclose(log_prob.numpy(), log_probs[0].numpy(), rtol=1e-6)


def test_conditioning_variables(model, tree):
    """Auxiliary and per-node conditioning are wired up: perturbing the flow's
    conditioner weights changes the tree marginal but not the parameter one."""
    approximation, variables_dict = build(
        model,
        tree,
        init_loc=dict(tree=tree),
        auxiliary_vars=("pop_size",),
        node_feature_vars=("rates",),
    )
    seed = (9, 10)
    before = approximation.sample(8, seed=seed)
    rng = np.random.default_rng(11)
    conditioner_names = [
        name
        for name in variables_dict
        if "auxiliary_output_kernel" in name or "node_output_kernel" in name
    ]
    assert len(conditioner_names) == 2
    for name in conditioner_names:
        variable = variables_dict[name]
        variable.assign(
            variable + _constant(rng.normal(scale=0.5, size=variable.shape))
        )
    after = approximation.sample(8, seed=seed)
    assert (
        np.abs(
            before["tree"].node_heights.numpy() - after["tree"].node_heights.numpy()
        ).max()
        > 1e-6
    )
    assert_allclose(before["pop_size"].numpy(), after["pop_size"].numpy(), rtol=1e-12)


def test_gradients_reach_every_variable(model, tree):
    approximation, variables_dict = build(
        model,
        tree,
        init_loc=dict(tree=tree),
        parameter_approximation="iaf",
        node_feature_vars=("rates",),
        num_layers=2,
        seed=(12, 13),
    )
    variables = list(variables_dict.values())
    with tf.GradientTape() as tape:
        sample, log_prob = approximation.experimental_sample_and_log_prob(
            4, seed=(14, 15)
        )
        elbo = tf.reduce_mean(model.unnormalized_log_prob(sample) - log_prob)
    gradients = tape.gradient(elbo, variables)
    assert all(gradient is not None for gradient in gradients)
    assert np.isfinite(elbo.numpy())


def test_improves_elbo(model, tree):
    """A short optimisation run: the flow trains, and does not do worse than
    where it started (the mean-field approximation)."""
    from treeflow.vi import fit_fixed_topology_variational_approximation

    _, trace = fit_fixed_topology_variational_approximation(
        model,
        topologies={TREE_NAME: tree.topology},
        optimizer=tf.optimizers.Adam(0.02),
        num_steps=300,
        init_loc=dict(tree=tree),
        approx_fn=get_fixed_topology_tree_flow_approximation,
        sample_size=8,
        seed=(16, 17),
    )
    loss = np.asarray(trace.loss)
    assert np.isfinite(loss).all()
    assert loss[-50:].mean() < loss[:50].mean()


def test_requires_single_topology(model, tree):
    with pytest.raises(ValueError, match="exactly one pinned topology"):
        build(model, tree, topology_pins={})


def test_rejects_unknown_conditioning_variable(model, tree):
    with pytest.raises(ValueError, match="not non-tree model variables"):
        build(model, tree, auxiliary_vars=("not_a_variable",))


def test_rejects_bad_parameter_approximation(model, tree):
    with pytest.raises(ValueError, match="parameter_approximation"):
        build(model, tree, parameter_approximation="laplace")


# ---------------------------------------------------------------------------
# Linear cross-coupling between the parameter and tree blocks
# ---------------------------------------------------------------------------


def test_linear_cross_coupling_is_identity_at_initialisation(model, tree):
    """The cross term starts at zero, so it does not disturb the identity
    initialisation."""
    seed = (20, 21)
    samples = []
    for linear_cross_coupling in (False, True):
        approximation, variables_dict = build(
            model,
            tree,
            init_loc=dict(tree=tree),
            linear_cross_coupling=linear_cross_coupling,
        )
        samples.append(approximation.sample(8, seed=seed))
        has_cross = any("cross_weight" in name for name in variables_dict)
        assert has_cross == linear_cross_coupling
    assert_allclose(
        samples[1]["tree"].node_heights.numpy(),
        samples[0]["tree"].node_heights.numpy(),
        rtol=1e-9,
    )


def test_linear_cross_coupling_correlates_parameters_with_heights(model, tree):
    """A non-zero cross weight makes the tree coordinates correlate with the
    parameter block -- the dependence a mean-field tree block cannot have.

    Measured on the unconstrained coordinates, where the cross term acts and the
    relationship it induces is exactly linear; the constraining transforms
    downstream (a sigmoid on the ratios) squash it by an amount that depends on
    where each coordinate sits, which would only blur the check.
    """
    approximation, variables_dict = build(
        model,
        tree,
        init_loc=dict(tree=tree),
        auxiliary_vars=(),  # no nonlinear conditioner: the cross term is the only coupling
        linear_cross_coupling=True,
    )
    (cross_name,) = [name for name in variables_dict if "cross_weight" in name]
    cross_weight = variables_dict[cross_name]
    coupling = approximation.bijector.bijectors[-1]
    tree_size = int(tree.taxon_count) - 1

    def unconstrained_correlation():
        base = approximation.distribution.sample(4_000, seed=(22, 23))
        unconstrained = coupling.forward(base).numpy()
        pop_size = unconstrained[:, 0]  # first by sorted variable name
        return np.abs(
            [
                np.corrcoef(pop_size, unconstrained[:, -tree_size + i])[0, 1]
                for i in range(tree_size)
            ]
        ).mean()

    assert unconstrained_correlation() < 0.05
    targeted = np.zeros(cross_weight.shape)
    targeted[0, :] = 3.0  # carry pop_size, and only pop_size, into every node
    cross_weight.assign(_constant(targeted))
    assert unconstrained_correlation() > 0.9


def test_structured_joint_gaussian_is_linear(model, tree):
    """Full-rank parameters, an affine tree block and the linear cross term make
    the whole unconstrained map linear -- a joint Gaussian with structured
    covariance."""
    approximation, variables_dict = build(
        model,
        tree,
        init_loc=dict(tree=tree),
        parameter_approximation="full_rank",
        nonlinearity="affine",
        auxiliary_vars=(),
        linear_cross_coupling=True,
    )
    rng = np.random.default_rng(25)
    for name, variable in variables_dict.items():
        if "scale_raw" not in name:
            variable.assign(
                variable + _constant(rng.normal(scale=0.3, size=variable.shape))
            )

    # The composition from the standard-normal base to unconstrained space is
    # linear, so its Jacobian does not depend on where it is evaluated.
    unconstrained = approximation.bijector.bijectors[-1]  # the coupling bijector
    total_dim = approximation.distribution.event_shape[0]

    def jacobian_at(value):
        base = _constant(np.full(total_dim, value))
        with tf.GradientTape() as tape:
            tape.watch(base)
            transformed = unconstrained.forward(base)
        return tape.jacobian(transformed, base).numpy()

    assert_allclose(jacobian_at(-1.0), jacobian_at(1.0), rtol=1e-9, atol=1e-12)
    # ... and it really does couple the blocks: the tree rows have non-zero
    # entries in the parameter columns.
    jacobian = jacobian_at(0.0)
    parameter_size = total_dim - (int(tree.taxon_count) - 1)
    assert np.abs(jacobian[parameter_size:, :parameter_size]).max() > 1e-6
