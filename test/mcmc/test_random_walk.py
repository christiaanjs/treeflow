"""Tests for the random-walk Metropolis-Hastings reference sampler."""
import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability.python.distributions as tfd
import yaml
from numpy.testing import assert_allclose

from treeflow import DEFAULT_FLOAT_DTYPE_TF
from treeflow.distributions.tree.coalescent.constant_coalescent import (
    ConstantCoalescent,
)
from treeflow.mcmc.random_walk import (
    DEFAULT_TARGET_ACCEPTANCE_RATE,
    check_effective_sample_size,
    effective_sample_size_summary,
    fit_fixed_topology_random_walk_metropolis,
)
from treeflow.model.phylo_model import (
    DEFAULT_TREE_VAR_NAME,
    PhyloModel,
    phylo_model_to_joint_distribution,
)
from treeflow.tree.rooted.tensorflow_rooted_tree import TensorflowRootedTree

TREE_NAME = "tree_dist_name"
_constant = lambda x: tf.constant(x, dtype=DEFAULT_FLOAT_DTYPE_TF)


@pytest.fixture
def prior_only_model(hello_tensor_tree):
    """A model whose likelihood term carries no information, so the posterior is
    the prior -- which makes the sampler's stationary distribution checkable."""
    tree = hello_tensor_tree
    return tfd.JointDistributionNamed(
        dict(
            pop_size=tfd.LogNormal(_constant(0.0), _constant(1.0)),
            tree=lambda pop_size: ConstantCoalescent(
                tree.taxon_count, pop_size, tree.sampling_times, tree_name=TREE_NAME
            ),
            obs=lambda tree: tfd.Normal(
                _constant(0.0) * tf.reduce_sum(tree.branch_lengths, axis=-1),
                _constant(1.0),
            ),
        )
    ).experimental_pin(obs=_constant(0.0))


@pytest.fixture
def phylo_model_pinned(actual_model_file, hello_tensor_tree, hello_alignment):
    with open(actual_model_file) as f:
        model_dict = yaml.safe_load(f)
    distribution = phylo_model_to_joint_distribution(
        PhyloModel(model_dict), hello_tensor_tree, hello_alignment
    )
    encoded = hello_alignment.get_encoded_sequence_tensor(
        hello_tensor_tree.taxon_set
    )
    return distribution.experimental_pin(alignment=encoded)


# ---------------------------------------------------------------------------
# Shapes and structure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_chains", [1, 3])
def test_sample_shapes(phylo_model_pinned, hello_tensor_tree, num_chains):
    num_results = 20
    result = fit_fixed_topology_random_walk_metropolis(
        phylo_model_pinned,
        topologies={DEFAULT_TREE_VAR_NAME: hello_tensor_tree.topology},
        num_results=num_results,
        num_burnin_steps=20,
        adaptation_chunk=10,
        num_chains=num_chains,
        init_state={DEFAULT_TREE_VAR_NAME: hello_tensor_tree},
        seed=(1, 2),
    )
    batch_shape = (num_results,) if num_chains == 1 else (num_results, num_chains)

    flat_names = phylo_model_pinned._flat_resolve_names()
    flat_samples = phylo_model_pinned._model_flatten(result.samples)
    for name, sample in zip(flat_names, flat_samples):
        if isinstance(sample, TensorflowRootedTree):
            assert tuple(sample.node_heights.shape)[: len(batch_shape)] == batch_shape
            assert tf.reduce_all(tf.math.is_finite(sample.node_heights))
        else:
            assert tuple(sample.shape)[: len(batch_shape)] == batch_shape
            assert tf.reduce_all(tf.math.is_finite(sample)), name


def test_tree_topology_and_sampling_times_preserved(
    phylo_model_pinned, hello_tensor_tree
):
    result = fit_fixed_topology_random_walk_metropolis(
        phylo_model_pinned,
        topologies={DEFAULT_TREE_VAR_NAME: hello_tensor_tree.topology},
        num_results=10,
        num_burnin_steps=10,
        adaptation_chunk=5,
        seed=(3, 4),
    )
    tree = phylo_model_pinned._model_flatten(result.samples)[
        phylo_model_pinned._flat_resolve_names().index(DEFAULT_TREE_VAR_NAME)
    ]
    assert tf.reduce_all(
        tree.topology.parent_indices == hello_tensor_tree.topology.parent_indices
    )
    assert_allclose(
        tree.sampling_times.numpy(), hello_tensor_tree.sampling_times.numpy()
    )


def test_reproducible_with_seed(prior_only_model, hello_tensor_tree):
    def run():
        return fit_fixed_topology_random_walk_metropolis(
            prior_only_model,
            topologies={TREE_NAME: hello_tensor_tree.topology},
            num_results=50,
            num_burnin_steps=50,
            adaptation_chunk=25,
            seed=(5, 6),
        )

    assert_allclose(
        run().samples["pop_size"].numpy(), run().samples["pop_size"].numpy()
    )


# ---------------------------------------------------------------------------
# Scale adaptation
# ---------------------------------------------------------------------------


def test_adaptation_reaches_target_acceptance(prior_only_model, hello_tensor_tree):
    """Starting from a far too large proposal, burn-in should bring the
    acceptance rate near the 0.234 target."""
    result = fit_fixed_topology_random_walk_metropolis(
        prior_only_model,
        topologies={TREE_NAME: hello_tensor_tree.topology},
        num_results=500,
        num_burnin_steps=1_000,
        scale=20.0,
        adaptation_chunk=50,
        seed=(7, 8),
    )
    assert result.scale.numpy() < 20.0
    assert (
        abs(float(result.acceptance_rate) - DEFAULT_TARGET_ACCEPTANCE_RATE) < 0.12
    )


def test_no_adaptation_keeps_scale(prior_only_model, hello_tensor_tree):
    result = fit_fixed_topology_random_walk_metropolis(
        prior_only_model,
        topologies={TREE_NAME: hello_tensor_tree.topology},
        num_results=50,
        num_burnin_steps=50,
        scale=0.3,
        adapt_scale=False,
        seed=(9, 10),
    )
    assert_allclose(result.scale.numpy(), 0.3)


def test_base_scales_are_applied(prior_only_model, hello_tensor_tree):
    """A per-variable multiplier of zero freezes that variable."""
    result = fit_fixed_topology_random_walk_metropolis(
        prior_only_model,
        topologies={TREE_NAME: hello_tensor_tree.topology},
        num_results=30,
        num_burnin_steps=0,
        scale=0.5,
        base_scales=[0.0, 1.0],  # names are sorted: pop_size, tree
        adapt_scale=False,
        init_state=dict(tree=hello_tensor_tree),
        seed=(11, 12),
    )
    assert result.names[0] == "pop_size"
    pop_size = result.samples["pop_size"].numpy()
    assert pop_size.std() == 0.0


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def test_effective_sample_size_and_r_hat(prior_only_model, hello_tensor_tree):
    result = fit_fixed_topology_random_walk_metropolis(
        prior_only_model,
        topologies={TREE_NAME: hello_tensor_tree.topology},
        num_results=400,
        num_burnin_steps=200,
        num_chains=3,
        adaptation_chunk=50,
        seed=(13, 14),
    )
    summary = effective_sample_size_summary(result)
    assert set(summary) == set(result.names)
    for name, entry in summary.items():
        assert 0 < entry["min ESS"] <= result.num_results * result.num_chains
        assert entry["max R-hat"] > 0.99
        assert np.isfinite(entry["ESS per sample"])

    # A lenient threshold: this checks the diagnostic plumbing, not whether this
    # deliberately short chain has converged.
    passed, checked = check_effective_sample_size(result, min_ess=1.0, max_r_hat=2.0)
    assert passed
    assert checked == summary


def test_check_effective_sample_size_flags_short_chain(
    prior_only_model, hello_tensor_tree
):
    result = fit_fixed_topology_random_walk_metropolis(
        prior_only_model,
        topologies={TREE_NAME: hello_tensor_tree.topology},
        num_results=30,
        num_burnin_steps=30,
        adaptation_chunk=15,
        seed=(15, 16),
    )
    assert result.potential_scale_reduction is None  # single chain
    with pytest.warns(UserWarning, match="has not mixed well enough"):
        passed, _ = check_effective_sample_size(result, min_ess=10_000)
    assert not passed
    with pytest.raises(RuntimeError, match="has not mixed well enough"):
        check_effective_sample_size(result, min_ess=10_000, raise_on_failure=True)


# ---------------------------------------------------------------------------
# Correctness: the chain targets the right distribution
# ---------------------------------------------------------------------------


def test_recovers_prior_when_likelihood_is_uninformative(
    prior_only_model, hello_tensor_tree
):
    """With a likelihood that does not depend on the parameters, the posterior is
    the prior, so the sampler's marginal for ``pop_size`` must match
    ``LogNormal(0, 1)`` to within Monte Carlo error."""
    result = fit_fixed_topology_random_walk_metropolis(
        prior_only_model,
        topologies={TREE_NAME: hello_tensor_tree.topology},
        num_results=4_000,
        num_burnin_steps=1_000,
        num_chains=4,
        adaptation_chunk=100,
        seed=(17, 18),
    )
    log_pop_size = np.log(result.samples["pop_size"].numpy()).reshape(-1)
    ess = float(result.effective_sample_size["pop_size"].numpy().min())
    assert ess > 100, f"chain too short to test the marginal (ESS {ess:.0f})"

    # log(pop_size) ~ Normal(0, 1) under the prior.
    standard_error = 1.0 / np.sqrt(ess)
    assert abs(log_pop_size.mean()) < 4 * standard_error
    assert abs(log_pop_size.std() - 1.0) < 4 * standard_error


# ---------------------------------------------------------------------------
# Thinning and preconditioning
# ---------------------------------------------------------------------------


def test_thinning_keeps_requested_number_of_samples(
    prior_only_model, hello_tensor_tree
):
    """``thin`` runs more steps per kept sample without changing the output
    shape -- which is how a long reference run stays within memory."""
    result = fit_fixed_topology_random_walk_metropolis(
        prior_only_model,
        topologies={TREE_NAME: hello_tensor_tree.topology},
        num_results=200,
        num_burnin_steps=200,
        num_chains=2,
        thin=5,
        adaptation_chunk=50,
        seed=(19, 20),
    )
    assert result.unconstrained_samples["pop_size"].shape[:2] == (200, 2)
    thinned_ess = result.effective_sample_size["pop_size"].numpy().min()

    unthinned = fit_fixed_topology_random_walk_metropolis(
        prior_only_model,
        topologies={TREE_NAME: hello_tensor_tree.topology},
        num_results=200,
        num_burnin_steps=200,
        num_chains=2,
        thin=1,
        adaptation_chunk=50,
        seed=(19, 20),
    )
    # Thinning spreads the same number of kept samples over 5x as many steps, so
    # they are less autocorrelated: more effective samples for the same storage.
    assert thinned_ess > unthinned.effective_sample_size["pop_size"].numpy().min()


def test_preconditioning_uses_per_coordinate_scales(
    prior_only_model, hello_tensor_tree
):
    """Preconditioning replaces the scalar proposal scale with a per-coordinate
    one, and leaves the sampler targeting the same distribution."""
    kwargs = dict(
        topologies={TREE_NAME: hello_tensor_tree.topology},
        num_results=500,
        num_burnin_steps=400,
        num_chains=2,
        adaptation_chunk=50,
        seed=(21, 22),
    )
    preconditioned = fit_fixed_topology_random_walk_metropolis(
        prior_only_model, precondition=True, **kwargs
    )
    plain = fit_fixed_topology_random_walk_metropolis(
        prior_only_model, precondition=False, **kwargs
    )
    # Different proposals, but both target the prior, so the marginals agree to
    # within their (generous, given these short chains) Monte Carlo error.
    for result in (preconditioned, plain):
        log_pop_size = np.log(result.samples["pop_size"].numpy())
        assert abs(log_pop_size.mean()) < 0.5
    assert not np.allclose(
        float(preconditioned.scale), float(plain.scale)
    ) or not np.allclose(
        preconditioned.samples["pop_size"].numpy(),
        plain.samples["pop_size"].numpy(),
    )
