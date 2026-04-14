"""End-to-end integration tests for the `discrete_trait` substitution block.

Exercises the full path from a PhyloModel dict with a discrete_trait
substitution model, through joint distribution construction, to log_prob
evaluation on a small 3-taxon tree.
"""
import numpy as np
import pytest
import tensorflow as tf
from numpy.testing import assert_allclose

from treeflow.evolution.traitio import DiscreteTraitData
from treeflow.model.phylo_model import (
    DISCRETE_TRAIT_KEY,
    PhyloModel,
    PhyloModelParseError,
    get_subst_model,
    phylo_model_to_joint_distribution,
)
from treeflow.evolution.substitution.discrete_trait.dta import DiscreteTraitModel


@pytest.fixture
def hello_traits():
    """A discrete-trait assignment for the 3 hello tips."""
    return DiscreteTraitData(
        trait_mapping={"mars": "A", "saturn": "B", "jupiter": "C"},
    )


@pytest.fixture
def hello_discrete_trait_model_dict():
    """A minimal discrete_trait phylo model for the 3-taxon hello tree.

    K = 3 states (`A`, `B`, `C`) => K*(K-1)/2 = 3 exchangeability rates.
    """
    return dict(
        tree="fixed",
        clock=dict(strict=dict(clock_rate=0.5)),
        substitution=dict(
            discrete_trait=dict(
                n_states=3,
                frequencies=dict(
                    dirichlet=dict(concentration=[1.0, 1.0, 1.0]),
                ),
                rates=dict(
                    dirichlet=dict(concentration=[1.0, 1.0, 1.0]),
                ),
            )
        ),
        site="none",
    )


def test_get_subst_model_returns_dta_with_k():
    model = get_subst_model(DISCRETE_TRAIT_KEY, dict(n_states=5))
    assert isinstance(model, DiscreteTraitModel)
    assert model.n_states == 5


def test_get_subst_model_requires_n_states():
    with pytest.raises(PhyloModelParseError, match="n_states"):
        get_subst_model(DISCRETE_TRAIT_KEY, None)
    with pytest.raises(PhyloModelParseError, match="n_states"):
        get_subst_model(DISCRETE_TRAIT_KEY, {})


def test_phylo_model_parses_discrete_trait(hello_discrete_trait_model_dict):
    model = PhyloModel(hello_discrete_trait_model_dict)
    assert model.subst_model == DISCRETE_TRAIT_KEY
    assert model.subst_params["n_states"] == 3
    # Free-parameter extraction should pick up the two priors but not n_states.
    free = model.free_params()
    assert "frequencies" in free
    assert "rates" in free
    assert "n_states" not in free


def test_joint_distribution_samples_and_evaluates(
    hello_tensor_tree, hello_traits, hello_discrete_trait_model_dict
):
    model = PhyloModel(hello_discrete_trait_model_dict)
    dist = phylo_model_to_joint_distribution(
        model, hello_tensor_tree, hello_traits
    )

    sample = dist.sample()
    keys = set(sample._asdict().keys())
    # The free parameters plus the observation named "alignment" (duck-typed).
    assert "frequencies" in keys
    assert "rates" in keys
    assert "alignment" in keys

    # Sampled rates should be a 3-simplex; frequencies a 3-simplex.
    assert sample.frequencies.shape[-1] == 3
    assert sample.rates.shape[-1] == 3  # K*(K-1)/2 == 3 for K=3
    assert_allclose(
        tf.reduce_sum(sample.frequencies, axis=-1).numpy(), 1.0, atol=1e-6
    )
    assert_allclose(tf.reduce_sum(sample.rates, axis=-1).numpy(), 1.0, atol=1e-6)

    # log_prob must be finite at the sampled state.
    lp = dist.log_prob(sample)
    assert np.isfinite(lp.numpy())


def test_log_prob_at_fixed_state(
    hello_tensor_tree, hello_traits, hello_discrete_trait_model_dict
):
    """Evaluate log_prob at a concrete parameter vector and confirm finiteness
    plus observation-vs-prior decomposition.

    With a fixed sample we can pin the joint log density down and check that
    changing observations (swapping trait labels) moves the value.
    """
    model = PhyloModel(hello_discrete_trait_model_dict)
    dist = phylo_model_to_joint_distribution(
        model, hello_tensor_tree, hello_traits
    )
    sample = dist.sample(seed=(1, 2))

    # Fix the parameters, evaluate at the observed data vs a flipped-label
    # version of the observed data.
    observed_alignment = sample.alignment.numpy().copy()
    flipped_alignment = observed_alignment[:, ::-1, :].copy()  # reverse taxa

    lp_observed = dist.log_prob(sample)

    sample_flipped = sample._replace(alignment=tf.constant(flipped_alignment))
    lp_flipped = dist.log_prob(sample_flipped)

    assert np.isfinite(lp_observed.numpy())
    assert np.isfinite(lp_flipped.numpy())
    # The priors are identical; any difference comes from the likelihood.
    # With distinct tip states, flipping the observation changes likelihood
    # (and therefore the joint log-density).
    assert lp_observed.numpy() != lp_flipped.numpy()


@pytest.mark.parametrize("n_states", [3, 4, 6])
def test_discrete_trait_models_at_various_k(
    hello_tensor_tree, n_states
):
    """Build and evaluate discrete_trait models for a handful of K values.

    K=2 is skipped: the single exchange rate is degenerate after
    normalisation, so a Dirichlet rates prior is undefined.
    """
    # Build a consistent trait mapping and dimensions
    state_labels = [chr(ord("A") + i) for i in range(n_states)]
    traits = DiscreteTraitData(
        trait_mapping={
            "mars": state_labels[0],
            "saturn": state_labels[1 % n_states],
            "jupiter": state_labels[(n_states - 1) % n_states],
        },
        states=tuple(state_labels),
    )
    n_rates = n_states * (n_states - 1) // 2
    model_dict = dict(
        tree="fixed",
        clock=dict(strict=dict(clock_rate=0.1)),
        substitution=dict(
            discrete_trait=dict(
                n_states=n_states,
                frequencies=dict(
                    dirichlet=dict(concentration=[1.0] * n_states),
                ),
                rates=dict(
                    dirichlet=dict(concentration=[1.0] * n_rates),
                ),
            )
        ),
        site="none",
    )
    model = PhyloModel(model_dict)
    dist = phylo_model_to_joint_distribution(model, hello_tensor_tree, traits)
    sample = dist.sample()
    assert sample.frequencies.shape[-1] == n_states
    assert sample.rates.shape[-1] == n_rates
    lp = dist.log_prob(sample)
    assert np.isfinite(lp.numpy())
