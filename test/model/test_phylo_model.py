import pytest
import numpy as np
from numpy.testing import assert_allclose
import tensorflow as tf
from treeflow import DEFAULT_FLOAT_DTYPE_TF
from treeflow.distributions.discretized import DiscretizedDistribution
from treeflow.evolution.seqio import Alignment
from treeflow.model.phylo_model import (
    phylo_model_to_joint_distribution,
    PhyloModel,
    get_site_rate_distribution,
    get_sequence_distribution,
)
from treeflow.tree.rooted.tensorflow_rooted_tree import TensorflowRootedTree
from treeflow.evolution.substitution.nucleotide.hky import HKY

model_dicts_and_keys = [
    (
        dict(
            tree=dict(
                birth_death=dict(
                    birth_diff_rate=dict(lognormal=dict(loc=0.05, scale=0.1)),
                    relative_death_rate=dict(lognormal=dict(loc=0.1, scale=0.2)),
                    sample_probability=dict(
                        beta=dict(concentration1=1.0, concentration0=5.0)
                    ),
                ),
            ),
            clock=dict(strict=dict(clock_rate=1e-3)),
            substitution=dict(
                gtr=dict(
                    frequencies=dict(
                        dirichlet=dict(
                            concentration=[4.0, 4.0, 4.0, 4.0],
                        ),
                    ),
                    gtr_rates=dict(
                        dirichlet=dict(
                            concentration=[2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                        )
                    ),
                )
            ),
            site="none",
        ),
        {
            "tree",
            "birth_diff_rate",
            "relative_death_rate",
            "sample_probability",
            "frequencies",
            "gtr_rates",
            "alignment",
        },
    ),
    (
        dict(
            tree=dict(
                coalescent=dict(pop_size=dict(lognormal=dict(loc=0.05, scale=0.1))),
            ),
            clock=dict(
                relaxed_lognormal=dict(
                    branch_rate_loc=dict(lognormal=dict(loc=0.1, scale=0.5)),
                    branch_rate_scale=dict(gamma=dict(concentration=2.0, rate=1.5)),
                )
            ),
            substitution=dict(
                hky=dict(
                    frequencies=dict(
                        dirichlet=dict(
                            concentration=[4.0, 4.0, 4.0, 4.0],
                        ),
                    ),
                    kappa=dict(lognormal=dict(loc=1, scale=0.8)),
                )
            ),
            site=dict(
                discrete_weibull=dict(
                    site_weibull_concentration=dict(
                        gamma=dict(concentration=2.0, rate=3.0)
                    ),
                    site_weibull_scale=1.0,
                    category_count=4,
                )
            ),
        ),
        {
            "tree",
            "pop_size",
            "branch_rates",
            "branch_rate_loc",
            "branch_rate_scale",
            "frequencies",
            "kappa",
            "site_weibull_concentration",
            "alignment",
        },
    ),
]


@pytest.fixture(params=model_dicts_and_keys)
def model_dict_and_keys(request):
    return request.param


@pytest.mark.parametrize("sample_shape", [(), (1,), (2, 3)])
def test_phylo_model_to_joint_distribution_sample(
    model_dict_and_keys, hello_tensor_tree, hello_alignment, sample_shape
):
    model_dict, expected_keys = model_dict_and_keys
    model = PhyloModel(model_dict)
    dist = phylo_model_to_joint_distribution(model, hello_tensor_tree, hello_alignment)
    sample = dist.sample(sample_shape)
    assert set(sample._asdict().keys()) == expected_keys


def test_get_site_rate_distribution():
    dist: DiscretizedDistribution = get_site_rate_distribution(
        "discrete_gamma",
        dict(
            site_gamma_shape=tf.constant(0.3, dtype=DEFAULT_FLOAT_DTYPE_TF),
            category_count=4,
        ),
    )
    category_rates_res = dist.normalised_support
    category_weights_res = dist.probabilities
    # Calculated using BEAST 2
    expected_category_rates = [
        0.002923662378691095,
        0.11617341856940992,
        0.7064988205902294,
        3.1744040984616695,
    ]
    expected_category_weights = [0.25, 0.25, 0.25, 0.25]
    assert_allclose(category_rates_res.numpy(), expected_category_rates)
    assert_allclose(category_weights_res.numpy(), expected_category_weights)


def test_phylo_model_to_joint_distribution_site_rate_variation(
    hello_tensor_tree: TensorflowRootedTree, hello_alignment: Alignment
):
    model_dict = dict(
        tree=dict(
            coalescent=dict(pop_size=1.0),
        ),
        clock=dict(strict=dict(clock_rate=1e-3)),
        substitution="jc",
        site=dict(
            discrete_gamma=dict(
                site_gamma_shape=0.3,
                category_count=4,
            )
        ),
    )
    model = PhyloModel(model_dict)
    dist = phylo_model_to_joint_distribution(model, hello_tensor_tree, hello_alignment)

    # Calculated using BEAST 2
    log_prior_expected = -0.5000000000000001
    log_prob_expected = -132.56406079329054
    log_prior_res, log_prob_res = dist.log_prob_parts(
        [
            hello_tensor_tree,
            hello_alignment.get_encoded_sequence_tensor(hello_tensor_tree.taxon_set),
        ]
    )
    assert_allclose(log_prior_res.numpy(), log_prior_expected)
    assert_allclose(log_prob_res.numpy(), log_prob_expected)


def test_phylo_model_get_sequence_distribution_vec(
    hello_alignment: Alignment, hello_tensor_tree: TensorflowRootedTree
):
    subst_model = HKY()
    encoded_sequences = hello_alignment.get_encoded_sequence_tensor(
        hello_tensor_tree.taxon_set
    )
    kappa = tf.constant([1.8, 2.3], dtype=DEFAULT_FLOAT_DTYPE_TF)
    frequencies = tf.constant(
        [[0.26, 0.22, 0.27, 0.25], [0.23, 0.25, 0.28, 0.24]],
        dtype=DEFAULT_FLOAT_DTYPE_TF,
    )
    node_heights = tf.stack(
        [hello_tensor_tree.node_heights, hello_tensor_tree.node_heights + 0.02]
    )
    site_gamma_shape = tf.constant([1.1, 1.3], dtype=DEFAULT_FLOAT_DTYPE_TF)
    clock_rate = tf.constant(1.0, dtype=DEFAULT_FLOAT_DTYPE_TF)
    batch_dist = get_sequence_distribution(
        hello_alignment,
        hello_tensor_tree.with_node_heights(node_heights),
        subst_model,
        dict(kappa=kappa, frequencies=frequencies),
        "discrete_gamma",
        dict(site_gamma_shape=site_gamma_shape, category_count=4),
        clock_rate,
    )
    batch_log_probs = batch_dist.log_prob(encoded_sequences).numpy()

    batch_size = kappa.shape[0]
    log_probs = np.zeros(batch_size)
    for i in range(batch_size):
        dist = get_sequence_distribution(
            hello_alignment,
            hello_tensor_tree.with_node_heights(node_heights[i]),
            subst_model,
            dict(kappa=kappa[i], frequencies=frequencies[i]),
            "discrete_gamma",
            dict(site_gamma_shape=site_gamma_shape[i], category_count=4),
            clock_rate,
        )
        log_probs[i] = dist.log_prob(encoded_sequences).numpy()
    assert_allclose(batch_log_probs, log_probs)


def test_phylo_model_get_sequence_distribution_vec_singleton(
    hello_alignment: Alignment, hello_tensor_tree: TensorflowRootedTree
):
    subst_model = HKY()
    encoded_sequences = hello_alignment.get_encoded_sequence_tensor(
        hello_tensor_tree.taxon_set
    )
    kappa = tf.constant([1.8], dtype=DEFAULT_FLOAT_DTYPE_TF)
    frequencies = tf.constant(
        [[0.26, 0.22, 0.27, 0.25]],
        dtype=DEFAULT_FLOAT_DTYPE_TF,
    )
    node_heights = tf.expand_dims(hello_tensor_tree.node_heights, axis=0)
    site_gamma_shape = tf.constant([1.3], dtype=DEFAULT_FLOAT_DTYPE_TF)
    clock_rate = tf.constant(1.0, dtype=DEFAULT_FLOAT_DTYPE_TF)
    batch_dist = get_sequence_distribution(
        hello_alignment,
        hello_tensor_tree.with_node_heights(node_heights),
        subst_model,
        dict(kappa=kappa, frequencies=frequencies),
        "discrete_gamma",
        dict(site_gamma_shape=site_gamma_shape, category_count=4),
        clock_rate,
    )
    batch_log_probs = batch_dist.log_prob(encoded_sequences).numpy()
    assert tuple(batch_log_probs.shape) == (1,)
    dist = get_sequence_distribution(
        hello_alignment,
        hello_tensor_tree.with_node_heights(node_heights[0]),
        subst_model,
        dict(kappa=kappa[0], frequencies=frequencies[0]),
        "discrete_gamma",
        dict(site_gamma_shape=site_gamma_shape[0], category_count=4),
        clock_rate,
    )
    log_prob = dist.log_prob(encoded_sequences).numpy()
    assert_allclose(batch_log_probs[0], log_prob)
