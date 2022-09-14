import pytest
import typing as tp
import numpy as np
from numpy.testing import assert_allclose
import tensorflow as tf
import tensorflow_probability.python.distributions as tfd
from treeflow import DEFAULT_FLOAT_DTYPE_TF
from treeflow.tree.rooted.tensorflow_rooted_tree import convert_tree_to_tensor
from treeflow.tree.io import parse_newick
from treeflow.model.approximation import (
    get_mean_field_approximation,
    get_fixed_topology_mean_field_approximation,
)
from treeflow.distributions.tree.coalescent.constant_coalescent import (
    ConstantCoalescent,
)
from treeflow.distributions.tree.birthdeath.yule import Yule
from treeflow_test_helpers.tree_helpers import TreeTestData, data_to_tensor_tree
from treeflow.model.phylo_model import (
    PhyloModel,
    get_sequence_distribution,
    phylo_model_to_joint_distribution,
)

_constant = lambda x: tf.constant(x, dtype=DEFAULT_FLOAT_DTYPE_TF)


def test_get_mean_field_approximation():
    sample_size = 3
    model = tfd.JointDistributionNamed(
        dict(
            a=tfd.Normal(_constant(0.0), _constant(1.0)),
            b=lambda a: tfd.Dirichlet(tf.fill((sample_size,), _constant(2.0))),
            obs=lambda b: tfd.Independent(
                tfd.Normal(b, _constant(1.0)), reinterpreted_batch_ndims=1
            ),
        )
    )
    obs = _constant([-1.1, 2.1, 0.1])
    pinned = model.experimental_pin(obs=obs)
    approximation = get_mean_field_approximation(
        pinned, init_loc=dict(a=_constant(0.1)), dtype=DEFAULT_FLOAT_DTYPE_TF
    )
    sample = approximation.sample()
    model_log_prob = pinned.unnormalized_log_prob(sample)
    approx_log_prob = approximation.log_prob(sample)
    assert np.isfinite(model_log_prob.numpy())
    assert np.isfinite(approx_log_prob.numpy())


@pytest.mark.parametrize("with_init_loc", [True, False])
def test_get_mean_field_approximation_tree(
    flat_tree_test_data: TreeTestData, with_init_loc: bool
):
    test_tree = data_to_tensor_tree(flat_tree_test_data)
    taxon_count = test_tree.taxon_count
    tree_name = "tree_dist_name"

    init_loc: tp.Optional[tp.Dict[str, object]]
    if with_init_loc:
        init_loc = dict(tree=test_tree)
    else:
        init_loc = None

    model = tfd.JointDistributionNamed(
        dict(
            pop_size=tfd.LogNormal(_constant(0.0), _constant(1.0)),
            tree=lambda pop_size: ConstantCoalescent(
                taxon_count, pop_size, test_tree.sampling_times, tree_name=tree_name
            ),
            obs=lambda tree: tfd.Normal(
                _constant(0.0), tf.reduce_sum(tree.branch_lengths)
            ),
        )
    )
    obs = _constant([10.0])
    pinned = model.experimental_pin(obs=obs)
    approximation = get_fixed_topology_mean_field_approximation(
        pinned,
        dtype=DEFAULT_FLOAT_DTYPE_TF,
        topology_pins={tree_name: test_tree.topology},
        init_loc=init_loc,
    )

    sample = approximation.sample()
    assert (
        tf.reduce_all(
            sample["tree"].topology.parent_indices == test_tree.topology.parent_indices
        )
        .numpy()
        .item()
    )
    assert_allclose(
        sample["tree"].sampling_times.numpy(), test_tree.sampling_times.numpy()
    )
    model_log_prob = pinned.unnormalized_log_prob(sample)
    approx_log_prob = approximation.log_prob(sample)
    assert np.isfinite(model_log_prob.numpy())
    assert np.isfinite(approx_log_prob.numpy())


@pytest.mark.parametrize("with_init_loc", [True, False])
def test_get_mean_field_approximation_tree_yule(
    tensor_constant, hello_newick_file, with_init_loc
):
    tree = convert_tree_to_tensor(parse_newick(hello_newick_file))
    tree_name = "tree_dist_name"
    model = tfd.JointDistributionNamed(
        {
            "rates": tfd.Sample(
                tfd.LogNormal(_constant(0.0), _constant(1.0)), tree.branch_lengths.shape
            ),
            "birth_rate": tfd.LogNormal(_constant(1.0), _constant(1.5)),
            tree_name: lambda birth_rate: Yule(
                tree.taxon_count, birth_rate, name=tree_name
            ),
            "a": lambda tree_dist_name, rates: tfd.Normal(
                tf.reduce_sum(tree_dist_name.branch_lengths * rates, axis=-1),
                _constant(1.0),
            ),
        }
    )

    if with_init_loc:
        init_loc = dict(tree=tree, birth_rate=_constant(2.0))
    else:
        init_loc = None

    approximation = get_fixed_topology_mean_field_approximation(
        model,
        dtype=DEFAULT_FLOAT_DTYPE_TF,
        topology_pins={tree_name: tree.topology},
        init_loc=init_loc,
    )
    sample = approximation.sample()
    model_log_prob = model.log_prob(sample)
    approx_log_prob = approximation.log_prob(sample)
    assert np.isfinite(model_log_prob.numpy())
    assert np.isfinite(approx_log_prob.numpy())


def test_mean_field_approximation_batch_log_prob(hello_tensor_tree, hello_alignment):
    model_dict = dict(
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
    )
    model = phylo_model_to_joint_distribution(
        PhyloModel(model_dict), hello_tensor_tree, hello_alignment
    )
