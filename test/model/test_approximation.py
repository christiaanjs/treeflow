import pytest
import typing as tp
import numpy as np
import tensorflow as tf
import tensorflow_probability.python.distributions as tfd
from treeflow import DEFAULT_FLOAT_DTYPE_TF
from treeflow.model.approximation import (
    get_mean_field_approximation,
    get_fixed_topology_mean_field_approximation,
)
from treeflow.distributions.tree.coalescent.constant_coalescent import (
    ConstantCoalescent,
)
from treeflow_test_helpers.tree_helpers import TreeTestData, data_to_tensor_tree

_constant = lambda x: tf.constant(x, dtype=DEFAULT_FLOAT_DTYPE_TF)


def test_get_mean_field_approximation():
    sample_size = 3
    model = tfd.JointDistributionNamed(
        dict(
            a=tfd.Normal(_constant(0.0), _constant(1.0)),
            b=lambda a: tfd.Sample(tfd.LogNormal(a, _constant(1.0)), sample_size),
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
    model_log_prob = pinned.unnormalized_log_prob(sample)
    approx_log_prob = approximation.log_prob(sample)
    assert np.isfinite(model_log_prob.numpy())
    assert np.isfinite(approx_log_prob.numpy())
