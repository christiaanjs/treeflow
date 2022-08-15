import pytest
import numpy as np
import tensorflow as tf
import tensorflow_probability.python.distributions as tfd
from treeflow import DEFAULT_FLOAT_DTYPE_TF
from treeflow.tree.rooted.tensorflow_rooted_tree import convert_tree_to_tensor
from treeflow.tree.io import parse_newick
from treeflow.distributions.tree.birthdeath.yule import Yule
from treeflow.model.ml import fit_fixed_topology_maximum_likelihood_sgd


_constant = lambda x: tf.constant(x, dtype=DEFAULT_FLOAT_DTYPE_TF)


@pytest.mark.parametrize("with_init", [True, False])
def test_ml_tree_yule(tensor_constant, hello_newick_file, with_init):
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
    obs = _constant(10.0)
    pinned = model.experimental_pin(a=obs)

    if with_init:
        init = dict(tree=tree, birth_rate=_constant(2.0))
    else:
        init = None

    res, trace, bijector = fit_fixed_topology_maximum_likelihood_sgd(
        pinned, topologies={tree_name: tree.topology}, num_steps=30, init=init
    )
    assert all(np.isfinite(trace.log_likelihood.numpy()))
    tf.nest.assert_same_structure(res, pinned.sample_unpinned())
