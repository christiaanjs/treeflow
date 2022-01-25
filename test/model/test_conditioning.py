import tensorflow as tf
from treeflow import DEFAULT_FLOAT_DTYPE_TF
from tensorflow_probability.python.distributions import (
    JointDistributionNamed,
    LogNormal,
    Normal,
)
from treeflow.distributions.tree.coalescent.constant_coalescent import (
    ConstantCoalescent,
)
from treeflow.model.conditioning import flatten_trees


def test_flatten_trees():
    taxon_count = 3
    sampling_times = tf.zeros(taxon_count, dtype=DEFAULT_FLOAT_DTYPE_TF)
    model = JointDistributionNamed(
        dict(
            pop_size=LogNormal(0.0, 1.0),
            tree=lambda pop_size: ConstantCoalescent(
                taxon_count=3, pop_size=pop_size, sampling_times=sampling_times
            ),
            observed_times=lambda tree: Normal(tree.node_heights, 0.01),
        )
    )
    flat_model = flatten_trees(model, ["tree"])
    sample = flat_model.sample()
    expected_keys = set(
        [
            "pop_size",
            "tree_node_heights",
            "tree_sampling_times",
            "tree_topology",
            "observed_times",
        ]
    )
    assert set(sample.keys()) == expected_keys
