import tensorflow as tf
import pytest
from treeflow import DEFAULT_FLOAT_DTYPE_TF
from tensorflow_probability.python.distributions import (
    JointDistributionNamed,
    LogNormal,
    Normal,
)
from treeflow.tree.rooted.tensorflow_rooted_tree import TensorflowRootedTree
from treeflow.distributions.tree.coalescent.constant_coalescent import (
    ConstantCoalescent,
)
from treeflow.model.conditioning import flatten_trees


@pytest.fixture
def tree_model(hello_tensor_tree: TensorflowRootedTree):
    taxon_count = hello_tensor_tree.taxon_count
    sampling_times = hello_tensor_tree.sampling_times
    return JointDistributionNamed(
        dict(
            pop_size=LogNormal(0.0, 1.0),
            tree=lambda pop_size: ConstantCoalescent(
                taxon_count=taxon_count,
                pop_size=pop_size,
                sampling_times=sampling_times,
            ),
            observed_times=lambda tree: Normal(tree.node_heights, 0.01),
        )
    )


def test_flatten_trees(tree_model: JointDistributionNamed):
    flat_model = flatten_trees(tree_model, ["tree"])
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
