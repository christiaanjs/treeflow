import numpy as np
from treeflow.tree.rooted.tensorflow_rooted_tree import TensorflowRootedTree
from treeflow.model.approximation.cascading_flows import (
    get_cascading_flows_tree_approximation,
)


def test_get_cascading_flows_tree_approximation(
    hello_tensor_tree: TensorflowRootedTree,
):
    approx = get_cascading_flows_tree_approximation(hello_tensor_tree)
    sample_shape = (4,)
    sample = approx.sample(sample_shape)
    assert isinstance(sample, TensorflowRootedTree)
    log_prob = approx.log_prob(sample)
    assert log_prob.numpy().shape == sample_shape
    assert all(np.isfinite(log_prob.numpy()))
