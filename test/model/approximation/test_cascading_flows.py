import numpy as np
import pytest
import tensorflow as tf
from treeflow.tree.rooted.tensorflow_rooted_tree import TensorflowRootedTree
from treeflow.model.approximation.cascading_flows import (
    get_cascading_flows_tree_approximation,
)


@pytest.mark.parametrize("function_mode", [True, False])
def test_get_cascading_flows_tree_approximation(
    hello_tensor_tree: TensorflowRootedTree, function_mode: bool
):
    approx = get_cascading_flows_tree_approximation(hello_tensor_tree)
    variable = {x.name: x for x in approx.trainable_variables}[
        "tree_U_diag_inv_softplus:0"
    ]
    sample_shape = (4,)

    def test_func():
        with tf.GradientTape() as t:
            t.watch(variable)
            sample = approx.sample(sample_shape)
            log_prob = approx.log_prob(sample)
        grad = t.gradient(log_prob, variable)
        return sample, log_prob, grad

    if function_mode:
        test_func = tf.function(test_func)

    sample, log_prob, grad = test_func()
    assert isinstance(sample, TensorflowRootedTree)
    assert log_prob.numpy().shape == sample_shape
    assert all(np.isfinite(log_prob.numpy()))
    assert grad is not None
