import numpy as np
import pytest
import tensorflow as tf
from tensorflow_probability.python.bijectors import (
    Sigmoid,
    Identity,
)
from treeflow.tree.rooted.tensorflow_rooted_tree import TensorflowRootedTree
from treeflow.model.approximation.cascading_flows import (
    get_cascading_flows_tree_approximation,
)


@pytest.mark.parametrize("function_mode", [True, False])
def test_get_cascading_flows_tree_approximation(
    hello_tensor_tree: TensorflowRootedTree, function_mode: bool
):
    activation_functions = [Sigmoid(), Sigmoid(), Identity()]
    approx = get_cascading_flows_tree_approximation(
        hello_tensor_tree, activation_functions=activation_functions
    )
    trainable_variables = approx.trainable_variables
    sample_shape = (4,)

    def test_func():
        with tf.GradientTape() as t:
            for variable in trainable_variables:
                t.watch(variable)
            sample = approx.sample(sample_shape)
            log_prob = approx.log_prob(sample)
        grads = t.gradient(log_prob, trainable_variables)
        return sample, log_prob, grads

    if function_mode:
        test_func = tf.function(test_func)

    sample, log_prob, grads = test_func()
    assert isinstance(sample, TensorflowRootedTree)
    assert log_prob.numpy().shape == sample_shape
    assert all(np.isfinite(log_prob.numpy()))
    for grad, variable in zip(grads, trainable_variables):
        assert grad is not None, f"Must have gradient wrt {variable.name}"
