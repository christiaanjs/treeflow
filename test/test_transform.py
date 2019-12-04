import tensorflow as tf
import numpy as np
import pytest
from numpy.testing import assert_allclose
from treeflow.tree_transform import BranchBreaking, TreeChain

# Heights, parent indices, preorder indices, transformed
test_data = [[tf.convert_to_tensor(np.array(x)) for x in test_case] for test_case in [
    ([0.6, 2.4], [1], [0], [0.25, 2.4]),
    ([1.2, 0.9, 1.8, 2.4], [3, 2, 3], [0, 2, 1], [0.5, 0.5, 0.75, 2.4])
]]

@pytest.mark.parametrize('heights,parent_indices,preorder_indices,transformed', test_data)
def test_branch_breaking_forward(heights, parent_indices, preorder_indices, transformed):
    transform = BranchBreaking(parent_indices, preorder_indices)
    res = transform.forward(transformed)
    assert_allclose(res.numpy(), heights.numpy())

@pytest.mark.parametrize('heights,parent_indices,preorder_indices,transformed', test_data)
def test_branch_breaking_inverse(heights, parent_indices, preorder_indices, transformed):
    transform = BranchBreaking(parent_indices, preorder_indices)
    res = transform.inverse(heights)
    assert_allclose(res.numpy(), transformed.numpy())

@pytest.mark.parametrize('heights,parent_indices,preorder_indices,transformed', test_data)
def test_branch_breaking_jac(heights, parent_indices, preorder_indices, transformed):
    transform = BranchBreaking(parent_indices, preorder_indices)
    with tf.GradientTape(persistent=True) as t:
        t.watch(heights)
        res = transform.inverse(heights)
    jac = t.jacobian(res, heights, experimental_use_pfor=False)
    log_det_jac = tf.math.log(tf.linalg.det(jac))
    assert_allclose(log_det_jac.numpy(), transform.inverse_log_det_jacobian(heights, event_ndims=1).numpy())

def logit(y):
    return tf.math.log(y) - tf.math.log1p(-y)

def unconstrain(transformed):
    return tf.concat([logit(transformed[:-1]), tf.math.log(transformed[-1:])], 0)

@pytest.mark.parametrize('heights,parent_indices,preorder_indices,transformed', test_data)
def test_tree_transform_forward(heights, parent_indices, preorder_indices, transformed):
    transform = TreeChain(parent_indices, preorder_indices)
    unconstrained = unconstrain(transformed)
    res = transform.forward(unconstrained)
    assert_allclose(res.numpy(), heights.numpy())

@pytest.mark.parametrize('heights,parent_indices,preorder_indices,transformed', test_data)
def test_tree_transform_inverse(heights, parent_indices, preorder_indices, transformed):
    transform = TreeChain(parent_indices, preorder_indices)
    unconstrained = unconstrain(transformed)
    res = transform.inverse(heights)
    assert_allclose(res.numpy(), unconstrained.numpy())

@pytest.mark.parametrize('heights,parent_indices,preorder_indices,transformed', test_data)
def test_tree_transform_jac(heights, parent_indices, preorder_indices, transformed):
    transform = TreeChain(parent_indices, preorder_indices)
    with tf.GradientTape(persistent=True) as t:
        t.watch(heights)
        res = transform.inverse(heights)
    jac = t.jacobian(res, heights, experimental_use_pfor=False)
    log_det_jac = tf.math.log(tf.linalg.det(jac))
    assert_allclose(log_det_jac.numpy(), transform.inverse_log_det_jacobian(heights, event_ndims=1).numpy())
