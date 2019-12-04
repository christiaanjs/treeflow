import tensorflow as tf
import numpy as np
from numpy.testing import assert_allclose
import pytest

from treeflow.coalescent import ConstantCoalescent

def test_coalescent_homochronous():
    pop_size = tf.convert_to_tensor(np.array(10000.0))
    heights = tf.convert_to_tensor(np.array([0.0, 0.0, 0.0, 1.0, 2.0]))
    node_mask = tf.convert_to_tensor(np.array([False, False, False, True, True]))
    dist = ConstantCoalescent(pop_size, node_mask)
    res = dist.log_prob(heights)
    expected =  -(4 / pop_size) - 2 * np.log(pop_size)
    assert_allclose(res.numpy(), expected)

test_data = [(123.0,-14.446309163678226),(999.0,-20.721465537146862)]
@pytest.mark.parametrize('pop_size,expected', test_data)
def test_coalescent_heterochronous(pop_size, expected):
    pop_size = tf.convert_to_tensor(np.array(pop_size))
    heights = tf.convert_to_tensor(np.array([0.0, 0.1, 0.4, 0.0, 0.2, 0.5, 0.8]))
    node_mask = tf.convert_to_tensor(np.array([False, False, False, False, True, True, True]))
    dist = ConstantCoalescent(pop_size, node_mask)
    res = dist.log_prob(heights)
    assert_allclose(res.numpy(), expected)