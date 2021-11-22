from treeflow.tree.topology.tensorflow_tree_topology import TensorflowTreeTopology
import tensorflow as tf
import numpy as np
import tensorflow as tf
from numpy.testing import assert_allclose
import pytest
from treeflow import DEFAULT_FLOAT_DTYPE_TF
from treeflow.distributions.tree.coalescent.constant_coalescent import (
    ConstantCoalescent,
)
from treeflow.tree.rooted.tensorflow_rooted_tree import (
    TensorflowRootedTree,
    tree_from_arrays,
)

taxon_count = 3
parent_indices = [3, 3, 4, 4]
heights = [0.0, 0.0, 0.0, 1.0, 2.0]
pop_sizes = [100.0, 10000.0]
sampling_times = heights[:taxon_count]


@pytest.mark.parametrize("function_mode", [False, True])
@pytest.mark.parametrize(
    "pop_size,heights,parent_indices",
    [
        (np.array(pop_size), np.array(heights), np.array(parent_indices))
        for pop_size in pop_sizes
    ]
    + [
        (
            np.array(pop_sizes),
            np.array([heights, heights]),
            np.array([parent_indices, parent_indices]),
        )
    ],
)
def test_coalescent_homochronous(pop_size, heights, parent_indices, function_mode):
    tree = tree_from_arrays(heights=heights, parent_indices=parent_indices)
    dist = ConstantCoalescent(
        taxon_count,
        pop_size,
        tf.convert_to_tensor(sampling_times, dtype=DEFAULT_FLOAT_DTYPE_TF),
    )
    test_func = tf.function(dist.log_prob) if function_mode else dist.log_prob
    res = test_func(tree)
    expected = -(4 / pop_size) - 2 * np.log(pop_size)
    assert_allclose(res.numpy(), expected)


test_data = [(123.0, -14.446309163678226), (999.0, -20.721465537146862)]


@pytest.mark.parametrize("pop_size,expected", test_data)
def test_coalescent_heterochronous(pop_size, expected):
    pop_size = tf.convert_to_tensor(pop_size, dtype=DEFAULT_FLOAT_DTYPE_TF)
    sampling_times = tf.convert_to_tensor(
        [0.0, 0.1, 0.4, 0.0], dtype=DEFAULT_FLOAT_DTYPE_TF
    )
    heights = tf.concat([sampling_times, [0.2, 0.5, 0.8]], axis=0)
    parent_indices = tf.convert_to_tensor([4, 4, 5, 6, 5, 6])
    dist = ConstantCoalescent(4, pop_size, sampling_times)
    tree = tree_from_arrays(heights=heights, parent_indices=parent_indices)
    res = dist.log_prob(tree)
    assert_allclose(res.numpy(), expected)


def test_coalescent_event_shape_tensor():
    pop_size = 1.2
    pop_size_tensor = tf.convert_to_tensor(pop_size, dtype=DEFAULT_FLOAT_DTYPE_TF)
    sampling_times = tf.convert_to_tensor(
        [0.0, 0.1, 0.4, 0.0], dtype=DEFAULT_FLOAT_DTYPE_TF
    )
    taxon_count = tf.shape(sampling_times)[-1]
    coalescent = ConstantCoalescent(taxon_count, pop_size_tensor, sampling_times)
    event_shape = coalescent.event_shape_tensor()
    taxon_count_int = taxon_count.numpy()
    assert tuple(event_shape.heights.numpy()) == (2 * taxon_count_int - 1,)
    assert tuple(event_shape.topology.parent_indices.numpy()) == (
        2 * taxon_count_int - 2,
    )


@pytest.mark.parametrize(
    "pop_size", [2.0]
)  # TODO: Test and fix for vectorisation over pop size
def test_coalescent_sample(pop_size):
    pop_size_tensor = tf.convert_to_tensor(pop_size, dtype=DEFAULT_FLOAT_DTYPE_TF)
    batch_shape = pop_size_tensor.shape
    sampling_times = tf.convert_to_tensor(
        [0.0, 0.1, 0.4, 0.0], dtype=DEFAULT_FLOAT_DTYPE_TF
    )
    taxon_count = sampling_times.shape[-1]
    coalescent = ConstantCoalescent(taxon_count, pop_size_tensor, sampling_times)
    sample = coalescent.sample()
    assert isinstance(sample, TensorflowRootedTree)
    assert sample.heights.shape == (2 * taxon_count - 1,) + batch_shape
    assert isinstance(sample.topology, TensorflowTreeTopology)
    assert sample.topology.parent_indices.shape == (2 * taxon_count - 2,) + batch_shape
