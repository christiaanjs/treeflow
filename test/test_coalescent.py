import tensorflow as tf
import numpy as np
import tensorflow as tf
from numpy.testing import assert_allclose
import pytest
from treeflow import DEFAULT_FLOAT_DTYPE_TF
from treeflow.coalescent import ConstantCoalescent

taxon_count = 3
parent_indices = [3, 3, 4, 4]
heights = [0.0, 0.0, 0.0, 1.0, 2.0]
pop_sizes = [100.0, 10000.0]
sampling_times = heights[:taxon_count]


@pytest.mark.parametrize(
    "pop_size,heights,parent_indices",
    [(pop_size, heights, parent_indices) for pop_size in pop_sizes]
    + [(pop_sizes, [heights, heights], [parent_indices, parent_indices])],
)
def test_coalescent_homochronous(pop_size, heights, parent_indices, function_mode):
    pop_size = tf.convert_to_tensor(pop_size, dtype=DEFAULT_FLOAT_DTYPE_TF)
    heights = tf.convert_to_tensor(heights, dtype=DEFAULT_FLOAT_DTYPE_TF)
    parent_indices = tf.convert_to_tensor(parent_indices)
    dist = ConstantCoalescent(
        taxon_count,
        pop_size,
        tf.convert_to_tensor(sampling_times, dtype=DEFAULT_FLOAT_DTYPE_TF),
    )
    test_func = tf.function(dist.log_prob) if function_mode else dist.log_prob
    res = test_func(dict(heights=heights, topology=dict(parent_indices=parent_indices)))
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
    res = dist.log_prob(
        dict(heights=heights, topology=dict(parent_indices=parent_indices))
    )
    assert_allclose(res.numpy(), expected)


def test_coalescent_event_shape():
    pop_size = 1.2
    pop_size_tensor = tf.convert_to_tensor(pop_size, dtype=DEFAULT_FLOAT_DTYPE_TF)
    sampling_times = tf.convert_to_tensor(
        [0.0, 0.1, 0.4, 0.0], dtype=DEFAULT_FLOAT_DTYPE_TF
    )
    taxon_count = tf.shape(sampling_times)[-1]
    coalescent = ConstantCoalescent(taxon_count, pop_size_tensor, sampling_times)
    event_shape = coalescent.event_shape


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
    assert isinstance(sample, dict)
    assert sample["heights"].shape == (2 * taxon_count - 1,) + batch_shape
    assert isinstance(sample["topology"], dict)
    assert (
        sample["topology"]["parent_indices"].shape
        == (2 * taxon_count - 2,) + batch_shape
    )
