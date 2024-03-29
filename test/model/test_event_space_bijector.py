import numpy as np
from numpy.testing import assert_allclose
import tensorflow as tf
import tensorflow.python.util.nest as tf_nest
from tensorflow_probability.python.distributions import (
    JointDistributionSequential,
    LogNormal,
    Normal,
    Dirichlet,
    Sample,
)
from tensorflow_probability.python.bijectors import Sigmoid
from treeflow import DEFAULT_FLOAT_DTYPE_TF
from treeflow.distributions.tree.coalescent.constant_coalescent import (
    ConstantCoalescent,
)
from treeflow.model.event_shape_bijector import (
    get_fixed_topology_event_shape_and_space_bijector,
)
from treeflow_test_helpers.ratio_helpers import (
    RatioTestData,
    tree_from_ratio_test_data,
)


def test_get_fixed_topology_event_shape_and_space_bijector(
    ratio_test_data: RatioTestData,
):
    sampling_times = tf.constant(
        ratio_test_data.sampling_times, dtype=DEFAULT_FLOAT_DTYPE_TF
    )
    taxon_count = sampling_times.shape[-1]
    dist = JointDistributionSequential(
        [
            LogNormal(
                tf.constant(0.0, DEFAULT_FLOAT_DTYPE_TF),
                tf.constant(1.0, DEFAULT_FLOAT_DTYPE_TF),
                name="pop_size",
            ),
            lambda pop_size: ConstantCoalescent(
                taxon_count,
                pop_size,
                sampling_times,
                name="test_tree",
                tree_name="test_tree",
            ),
            Sample(
                Normal(
                    tf.constant(2.0, DEFAULT_FLOAT_DTYPE_TF),
                    tf.constant(1.0, DEFAULT_FLOAT_DTYPE_TF),
                ),
                (3, 2),
                name="other_variable",
            ),
            Dirichlet(
                tf.constant([2.0, 2.0, 2.0, 2.0], DEFAULT_FLOAT_DTYPE_TF),
                name="frequencies",
            ),
        ]
    )
    tree = tree_from_ratio_test_data(ratio_test_data)
    bijector, base_event_shape = get_fixed_topology_event_shape_and_space_bijector(
        dist, dict(test_tree=tree.topology)
    )
    ratios = ratio_test_data.ratios
    unconstrained_ratios_tensor = tf.constant(
        np.concatenate(
            [Sigmoid().inverse(ratios[..., :-1]).numpy(), np.log(ratios[..., -1:])],
            axis=-1,
        ),
        dtype=DEFAULT_FLOAT_DTYPE_TF,
    )
    unconstrained_frequencies = tf.constant([1.0, -1.0, 0.5], DEFAULT_FLOAT_DTYPE_TF)
    frequencies = tf.math.softmax(
        tf.concat(
            [unconstrained_frequencies, tf.zeros((1,), dtype=DEFAULT_FLOAT_DTYPE_TF)],
            axis=0,
        )
    )
    log_pop_size = tf.constant(-0.5, DEFAULT_FLOAT_DTYPE_TF)
    other_variable = tf.random.normal((3, 2), seed=1)
    other_variable_flat = tf.reshape(other_variable, (6,))
    unconstrained = dict(
        test_tree=unconstrained_ratios_tensor,
        other_variable=other_variable_flat,
        pop_size=tf.expand_dims(log_pop_size, 0),
        frequencies=unconstrained_frequencies,
    )
    res = bijector.forward(unconstrained)
    expected = [tf.exp(log_pop_size), tree, other_variable, frequencies]
    tf_nest.map_structure(
        lambda res, expected: assert_allclose(res.numpy(), expected.numpy()),
        res,
        expected,
    )
