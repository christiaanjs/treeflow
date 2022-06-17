import pytest
import io
import tensorflow as tf
import treeflow
from treeflow.model.io import (
    flatten_tensor_to_1d_slices,
    flatten_samples_to_dict,
    write_samples_to_file,
)
from itertools import product
from numpy.testing import assert_allclose
from tensorflow_probability.python.distributions import (
    JointDistributionNamed,
    Sample,
    Normal,
)
from treeflow.tree.rooted.tensorflow_rooted_tree import TensorflowRootedTree


def test_flatten_tensor_to_1d_slices():
    m = 2
    n = 3
    sample_size = 5
    x = tf.reshape(
        tf.range(m * n * sample_size, dtype=treeflow.DEFAULT_FLOAT_DTYPE_TF),
        (sample_size, m, n),
    )
    x_np = x.numpy()
    name = "var"
    indices = list(product(range(m), range(n)))
    key_func = lambda i, j: f"var_{i}_{j}"
    keys = [key_func(i, j) for i, j in indices]
    res = flatten_tensor_to_1d_slices(name, x)
    assert set(res.keys()) == set(keys)
    for (i, j), key in zip(indices, keys):
        arr = res[key].numpy()
        assert arr.shape == (sample_size,)
        assert_allclose(arr, x_np[:, i, j])


@pytest.fixture
def joint_dist(tensor_constant):
    x_event_shape = (2, 3)
    dist = JointDistributionNamed(
        dict(
            x=Sample(
                Normal(loc=tensor_constant(0.0), scale=tensor_constant(1.0)),
                sample_shape=x_event_shape,
            ),
            y=lambda x: Normal(
                loc=tf.reduce_sum(tf.reduce_sum(x, axis=-1), axis=-1),
                scale=tensor_constant(1.0),
            ),
        )
    )
    return dist


def test_samples_to_dict(joint_dist):
    sample_shape = (5,)

    samples = joint_dist.sample(sample_shape, seed=1)
    res, _ = flatten_samples_to_dict(samples, joint_dist)
    keys = set(
        [
            f"x_{i}_{j}"
            for i, j in product(*[range(k) for k in joint_dist.event_shape["x"]])
        ]
        + ["y"]
    )
    assert set(res.keys()) == keys
    for key in keys:
        assert res[key].numpy().shape == sample_shape


@pytest.mark.parametrize("vars", [None, ["y"]])
def test_write_samples_to_file(joint_dist, vars):
    sample_shape = (5,)

    samples = joint_dist.sample(sample_shape, seed=1)
    stringio = io.StringIO()
    write_samples_to_file(samples, joint_dist, stringio, vars=vars)
    res = stringio.getvalue()
    print(res)


def test_write_samples_to_file_with_tree(
    joint_dist, hello_tensor_tree: TensorflowRootedTree
):
    sample_size = 3
    sample_shape = (sample_size,)
    tiled_tree = hello_tensor_tree.with_node_heights(
        tf.stack([hello_tensor_tree.node_heights] * sample_size, axis=0)
    )

    samples = joint_dist.sample(sample_shape, seed=1)
    stringio = io.StringIO()
    write_samples_to_file(
        samples, joint_dist, stringio, tree_vars=dict(hello_tree=tiled_tree)
    )
    res = stringio.getvalue()
    print(res)
