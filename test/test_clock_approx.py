import pytest
import treeflow
import treeflow.clock_approx
import tensorflow as tf
import tensorflow_probability as tfp
import treeflow.tree_processing

lognormal_kwargs = dict(
    loc=tf.convert_to_tensor(0.0, dtype=treeflow.DEFAULT_FLOAT_DTYPE_TF),
    scale=tf.convert_to_tensor(1.0, dtype=treeflow.DEFAULT_FLOAT_DTYPE_TF)
)
lognormal_dist = tfp.distributions.LogNormal(**lognormal_kwargs)
clock_rate = tf.convert_to_tensor(0.1, dtype=treeflow.DEFAULT_FLOAT_DTYPE_TF)

@pytest.fixture()
def hello_tensor_tree_and_count(hello_newick_file):
    tree, taxon_names = treeflow.tree_processing.parse_newick(hello_newick_file)
    return treeflow.tree_processing.tree_to_tensor(tree), len(taxon_names)

@pytest.fixture()
def hello_tensor_tree_vec_and_count(hello_tensor_tree_and_count):
    tree, taxon_count = hello_tensor_tree_and_count
    sample_shape = 2
    return dict(
        topology=dict(parent_indices=tf.broadcast_to(tree['topology']['parent_indices'], [sample_shape, tree['topology']['parent_indices'].shape[0]])),
        heights=tf.broadcast_to(tree['heights'], [sample_shape, tree['heights'].shape[0]])
    ), taxon_count

def test_approx_sample_scalar(hello_tensor_tree_and_count):
    tree, _ = hello_tensor_tree_and_count
    dist = treeflow.clock_approx.ScaledDistribution(lognormal_dist, tree)
    sample = dist.sample()
    assert sample.numpy().shape == ()

def test_approx_sample_vector(hello_tensor_tree_vec_and_count):
    tree, taxon_count = hello_tensor_tree_vec_and_count
    base_dist = lambda **x: tfp.distributions.LogNormal(**x)
    dist = treeflow.clock_approx.ScaledDistribution(base_dist, tree, **lognormal_kwargs)
    sample = dist.sample()
    sample_shape = tf.shape(tree["heights"])[:-1]
    assert sample.numpy().shape == sample_shape

def test_rate_approx_sample_scalar(hello_tensor_tree_and_count):
    tree, taxon_count = hello_tensor_tree_and_count
    dist = treeflow.clock_approx.ScaledRateDistribution(tfp.distributions.Sample(lognormal_dist, 2*taxon_count - 2), tree, clock_rate)
    sample = dist.sample()
    assert sample.numpy().shape == (2*taxon_count - 2,)

def test_rate_approx_sample_vector(hello_tensor_tree_vec_and_count):
    tree, taxon_count = hello_tensor_tree_vec_and_count
    sample_shape = tf.shape(tree["heights"])[:-1]
    clock_vec = tf.broadcast_to(clock_rate, sample_shape)
    base_rate_dist = lambda **x: tfp.distributions.Sample(tfp.distributions.LogNormal(**x), 2*taxon_count - 2)
    rate_dist = treeflow.clock_approx.ScaledRateDistribution(base_rate_dist, tree, clock_vec, **lognormal_kwargs)
    rates = rate_dist.sample()
    assert rates.numpy().shape == (sample_shape, 2*taxon_count - 2,)