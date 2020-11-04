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

def test_clock_approx_sample_scalar(hello_newick_file):
    tree, taxon_names = treeflow.tree_processing.parse_newick(hello_newick_file)
    tree = treeflow.tree_processing.tree_to_tensor(tree)
    taxon_count = len(taxon_names)
    rate_dist = treeflow.clock_approx.ScaledRateDistribution(tfp.distributions.Sample(lognormal_dist, 2*taxon_count - 2), tree, clock_rate)
    rates = rate_dist.sample()
    assert rates.numpy().shape == (2*taxon_count - 2,)

def test_clock_approx_sample_vector(hello_newick_file):
    tree, taxon_names = treeflow.tree_processing.parse_newick(hello_newick_file)
    sample_shape = 2
    tree = treeflow.tree_processing.tree_to_tensor(tree)
    tree = dict(
        topology=dict(parent_indices=tf.broadcast_to(tree['topology']['parent_indices'], [sample_shape, tree['topology']['parent_indices'].shape[0]])),
        heights=tf.broadcast_to(tree['heights'], [sample_shape, tree['heights'].shape[0]])
    )
    clock_vec = tf.broadcast_to(clock_rate, [sample_shape])
    taxon_count = len(taxon_names)
    base_rate_dist = lambda **x: tfp.distributions.Sample(tfp.distributions.LogNormal(**x), 2*taxon_count - 2)
    rate_dist = treeflow.clock_approx.ScaledRateDistribution(base_rate_dist, tree, clock_vec, **lognormal_kwargs)
    rates = rate_dist.sample()
    assert rates.numpy().shape == (sample_shape, 2*taxon_count - 2,)