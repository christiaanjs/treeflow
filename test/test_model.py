import treeflow
import treeflow.model
import treeflow.sequences
import tensorflow as tf
import tensorflow_probability as tfp
import pytest

@pytest.mark.parametrize('rate_approx', ['scaled', 'mean_field'])
def test_rate_approx_joint_sample_vector(hello_newick_file, rate_approx):
    tree_approx, _ = treeflow.model.construct_tree_approximation(hello_newick_file)
    blens = treeflow.sequences.get_branch_lengths(tree_approx.sample())
    rate_dist = tfp.distributions.LogNormal(tf.zeros_like(blens), tf.ones_like(blens))
    approx = tfp.distributions.JointDistributionNamed(dict(
        clock_rate=tfp.distributions.LogNormal(tf.convert_to_tensor(0.0, dtype=treeflow.DEFAULT_FLOAT_DTYPE_TF), tf.convert_to_tensor(1.0, dtype=treeflow.DEFAULT_FLOAT_DTYPE_TF)),
        tree=tree_approx,
        rates=treeflow.model.construct_rate_approximation(rate_dist, approx_model=rate_approx)[0]
    ))
    sample_shape = 2
    samples = approx.sample(sample_shape)
    assert samples['rates'].numpy().shape == (sample_shape, blens.numpy().shape[0])