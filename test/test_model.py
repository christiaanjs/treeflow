import treeflow
import treeflow.model
import treeflow.sequences
import tensorflow as tf
import tensorflow_probability as tfp
import pytest

lognormal_dist = tfp.distributions.LogNormal(
    tf.convert_to_tensor(0.0, dtype=treeflow.DEFAULT_FLOAT_DTYPE_TF),
    tf.convert_to_tensor(1.0, dtype=treeflow.DEFAULT_FLOAT_DTYPE_TF),
)


class MockJointDistribution:
    def __init__(self, keys):
        self.model = {x: 0 for x in keys}


@pytest.mark.parametrize("prior_keys", [("clock_rate", "tree"), ("tree",)])
@pytest.mark.parametrize("rate_approx", ["scaled", "mean_field"])
def test_rate_approx_joint_sample_vector(hello_newick_file, rate_approx, prior_keys):
    mock_prior = MockJointDistribution(prior_keys)
    tree_approx, _ = treeflow.model.construct_tree_approximation(hello_newick_file)
    blens = treeflow.sequences.get_branch_lengths(tree_approx.sample())
    rate_dist = tfp.distributions.LogNormal(tf.zeros_like(blens), tf.ones_like(blens))
    approx = tfp.distributions.JointDistributionNamed(
        dict(
            clock_rate=lognormal_dist,
            tree=tree_approx,
            rates=treeflow.model.construct_rate_approximation(
                rate_dist, mock_prior, approx_model=rate_approx
            )[0],
        )
    )
    sample_shape = 2
    samples = approx.sample(sample_shape)
    assert samples["rates"].numpy().shape == (sample_shape, blens.numpy().shape[0])


@pytest.mark.parametrize("clock_rate_approx", ["scaled", "mean_field"])
@pytest.mark.parametrize("tree_statistic", ["length", "height"])
def test_clock_approx_joint_sample_vector(
    hello_newick_file, clock_rate_approx, tree_statistic
):
    tree_approx, _ = treeflow.model.construct_tree_approximation(hello_newick_file)
    tree = tree_approx.sample()
    approx_kwargs = (
        dict(tree=tree, tree_statistic=tree_statistic)
        if clock_rate_approx == "scaled"
        else {}
    )
    approx = tfp.distributions.JointDistributionNamed(
        dict(
            tree=tree_approx,
            clock_rate=treeflow.model.construct_distribution_approximation(
                "q",
                "clock_rate",
                lognormal_dist,
                approx=clock_rate_approx,
                **approx_kwargs
            )[0],
        )
    )
    sample_shape = 2
    samples = approx.sample(sample_shape)
    assert samples["clock_rate"].numpy().shape == (sample_shape,)


def test_construct_prior_approx(hello_newick_file):
    prior = tfp.distributions.JointDistributionNamed(
        dict(clock_rate=lognormal_dist, pop_size=lognormal_dist)
    )
    prior_sample = prior.sample()
    approx = tfp.distributions.JointDistributionNamed(
        treeflow.model.construct_prior_approximation(prior, prior_sample)[0]
    )
    sample = approx.sample()
    for key in prior.sample():
        assert key in sample
