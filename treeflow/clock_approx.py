import tensorflow_probability as tfp
import treeflow.sequences

class ScaledRateDistribution(tfp.distributions.TransformedDistribution):
    def __init__(self, distance_distribution, tree, clock_rate):
        blens = treeflow.sequences.get_branch_lengths(tree)
        bij = tfp.bijectors.Scale(1.0 / (blens * clock_rate))
        super(ScaledRateDistribution, self).__init__(distribution=distance_distribution, bijector=bij)