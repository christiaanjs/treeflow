import tensorflow as tf
from treeflow.distributions.tree.rooted_tree_distribution import RootedTreeDistribution
from treeflow.tree.rooted.tensorflow_rooted_tree import TensorflowRootedTree
from tensorflow_probability.python.internal import reparameterization
import pytest


class DumbRootedTreeDistribution(RootedTreeDistribution):
    def __init__(self, taxon_count, name="DumbRootedTreeDistribution"):
        super().__init__(taxon_count, reparameterization.NOT_REPARAMETERIZED, name=name)

    def _parameter_properties(self, num_classes=None):
        return dict()

    def _sample_n(
        self,
        n,
        seed=None,
    ):
        event_shape = self.event_shape
        dtype = self.dtype

        shape_func = lambda event_shape: tf.concat([[n], event_shape], axis=0)

        return tf.nest.map_structure(
            lambda event_shape, dtype: tf.zeros(shape_func(event_shape), dtype),
            event_shape,
            dtype,
        )


@pytest.mark.parametrize("sample_shape", [(), 1, 3, (3,), (3, 2)])
def test_rooted_tree_distribution_sample(sample_shape):
    taxon_count = 4
    distribution_instance = DumbRootedTreeDistribution(taxon_count)
    samples = distribution_instance.sample(sample_shape)
