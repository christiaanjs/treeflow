import typing as tp
from treeflow import DEFAULT_FLOAT_DTYPE_TF
from tensorflow_probability.python.distributions.distribution import (
    _set_sample_static_shape_for_tensor,
)
from treeflow.distributions.tree.base_tree_distribution import BaseTreeDistribution
from treeflow.tree.rooted.tensorflow_rooted_tree import TensorflowRootedTree
import tensorflow as tf
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import reparameterization
import functools


class RootedTreeDistribution(BaseTreeDistribution[TensorflowRootedTree]):
    def __init__(
        self,
        taxon_count,
        node_height_reparameterization_type: reparameterization.ReparameterizationType,
        sampling_time_reparameterization_type: reparameterization.ReparameterizationType,
        time_dtype=DEFAULT_FLOAT_DTYPE_TF,
        validate_args=False,
        allow_nan_stats=True,
        tree_name: tp.Optional[str] = None,
        name="RootedTreeDistribution",
        support_topology_batch_dims=False,
        parameters=None,
    ):
        super().__init__(
            taxon_count,
            reparameterization_type=TensorflowRootedTree(
                node_heights=node_height_reparameterization_type,
                sampling_times=sampling_time_reparameterization_type,
                topology=BaseTreeDistribution._topology_reparameterization_type,
            ),
            dtype=TensorflowRootedTree(
                node_heights=time_dtype,
                sampling_times=time_dtype,
                topology=BaseTreeDistribution._topology_dtype,
            ),
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name=name,
            tree_name=tree_name,
            parameters=parameters,
            support_topology_batch_dims=support_topology_batch_dims,
        )

    def _event_shape(self) -> TensorflowRootedTree:
        taxon_count = self.taxon_count
        return TensorflowRootedTree(
            node_heights=tf.TensorShape([taxon_count - 1]),
            sampling_times=tf.TensorShape([taxon_count]),
            topology=self._topology_event_shape(),
        )

    def _event_shape_tensor(self) -> TensorflowRootedTree:
        taxon_count = tf.expand_dims(tf.convert_to_tensor(self.taxon_count), 0)
        return TensorflowRootedTree(
            node_heights=taxon_count - 1,
            sampling_times=taxon_count,
            topology=self._topology_event_shape_tensor(),
        )

    def _process_sample_shape(self, sample_shape, name="sample_shape"):
        sample_shape = ps.convert_to_shape_tensor(
            ps.cast(sample_shape, tf.int32), name=name
        )
        sample_shape, n = self._expand_sample_shape_to_vector(sample_shape, name)
        return sample_shape, n

    def _call_sample_n(self, sample_shape, seed, **kwargs) -> TensorflowRootedTree:
        """Wrapper around _sample_n."""
        sample_shape, n = self._process_sample_shape(sample_shape)
        flat_samples = self._sample_n(
            n, seed=seed() if callable(seed) else seed, **kwargs
        )
        batch_shape = self.batch_shape
        event_shape = self.event_shape

        def reshape_samples(sample_element):
            batch_event_shape = ps.shape(sample_element)[1:]
            final_shape = ps.concat([sample_shape, batch_event_shape], 0)
            return tf.reshape(sample_element, final_shape)

        if self.support_topology_batch_dims:
            topology_samples = tf.nest.map_structure(
                reshape_samples, flat_samples.topology
            )
            topology_sample_shape = sample_shape
            topology_batch_shape = batch_shape

        else:
            topology_samples = flat_samples.topology
            topology_sample_shape = ()
            topology_batch_shape = ()
        topology_samples = tf.nest.map_structure(
            functools.partial(
                _set_sample_static_shape_for_tensor,
                batch_shape=topology_batch_shape,
                sample_shape=topology_sample_shape,
            ),
            topology_samples,
            event_shape.topology,
        )
        samples = TensorflowRootedTree(
            node_heights=_set_sample_static_shape_for_tensor(
                reshape_samples(flat_samples.node_heights),
                event_shape.node_heights,
                batch_shape,
                sample_shape,
            ),
            sampling_times=_set_sample_static_shape_for_tensor(
                reshape_samples(flat_samples.sampling_times),
                event_shape.sampling_times,
                batch_shape,
                sample_shape,
            ),
            topology=topology_samples,
        )
        return samples

    def _make_dummy_samples(self, sampling_times, n):
        dtype = self.dtype
        event_shape = self.event_shape_tensor()
        batch_shape = self.batch_shape_tensor()

        shape_func = lambda event_shape, batch_shape: tf.concat(
            [[n], batch_shape, event_shape], axis=0
        )

        sampling_times_b = tf.broadcast_to(
            sampling_times, shape_func(event_shape.sampling_times, batch_shape)
        )
        node_heights = tf.zeros(
            shape_func(event_shape.node_heights, batch_shape), sampling_times_b.dtype
        )
        if self.support_topology_batch_dims:
            raise NotImplemented(
                "Dummy topology sampling with batch dims not implemented"
            )
        else:
            topology = tf.nest.map_structure(  # No topology batch dims
                lambda event_shape, dtype: tf.zeros(event_shape, dtype),
                event_shape.topology,
                dtype.topology,
            )

        return TensorflowRootedTree(
            sampling_times=sampling_times_b,
            node_heights=node_heights,
            topology=topology,
        )


__all__ = [RootedTreeDistribution.__name__]
