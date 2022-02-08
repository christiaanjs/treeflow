from abc import abstractmethod
import typing as tp
import tensorflow as tf
from tensorflow.python.framework.ops import Tensor
from tensorflow_probability.python.distributions import Distribution
from tensorflow_probability.python.internal import prefer_static as ps
from treeflow import DEFAULT_FLOAT_DTYPE_TF
from treeflow.tree.base_tree import AbstractTree
from treeflow.tree.topology.tensorflow_tree_topology import TensorflowTreeTopology
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import nest_util

TTree = tp.TypeVar("TTree", bound=AbstractTree)


# TODO: Think carefully about the type hierarchy
class BaseTreeDistribution(Distribution, tp.Generic[TTree]):
    _topology_reparameterization_type = TensorflowTreeTopology(
        reparameterization.NOT_REPARAMETERIZED,
        reparameterization.NOT_REPARAMETERIZED,
        reparameterization.NOT_REPARAMETERIZED,
    )
    _topology_dtype = TensorflowTreeTopology(tf.int32, tf.int32, tf.int32)

    def __init__(
        self,
        taxon_count,
        reparameterization_type,
        dtype,
        validate_args=False,
        allow_nan_stats=True,
        tree_name: tp.Optional[str] = None,
        name="TreeDistribution",
        parameters=None,
    ):
        self.taxon_count = taxon_count
        self.tree_name = tree_name
        super().__init__(
            dtype=dtype,
            reparameterization_type=reparameterization_type,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name=name,
            parameters=parameters,
        )

    def _call_sample_n(self, sample_shape, seed, name: str, **kwargs) -> TTree:
        """Wrapper around _sample_n."""
        with self._name_and_control_scope(name):
            sample_shape = ps.convert_to_shape_tensor(
                ps.cast(sample_shape, tf.int32), name="sample_shape"
            )
            sample_shape, n = self._expand_sample_shape_to_vector(
                sample_shape, "sample_shape"
            )
            flat_samples = self._sample_n(
                n, seed=seed() if callable(seed) else seed, **kwargs
            )

            def reshape_samples(sample_element):
                batch_event_shape = ps.shape(sample_element)[1:]
                final_shape = ps.concat([sample_shape, batch_event_shape], 0)
                return tf.reshape(sample_element, final_shape)

            samples = tf.nest.map_structure(reshape_samples, flat_samples)
            samples = self._set_sample_static_shape(samples, sample_shape)
            return samples

    def _call_log_prob(self, value, name, **kwargs):
        """Wrapper around _log_prob."""
        value = tf.nest.pack_sequence_as(self.dtype, tf.nest.flatten(value))
        value = nest_util.convert_to_nested_tensor(
            value, name="value", dtype_hint=self.dtype, allow_packing=True
        )
        with self._name_and_control_scope(name, value, kwargs):
            if hasattr(self, "_log_prob"):
                return self._log_prob(value, **kwargs)
            if hasattr(self, "_prob"):
                return tf.math.log(self._prob(value, **kwargs))
            raise NotImplementedError(
                "log_prob is not implemented: {}".format(type(self).__name__)
            )

    def _call_unnormalized_log_prob(self, value, name, **kwargs):
        """Wrapper around _unnormalized_log_prob."""
        value = tf.nest.pack_sequence_as(self.dtype, tf.nest.flatten(value))
        value = nest_util.convert_to_nested_tensor(
            value, name="value", dtype_hint=self.dtype, allow_packing=True
        )
        with self._name_and_control_scope(name, value, kwargs):
            if hasattr(self, "_unnormalized_log_prob"):
                return self._unnormalized_log_prob(value, **kwargs)
            if hasattr(self, "_unnormalized_prob"):
                return tf.math.log(self._unnormalized_prob(value, **kwargs))
            if hasattr(self, "_log_prob"):
                return self._log_prob(value, **kwargs)
            if hasattr(self, "_prob"):
                return tf.math.log(self._prob(value, **kwargs))
            raise NotImplementedError(
                "unnormalized_log_prob is not implemented: {}".format(
                    type(self).__name__
                )
            )

    @abstractmethod
    def _event_shape(self) -> TTree:
        pass

    @abstractmethod
    def _event_shape_tensor(self) -> TTree:
        pass

    @staticmethod
    def _static_topology_event_shape(taxon_count) -> TensorflowTreeTopology:
        return TensorflowTreeTopology(
            parent_indices=tf.TensorShape([2 * taxon_count - 2]),
            child_indices=tf.TensorShape([2 * taxon_count - 1, 2]),
            preorder_indices=tf.TensorShape([2 * taxon_count - 1]),
        )

    def _topology_event_shape(self) -> TensorflowTreeTopology:
        taxon_count = self.taxon_count
        return type(self)._static_topology_event_shape(taxon_count)

    @staticmethod
    def _static_topology_event_shape_tensor(
        taxon_count,
    ) -> TensorflowTreeTopology:
        taxon_count_tensor = tf.reshape(tf.convert_to_tensor(taxon_count), [1])
        node_count = 2 * taxon_count_tensor - 1
        return TensorflowTreeTopology(
            parent_indices=2 * taxon_count_tensor - 2,
            child_indices=tf.concat([node_count, [2]], axis=0),
            preorder_indices=node_count,
        )

    def _topology_event_shape_tensor(self) -> TensorflowTreeTopology:
        taxon_count = self.taxon_count
        return type(self)._static_topology_event_shape_tensor(taxon_count)
