import typing as tp
from sympy import root
import tensorflow as tf
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.bijectors import Bijector
from treeflow.tree.topology.tensorflow_tree_topology import TensorflowTreeTopology
from treeflow.traversal.preorder import preorder_traversal


def _move_node_dimension_to_beginning(x):
    return tf.nest.map_structure(
        lambda x: distribution_util.move_dimension(x, -1, 0), x
    )


class PreorderNodeBijector(Bijector):
    def __init__(
        self,
        topology: TensorflowTreeTopology,
        input: object,
        bijector_func: tp.Callable[[object, object], Bijector],
        bijector_init: object,
        name=None,
        validate_args=False,
    ):
        """
        Parameters
        ----------
        input
            Tensor or structure of Tensors with node dimension first
        bijector_func
            Callable with signature (parent_output, input_element)
        bijector_init
            Passed to bijector_func for root node
        """
        parameters = locals()
        self._topology = topology
        self._input_node_first = _move_node_dimension_to_beginning(input)
        self._bijector_func = bijector_func
        self._bijector_init = bijector_init

        self._concrete_bijector = self._bijector_func(
            bijector_init,
            tf.nest.map_structure(lambda x: x[-1], self._input_node_first),
        )

        super().__init__(
            parameters=parameters,
            dtype=self._concrete_bijector.dtype,
            forward_min_event_ndims=tf.nest.map_structure(
                lambda x: x + 1, self._concrete_bijector.forward_min_event_ndims
            ),
            inverse_min_event_ndims=tf.nest.map_structure(
                lambda x: x + 1, self._concrete_bijector.inverse_min_event_ndims
            ),
            validate_args=validate_args,
            name=name,
        )

    def _forward_mapping(self, parent_output: object, input_and_x: object) -> object:
        bijector_input, x = input_and_x
        return self._bijector_func(parent_output, bijector_input).forward(x)

    def _forward(self, x):
        x_node_first = _move_node_dimension_to_beginning(x)
        res = preorder_traversal(
            self._topology,
            self._forward_mapping,
            (self._input_node_first, x_node_first),
            tf.nest.map_structure(lambda x: x[-1], x_node_first),
        )
        return tf.nest.map_structure(
            lambda x: distribution_util.move_dimension(x, 0, -1), res
        )

    def _inverse_bijectors_and_y_node_first(self, y) -> tp.Tuple[Bijector, object]:
        y_node_first = _move_node_dimension_to_beginning(y)
        parent_values = tf.nest.map_structure(
            lambda y_elem: tf.gather(
                y_elem,
                self._topology.parent_indices[self._topology.taxon_count :]
                - self._topology.taxon_count,
            ),
            y_node_first,
        )
        bijectors = self._bijector_func(parent_values, self._input_node_first)
        return bijectors, y_node_first

    def _inverse_log_det_jacobian(self, y):
        bijectors, y_node_first = self._inverse_bijectors_and_y_node_first(y)
        node_values = bijectors.inverse_log_det_jacobian(y_node_first[:-1])
        return tf.reduce_sum(node_values, 0)

    def _inverse(self, y):
        bijectors, y_node_first = self._inverse_bijectors_and_y_node_first(y)
        nonroot_values_node_first = bijectors.inverse(y_node_first[:-1])
        nonroot_values = tf.nest.map_structure(
            lambda x: distribution_util.move_dimension(x, 0, -1),
            nonroot_values_node_first,
        )
        return tf.nest.map_structure(
            lambda nonroot_elem, y_elem: tf.concat(
                [nonroot_elem, y_elem[..., -1:]], -1
            ),
            nonroot_values,
            y,
        )