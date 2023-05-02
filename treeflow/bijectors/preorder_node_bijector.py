import typing as tp
import tensorflow as tf
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.bijectors import Bijector
from treeflow.tree.topology.tensorflow_tree_topology import TensorflowTreeTopology
from treeflow.traversal.preorder import preorder_traversal


def _move_node_dimension_to_beginning(x, event_ndims):
    return tf.nest.map_structure(
        lambda x_elem, event_ndims_elem: distribution_util.move_dimension(
            x_elem, -(event_ndims_elem + 1), 0
        ),
        x,
        event_ndims,
    )


class PreorderNodeBijector(Bijector):
    def __init__(
        self,
        topology: TensorflowTreeTopology,
        input: object,
        bijector_func: tp.Callable[[object, object], Bijector],
        root_bijector_func: tp.Callable[[object], Bijector],
        input_event_ndims: object = 0,
        forward_event_ndims: object = 0,
        inverse_event_ndims: object = 0,
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
        root_bijector_func
            Callable with signature (input_element)
        """
        parameters = locals()
        self._topology = topology
        self._bijector_func = bijector_func
        self._root_bijector_func = root_bijector_func

        self._input_event_ndims = input_event_ndims
        self._forward_event_ndims = forward_event_ndims
        self._inverse_event_ndims = inverse_event_ndims

        self._input = input

        root_bijector = self._root_bijector
        super().__init__(
            parameters=parameters,
            dtype=self._root_bijector.dtype,
            forward_min_event_ndims=tf.nest.map_structure(
                lambda x: x + 1, root_bijector.forward_min_event_ndims
            ),
            inverse_min_event_ndims=tf.nest.map_structure(
                lambda x: x + 1, root_bijector.inverse_min_event_ndims
            ),
            validate_args=validate_args,
            name=name,
        )

    @property
    def _input_node_first(self):
        return _move_node_dimension_to_beginning(self._input, self._input_event_ndims)

    @property
    def _root_bijector(self):
        return self._root_bijector_func(
            tf.nest.map_structure(lambda x: x[-1], self._input_node_first),
        )

    def _forward_mapping(
        self, parent_output: object, input_and_x: tp.Tuple[object, object]
    ) -> object:
        bijector_input, x = input_and_x
        return self._bijector_func(parent_output, bijector_input).forward(x)

    def _forward(self, x):
        x_node_first = _move_node_dimension_to_beginning(x, self._forward_event_ndims)
        res_node_first = preorder_traversal(
            self._topology,
            self._forward_mapping,
            (self._input_node_first, x_node_first),
            self._root_bijector.forward(
                tf.nest.map_structure(lambda x: x[-1], x_node_first)
            ),
        )
        res = tf.nest.map_structure(
            lambda x: distribution_util.move_dimension(
                x, 0, -(1 + self._forward_event_ndims)
            ),
            res_node_first,
        )
        return res

    def _inverse_bijectors_and_y_node_first(
        self, y
    ) -> tp.Tuple[Bijector, object, object]:
        y_node_first = _move_node_dimension_to_beginning(y, self._inverse_event_ndims)
        parent_values = tf.nest.map_structure(
            lambda y_elem: tf.gather(
                y_elem,
                self._topology.parent_indices[self._topology.taxon_count :]
                - self._topology.taxon_count,
            ),
            y_node_first,
        )
        nonroot_input = tf.nest.map_structure(lambda x: x[:-1], self._input_node_first)
        bijectors = self._bijector_func(parent_values, nonroot_input)
        nonroot_values = tf.nest.map_structure(lambda y: y[:-1], y_node_first)
        root_values = tf.nest.map_structure(lambda y: y[-1], y_node_first)
        return bijectors, nonroot_values, root_values

    def _inverse_log_det_jacobian(self, y):
        (
            bijectors,
            nonroot_values,
            root_values,
        ) = self._inverse_bijectors_and_y_node_first(y)
        node_values = bijectors.inverse_log_det_jacobian(nonroot_values)
        return tf.reduce_sum(
            node_values, 0
        ) + self._root_bijector.inverse_log_det_jacobian(root_values)

    def _inverse(self, y):
        (
            bijectors,
            nonroot_y_values,
            root_y_values,
        ) = self._inverse_bijectors_and_y_node_first(y)
        nonroot_x_values_node_first = bijectors.inverse(nonroot_y_values)
        root_x_values = self._root_bijector.inverse(root_y_values)
        nonroot_x_values = tf.nest.map_structure(
            lambda x: distribution_util.move_dimension(x, 0, -1),
            nonroot_x_values_node_first,
        )

        return tf.nest.map_structure(
            lambda nonroot_elem, root_elem: tf.concat(
                [nonroot_elem, tf.expand_dims(root_elem, -1)], -1
            ),
            nonroot_x_values,
            root_x_values,
        )
