import typing as tp
import tensorflow as tf
from treeflow.tree.topology.tensorflow_tree_topology import TensorflowTreeTopology

TInputStructure = tp.TypeVar("TInputStructure")
TOutputStructure = tp.TypeVar("TOutputStructure")


@tf.function
def preorder_traversal(
    topology: TensorflowTreeTopology,
    mapping: tp.Callable[[TOutputStructure, TInputStructure], TOutputStructure],
    input: TInputStructure,
    root_init: TOutputStructure,
) -> TOutputStructure:
    taxon_count = topology.taxon_count
    node_count = taxon_count - 1
    tensorarrays = tf.nest.map_structure(
        lambda x: tf.TensorArray(
            dtype=x.dtype,
            size=node_count,
            element_shape=x.shape,
            clear_after_read=False,
        ),
        root_init,
    )
    tensorarrays = tf.nest.map_structure(
        lambda x, ta: ta.write(node_count - 1, x), root_init, tensorarrays
    )

    parent_indices = topology.parent_indices[topology.taxon_count :] - taxon_count
    for i in topology.preorder_node_indices[1:] - taxon_count:
        parent_index = parent_indices[i]
        parent_output = tf.nest.map_structure(
            lambda ta: ta.read(parent_index), tensorarrays
        )
        node_input = tf.nest.map_structure(lambda x: x[i], input)
        output = mapping(parent_output, node_input)
        tensorarrays = tf.nest.map_structure(
            lambda x, ta: ta.write(i, x), output, tensorarrays
        )

    return tf.nest.map_structure(lambda x: x.stack(), tensorarrays)
