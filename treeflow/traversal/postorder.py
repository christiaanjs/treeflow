import typing as tp
import attr
import tensorflow as tf
from treeflow.tree.topology.tensorflow_tree_topology import TensorflowTreeTopology


TInputStructure = tp.TypeVar("TInputStructure")
TOutputStructure = tp.TypeVar("TOutputStructure")


@attr.attrs(auto_attribs=True)
class PostorderTopologyData:
    child_indices: tf.Tensor


def postorder_node_traversal(
    topology: TensorflowTreeTopology,
    mapping: tp.Callable[
        [TOutputStructure, TInputStructure, PostorderTopologyData], TOutputStructure
    ],
    input: TInputStructure,
    leaf_init: TOutputStructure,
) -> TOutputStructure:
    taxon_count = topology.taxon_count
    node_count = 2 * taxon_count - 1
    tensorarrays = tf.nest.map_structure(
        lambda x: tf.TensorArray(
            dtype=x.dtype,
            size=node_count,
            element_shape=x.shape[1:],
            clear_after_read=False,
        ),
        leaf_init,
    )
    for i in tf.range(taxon_count):
        tensorarrays = tf.nest.map_structure(
            lambda x, ta: ta.write(i, x[i]), leaf_init, tensorarrays
        )
    postorder_node_indices = topology.postorder_node_indices
    child_indices = topology.child_indices
    for i in tf.range(taxon_count - 1):
        node_index = postorder_node_indices[i]
        node_child_indices = child_indices[node_index]
        child_output = tf.nest.map_structure(
            lambda ta: ta.gather(node_child_indices), tensorarrays
        )
        node_input = tf.nest.map_structure(lambda x: x[node_index - taxon_count], input)
        topology_data = PostorderTopologyData(child_indices=node_child_indices)
        output = mapping(child_output, node_input, topology_data)
        tensorarrays = tf.nest.map_structure(
            lambda x, ta: ta.write(node_index, x), output, tensorarrays
        )
    return tf.nest.map_structure(lambda x: x.stack(), tensorarrays)
