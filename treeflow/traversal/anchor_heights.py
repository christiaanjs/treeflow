import numpy as np
from treeflow.tree.rooted.numpy_rooted_tree import NumpyRootedTree
from treeflow.tree.topology.tensorflow_tree_topology import TensorflowTreeTopology
import tensorflow as tf


def get_anchor_heights(tree: NumpyRootedTree) -> np.ndarray:
    taxon_count = tree.taxon_count
    anchor_heights = np.zeros_like(tree.heights)
    anchor_heights[..., :taxon_count] = tree.heights[..., :taxon_count]

    for i in tree.topology.postorder_node_indices:
        anchor_heights[..., i] = np.max(
            anchor_heights[..., tree.topology.child_indices[i]], axis=-1
        )

    return anchor_heights[..., taxon_count:]


def get_anchor_heights_tensor(
    topology: TensorflowTreeTopology, sampling_times: tf.Tensor
):
    taxon_count = topology.taxon_count
    anchor_heights = tf.TensorArray(sampling_times.dtype, size=taxon_count * 2 - 1)
    for i in tf.range(taxon_count):
        anchor_heights = anchor_heights.write(i, sampling_times[..., i])
    child_indices = topology.child_indices
    for i in topology.postorder_node_indices:
        child_anchor_heights = anchor_heights.gather(child_indices[i])
        anchor_heights = anchor_heights.write(
            i, tf.reduce_max(child_anchor_heights, axis=0)
        )

    rank = tf.shape(tf.shape(sampling_times))[0]
    perm = tf.concat([tf.range(1, rank), [0]], axis=0)
    return tf.transpose(anchor_heights.gather(topology.postorder_node_indices), perm)


__all__ = [get_anchor_heights.__name__, get_anchor_heights_tensor.__name__]
