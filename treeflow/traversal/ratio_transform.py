import typing as tp
import tensorflow as tf
from treeflow.traversal.preorder import preorder_traversal
from treeflow.tree.topology.tensorflow_tree_topology import TensorflowTreeTopology


def move_outside_axis_to_inside(x):
    rank = tf.rank(x)
    perm = tf.concat([tf.range(1, rank), [0]], axis=0)
    return tf.transpose(x, perm)


def _node_axis_to_front(x):
    """Move the (last) node axis to the front -- inverse of move_outside_axis_to_inside."""
    rank = tf.rank(x)
    perm = tf.concat([[rank - 1], tf.range(0, rank - 1)], axis=0)
    return tf.transpose(x, perm)


def ratios_to_node_heights(
    topology: TensorflowTreeTopology,
    ratios: tf.Tensor,
    anchor_heights: tf.Tensor,
    unroll: tp.Union[bool, str] = "auto",
):
    """Node-height ratio transform, on the generic ``preorder_traversal``.

    The node axis of ``ratios``/``anchor_heights`` is the last axis, indexed by
    internal-node id.

    unroll
        Forwarded to :func:`preorder_traversal`: ``"auto"`` unrolls when the topology
        is statically known (it is when ``topology`` is a constant/eager object, even
        captured inside a ``tf.function``), ``True`` forces it (raising otherwise),
        ``False`` keeps the dynamic ``tf.while_loop``.
    """
    n_internal = ratios.shape[-1]
    ratios_nf = _node_axis_to_front(ratios)          # [n_internal, ...batch]
    anchor_nf = _node_axis_to_front(anchor_heights)  # [n_internal, ...] (broadcasts)
    root_init = ratios_nf[n_internal - 1] + anchor_nf[n_internal - 1]

    def mapping(parent_height, node_input):
        ratio_i, anchor_i = node_input
        return (parent_height - anchor_i) * ratio_i + anchor_i

    heights = preorder_traversal(
        topology, mapping, (ratios_nf, anchor_nf), root_init, unroll=unroll
    )
    return move_outside_axis_to_inside(heights)
