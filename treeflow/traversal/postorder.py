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
    child_read_stack: bool = False,
) -> TOutputStructure:
    """Postorder (children-before-parents) traversal over a fixed topology.

    Parameters
    ----------
    topology
        The tree topology providing ``postorder_node_indices`` and ``child_indices``.
    mapping
        ``(child_output, node_input, topology_data) -> node_output`` applied at each
        internal node, where ``child_output`` stacks the children's outputs on axis 0.
    input
        Per-internal-node input structure, indexed ``input[node_index - taxon_count]``.
    leaf_init
        Per-leaf output structure (leaves on axis 0), used to seed the traversal.
    child_read_stack
        How children are read from the per-node ``TensorArray`` each step:

        - ``False`` (default): use ``TensorArray.gather`` — a single batched read,
          fastest for eager/graph execution.
        - ``True``: read each child with ``TensorArray.read`` and ``tf.stack`` them.
          Slightly slower (~7% in graph mode) but its gradient avoids
          ``TensorListScatterIntoExistingList``, which has **no XLA kernel** on CPU.
          Set this to ``True`` only when you need to ``jit_compile`` the *value and
          gradient* of the traversal; the forward pass compiles either way.

        ``True`` requires a statically known number of children per node
        (``topology.child_indices.shape[-1]``), which holds for bifurcating trees.
    """
    taxon_count = topology.taxon_count
    node_count = 2 * taxon_count - 1

    # Seed the per-node TensorArrays in one vectorised `unstack` rather than a
    # sequential per-leaf write loop. We pad the leaf block with zeros for the
    # internal slots (overwritten during the traversal); `unstack` is used instead
    # of `scatter` because `TensorListScatterIntoExistingList` has no XLA kernel.
    def init_ta(x):
        pad = tf.zeros(
            tf.concat([[node_count - taxon_count], tf.shape(x)[1:]], 0), x.dtype
        )
        return tf.TensorArray(
            dtype=x.dtype,
            size=node_count,
            element_shape=x.shape[1:],
            clear_after_read=False,
        ).unstack(tf.concat([x, pad], 0))

    tensorarrays = tf.nest.map_structure(init_ta, leaf_init)
    postorder_node_indices = topology.postorder_node_indices
    child_indices = topology.child_indices
    num_children = child_indices.shape[-1]

    def read_children(ta, node_child_indices):
        if child_read_stack:
            return tf.stack(
                [ta.read(node_child_indices[c]) for c in range(num_children)]
            )
        return ta.gather(node_child_indices)

    # Explicit bounded while_loop (rather than a `for i in tf.range(...)` relying on
    # AutoGraph): runs in eager/graph/XLA without conversion, and the static
    # `maximum_iterations` lets XLA fix the TensorArray list size when compiling.
    def cond(i, tas):
        return i < taxon_count - 1

    def body(i, tas):
        node_index = postorder_node_indices[i]
        node_child_indices = child_indices[node_index]
        child_output = tf.nest.map_structure(
            lambda ta: read_children(ta, node_child_indices), tas
        )
        node_input = tf.nest.map_structure(lambda x: x[node_index - taxon_count], input)
        topology_data = PostorderTopologyData(child_indices=node_child_indices)
        output = mapping(child_output, node_input, topology_data)
        tas = tf.nest.map_structure(
            lambda x, ta: ta.write(node_index, x), output, tas
        )
        return i + 1, tas

    _, tensorarrays = tf.while_loop(
        cond, body, (tf.constant(0), tensorarrays),
        maximum_iterations=taxon_count - 1,
    )
    return tf.nest.map_structure(lambda x: x.stack(), tensorarrays)
