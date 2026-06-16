import typing as tp
import tensorflow as tf
from treeflow.tree.topology.tensorflow_tree_topology import TensorflowTreeTopology
from treeflow.traversal.postorder import (
    UNROLL_MODES,
    _resolve_unroll_mode,
    static_taxon_count,
)

TInputStructure = tp.TypeVar("TInputStructure")
TOutputStructure = tp.TypeVar("TOutputStructure")


def preorder_traversal(
    topology: TensorflowTreeTopology,
    mapping: tp.Callable[[TOutputStructure, TInputStructure], TOutputStructure],
    input: TInputStructure,
    root_init: TOutputStructure,
    unroll: str = "auto",
) -> TOutputStructure:
    """Preorder (parents-before-children) traversal over a fixed topology.

    Carries an arbitrary ``tf.nest`` output structure per internal node and applies
    ``mapping`` at each non-root internal node. General enough to host the
    node-height ratio transform (``input`` = per-node ``(ratio, anchor)``,
    ``root_init`` = the root height, ``mapping`` = the affine update).

    Parameters
    ----------
    topology
        The tree topology providing ``preorder_node_indices`` and ``parent_indices``.
    mapping
        ``(parent_output, node_input) -> node_output`` applied at each non-root
        internal node.
    input
        Per-internal-node input structure, indexed ``input[node_index - taxon_count]``.
    root_init
        Output structure for the root node, used to seed the traversal.
    unroll
        Which traversal strategy to use -- one of ``"auto"`` (default), ``"unrolled"``,
        ``"tensorarray"`` or ``"while_loop"``:

        - ``"unrolled"``: a straight-line Python-unrolled graph with **no
          TensorArray**. Requires the topology index *values* to be statically known
          (``tf.get_static_value`` folds them). Fastest overall.
        - ``"tensorarray"``: a Python-unrolled loop that writes a ``TensorArray``.
          Requires only the node *count* (index shape) to be static.
        - ``"while_loop"``: an AutoGraph ``tf.while_loop`` over a ``TensorArray``.
          Requires nothing static; O(1) graph, for a varying or very large topology.
        - ``"auto"``: ``"unrolled"`` if values static, else ``"tensorarray"`` if count
          static, else ``"while_loop"``.

        XLA / ``jit_compile`` notes (empirically verified): ``"unrolled"`` is the only
        mode that XLA-compiles for **value+gradient** -- under ``jit_compile`` always
        use it (a static-value topology), because the TensorArray modes' backward pass
        materialises a ``TensorList`` that cannot cross the XLA/TF boundary. The
        TensorArray modes XLA-compile only the **forward** pass. Wrapping the
        TensorArray traversal in an inner ``tf.function`` does not shield it from an
        enclosing ``jit_compile`` (compilation propagates through).
    """
    # `preorder_node_indices` is static-preferring (see TensorflowTreeTopology), so it
    # folds via get_static_value when the topology is constant -- including inside a
    # tf.function with a captured topology -- enabling the unrolled path there too.
    pre_np = tf.get_static_value(topology.preorder_node_indices)
    par_np = tf.get_static_value(topology.parent_indices)
    values_static = pre_np is not None and par_np is not None
    count_static = static_taxon_count(topology) is not None

    mode = _resolve_unroll_mode(unroll, values_static, count_static)
    if mode == "unrolled":
        return _preorder_unrolled(pre_np, par_np, mapping, input, root_init)
    return _preorder_tensorarray(
        topology, mapping, input, root_init, while_loop=(mode == "while_loop")
    )


def _preorder_unrolled(pre_np, par_np, mapping, input, root_init):
    """Straight-line traversal for a statically known topology (no TensorArray)."""
    n_internal = int(pre_np.shape[0])
    taxon_count = n_internal + 1
    # Parent of each non-root internal node, in internal-node space.
    parent_internal = par_np[taxon_count:] - taxon_count

    vals = [None] * n_internal
    vals[int(pre_np[0]) - taxon_count] = root_init  # root (== n_internal - 1)
    for full_id in pre_np[1:].tolist():
        i = full_id - taxon_count
        node_input = tf.nest.map_structure(lambda x: x[i], input)
        vals[i] = mapping(vals[int(parent_internal[i])], node_input)
    return tf.nest.map_structure(lambda *xs: tf.stack(xs, axis=0), *vals)


@tf.function
def _preorder_tensorarray(topology, mapping, input, root_init, while_loop):
    """TensorArray traversal. ``while_loop=False`` Python-unrolls the loop (needs a
    static node count); ``while_loop=True`` runs an AutoGraph ``tf.while_loop``.
    Decorated with ``tf.function`` so AutoGraph converts the ``for k in tf.range(...)``
    loop into a ``while_loop``.
    """
    taxon_count = topology.taxon_count
    n_internal = taxon_count - 1
    # Tensors so a (possibly symbolic) loop variable can index them.
    preorder_node_indices = tf.convert_to_tensor(topology.preorder_node_indices)
    parent_indices = tf.convert_to_tensor(topology.parent_indices)[taxon_count:] - (
        taxon_count
    )

    tensorarrays = tf.nest.map_structure(
        lambda x: tf.TensorArray(
            dtype=x.dtype,
            size=n_internal,
            element_shape=x.shape,
            clear_after_read=False,
        ),
        root_init,
    )
    tensorarrays = tf.nest.map_structure(
        lambda x, ta: ta.write(n_internal - 1, x), root_init, tensorarrays
    )

    def body(k, tas):
        i = preorder_node_indices[k] - taxon_count
        parent_index = parent_indices[i]
        parent_output = tf.nest.map_structure(lambda ta: ta.read(parent_index), tas)
        node_input = tf.nest.map_structure(lambda x: x[i], input)
        output = mapping(parent_output, node_input)
        return tf.nest.map_structure(lambda x, ta: ta.write(i, x), output, tas)

    if while_loop:
        for k in tf.range(1, n_internal):  # AutoGraph -> tf.while_loop
            tensorarrays = body(k, tensorarrays)
    else:
        # Static node count -> Python-unrolled loop (no while_loop).
        for k in range(1, static_taxon_count(topology) - 1):
            tensorarrays = body(k, tensorarrays)
    return tf.nest.map_structure(lambda x: x.stack(), tensorarrays)
