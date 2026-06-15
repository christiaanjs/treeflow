import typing as tp
import tensorflow as tf
from treeflow.tree.topology.tensorflow_tree_topology import TensorflowTreeTopology

TInputStructure = tp.TypeVar("TInputStructure")
TOutputStructure = tp.TypeVar("TOutputStructure")


def preorder_traversal(
    topology: TensorflowTreeTopology,
    mapping: tp.Callable[[TOutputStructure, TInputStructure], TOutputStructure],
    input: TInputStructure,
    root_init: TOutputStructure,
    unroll: tp.Union[bool, str] = "auto",
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
        Whether to unroll the traversal into a straight-line graph for the (static)
        topology instead of running a ``tf.while_loop`` over a ``TensorArray``:

        - ``"auto"`` (default): unroll iff the topology index tensors are statically
          known (``tf.get_static_value`` succeeds — true when the topology is a
          constant/eager tensor rather than a ``tf.function`` placeholder).
        - ``True``/``False``: force on/off (``True`` raises if not static).

        The unrolled graph is much faster (linear, no per-step loop overhead) and is
        differentiated by a single reverse sweep, at the cost of a per-topology
        trace/compile whose size grows with the node count. Prefer it for a fixed
        topology evaluated many times; the dynamic path handles a varying topology.
    """
    # `preorder_node_indices` is static-preferring (see TensorflowTreeTopology), so
    # it folds via get_static_value when the topology is constant -- including inside
    # a tf.function with a captured topology -- enabling the unrolled path there too.
    pre_np = tf.get_static_value(topology.preorder_node_indices)
    par_np = tf.get_static_value(topology.parent_indices)
    static = pre_np is not None and par_np is not None
    if unroll is True and not static:
        raise ValueError(
            "unroll=True requires a statically known topology, but "
            "tf.get_static_value could not fold the topology index tensors "
            "(is the topology a tf.function argument rather than a constant?)."
        )
    if static and unroll is not False:
        return _preorder_unrolled(pre_np, par_np, mapping, input, root_init)
    return _preorder_tensorarray(topology, mapping, input, root_init)


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


def _preorder_tensorarray(topology, mapping, input, root_init):
    """Dynamic-topology traversal: a bounded ``tf.while_loop`` over a TensorArray."""
    taxon_count = topology.taxon_count
    n_internal = taxon_count - 1
    # Convert to tensors so the (symbolic) loop variable can index them: a static
    # NumPy topology exposes these as NumPy arrays, which can't be indexed by a
    # traced tf.while_loop counter.
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

    def cond(k, tas):
        return k < n_internal

    def body(k, tas):
        i = preorder_node_indices[k] - taxon_count
        parent_index = parent_indices[i]
        parent_output = tf.nest.map_structure(lambda ta: ta.read(parent_index), tas)
        node_input = tf.nest.map_structure(lambda x: x[i], input)
        output = mapping(parent_output, node_input)
        tas = tf.nest.map_structure(lambda x, ta: ta.write(i, x), output, tas)
        return k + 1, tas

    _, tensorarrays = tf.while_loop(
        cond, body, (tf.constant(1), tensorarrays),
        maximum_iterations=n_internal - 1,
    )
    return tf.nest.map_structure(lambda x: x.stack(), tensorarrays)
