import typing as tp
import attr
import tensorflow as tf
from treeflow.tree.topology.tensorflow_tree_topology import TensorflowTreeTopology


TInputStructure = tp.TypeVar("TInputStructure")
TOutputStructure = tp.TypeVar("TOutputStructure")

# The three traversal strategies `unroll` selects between (plus "auto"):
#   "unrolled"    -- straight-line Python-unrolled graph, NO TensorArray. Needs the
#                    topology index *values* statically known. Fastest overall, and
#                    the only path that XLA-compiles for value+gradient.
#   "tensorarray" -- Python-unrolled loop that writes a TensorArray. Needs only the
#                    node *count* (shape) static; the index values may be tensors.
#                    Might incur significant compilation time for large trees, but
#                    faster than "while_loop" once compiled.
#   "while_loop"  -- AutoGraph `tf.while_loop` over a TensorArray. Needs nothing
#                    static; O(1) graph.
UNROLL_MODES = ("unrolled", "tensorarray", "while_loop")


@attr.attrs(auto_attribs=True)
class PostorderTopologyData:
    child_indices: tf.Tensor


def static_taxon_count(topology) -> tp.Optional[int]:
    """The taxon count as a Python int if statically known, else ``None``.

    ``topology.taxon_count`` is prefer_static (derived from ``parent_indices``'s
    shape, which ``tf.function`` preserves), so it resolves to a concrete value when
    the node count is known -- unlike derived index arrays such as
    ``preorder_node_indices``, whose static shape a ``boolean_mask`` can drop even
    when the input shape is known."""
    tc = tf.get_static_value(topology.taxon_count)
    return None if tc is None else int(tc)


def _resolve_unroll_mode(unroll, values_static, count_static):
    """Map the ``unroll`` argument to a concrete traversal mode, validating that the
    requested mode's staticness requirement is met."""
    if unroll == "auto":
        # Prefer the no-TensorArray unroll: it is the fastest overall and the only
        # path that XLA-compiles for value+gradient. Fall back to a TensorArray loop
        # (unrolled while the count is static, otherwise a dynamic while_loop).
        if values_static:
            return "unrolled"
        if count_static:
            return "tensorarray"
        return "while_loop"
    if unroll == "unrolled":
        if not values_static:
            raise ValueError(
                "unroll='unrolled' requires statically-known topology index *values* "
                "(tf.get_static_value must fold them); the topology is dynamic. Use "
                "'tensorarray' (static count) or 'while_loop' (fully dynamic)."
            )
        return "unrolled"
    if unroll == "tensorarray":
        if not count_static:
            raise ValueError(
                "unroll='tensorarray' requires a statically-known node *count* "
                "(topology index shape); it is unknown. Use 'while_loop'."
            )
        return "tensorarray"
    if unroll == "while_loop":
        return "while_loop"
    raise ValueError(
        f"unroll must be 'auto' or one of {UNROLL_MODES}; got {unroll!r}."
    )


def postorder_node_traversal(
    topology: TensorflowTreeTopology,
    mapping: tp.Callable[
        [TOutputStructure, TInputStructure, PostorderTopologyData], TOutputStructure
    ],
    input: TInputStructure,
    leaf_init: TOutputStructure,
    xla_compatible: bool = False,
    unroll: str = "auto",
) -> TOutputStructure:
    """Postorder (children-before-parents) traversal over a fixed topology.

    Carries an arbitrary ``tf.nest`` output structure per node and applies
    ``mapping`` at each internal node. General enough to host the phylogenetic
    likelihood (``input`` = per-node child transition matrices, ``leaf_init`` =
    leaf partials, ``mapping`` = the Felsenstein combine step; the rescaled variant
    can carry ``(partials, log_scale)`` as the output structure and sum the
    ``log_scale`` component afterwards).

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
    xla_compatible
        Only affects the TensorArray paths (``"tensorarray"``/``"while_loop"``). When
        ``True`` the TensorArray is seeded with ``unstack`` and children are read with
        per-child ``read`` + ``stack``; when ``False`` (default) it is seeded with
        ``scatter`` and children are read with ``gather``. ``scatter``/``gather`` have
        no XLA kernel, so ``xla_compatible=True`` is required to XLA-compile the
        TensorArray *forward* pass (see the ``unroll`` notes on XLA). It can scale
        quadratically under XLA, so it is for forward-only XLA use, not a default.
    unroll
        Which traversal strategy to use -- one of ``"auto"`` (default), ``"unrolled"``,
        ``"tensorarray"`` or ``"while_loop"``:

        - ``"unrolled"``: a straight-line Python-unrolled graph with **no
          TensorArray**. Requires the topology index *values* to be statically known
          (``tf.get_static_value`` folds them -- true for a constant/eager topology).
          Fastest overall (linear, no per-step loop overhead; one reverse-sweep
          gradient).
        - ``"tensorarray"``: a Python-unrolled loop that writes a ``TensorArray``.
          Requires only the node *count* (index shape) to be static; the index values
          may be runtime tensors (e.g. a topology routed through a JointDistribution).
        - ``"while_loop"``: an AutoGraph ``tf.while_loop`` over a ``TensorArray``.
          Requires nothing static; O(1) graph, so it is the right choice when the
          topology (or its size) varies per call or is very large.
        - ``"auto"``: pick the fastest feasible -- ``"unrolled"`` if the values are
          static, else ``"tensorarray"`` if the count is static, else ``"while_loop"``.

        XLA / ``jit_compile`` notes (empirically verified):

        - ``"unrolled"`` is the **only** mode that XLA-compiles for **value+gradient**.
          Under ``jit_compile`` always use ``"unrolled"`` (so a static-value topology),
          since the TensorArray modes' backward pass materialises a ``TensorList`` that
          cannot cross the XLA/TF boundary.
        - The ``"tensorarray"``/``"while_loop"`` modes XLA-compile only the **forward**
          pass, and only with ``xla_compatible=True`` (the default scatter/gather ops
          have no XLA kernel). Their value+gradient never XLA-compiles regardless of
          ``xla_compatible``.
        - Wrapping a TensorArray traversal in an inner ``tf.function`` does **not**
          shield it from an enclosing ``jit_compile``: the compilation propagates
          through (even with ``jit_compile=False`` on the inner function), so it is not
          an escape hatch for the above.
    """
    post_static = tf.get_static_value(topology.postorder_node_indices)
    child_static = tf.get_static_value(topology.child_indices)
    values_static = post_static is not None and child_static is not None
    count_static = static_taxon_count(topology) is not None

    mode = _resolve_unroll_mode(unroll, values_static, count_static)
    if mode == "unrolled":
        return _postorder_unrolled(post_static, child_static, mapping, input, leaf_init)
    return _postorder_tensorarray(
        topology, mapping, input, leaf_init, xla_compatible,
        while_loop=(mode == "while_loop"),
    )


def _postorder_unrolled(post_np, child_np, mapping, input, leaf_init):
    """Straight-line traversal for a statically known topology (no TensorArray)."""
    node_count = int(child_np.shape[0])
    n_internal = int(post_np.shape[0])
    taxon_count = node_count - n_internal

    vals = [None] * node_count
    for i in range(taxon_count):
        vals[i] = tf.nest.map_structure(lambda x: x[i], leaf_init)
    for node_index in post_np.tolist():
        children = child_np[node_index]
        child_output = tf.nest.map_structure(
            lambda *cs: tf.stack(cs, axis=0), *[vals[int(c)] for c in children]
        )
        node_input = tf.nest.map_structure(lambda x: x[node_index - taxon_count], input)
        topology_data = PostorderTopologyData(child_indices=tf.constant(children))
        vals[node_index] = mapping(child_output, node_input, topology_data)
    return tf.nest.map_structure(lambda *xs: tf.stack(xs, axis=0), *vals)


@tf.function
def _postorder_tensorarray(
    topology, mapping, input, leaf_init, xla_compatible, while_loop
):
    """TensorArray traversal. ``while_loop=False`` Python-unrolls the loop (needs a
    static node count); ``while_loop=True`` runs an AutoGraph ``tf.while_loop`` (no
    staticness required). Decorated with ``tf.function`` so AutoGraph converts the
    ``for i in tf.range(...)`` loop into a ``while_loop``.
    """
    taxon_count = topology.taxon_count
    node_count = 2 * taxon_count - 1
    # Tensors so a (possibly symbolic) loop variable can index them: a static NumPy
    # topology exposes these as NumPy arrays, which a while_loop counter can't index.
    postorder_node_indices = tf.convert_to_tensor(topology.postorder_node_indices)
    child_indices = tf.convert_to_tensor(topology.child_indices)
    num_children = child_indices.shape[-1]

    def init_ta(x):
        ta = tf.TensorArray(
            dtype=x.dtype,
            size=node_count,
            element_shape=x.shape[1:],
            clear_after_read=False,
        )
        if xla_compatible:
            pad = tf.zeros(
                tf.concat([[node_count - taxon_count], tf.shape(x)[1:]], 0), x.dtype
            )
            return ta.unstack(tf.concat([x, pad], 0))
        return ta.scatter(tf.range(taxon_count), x)

    tensorarrays = tf.nest.map_structure(init_ta, leaf_init)

    def read_children(ta, node_child_indices):
        if xla_compatible:
            return tf.stack(
                [ta.read(node_child_indices[c]) for c in range(num_children)]
            )
        return ta.gather(node_child_indices)

    def body(i, tas):
        node_index = postorder_node_indices[i]
        node_child_indices = child_indices[node_index]
        child_output = tf.nest.map_structure(
            lambda ta: read_children(ta, node_child_indices), tas
        )
        node_input = tf.nest.map_structure(lambda x: x[node_index - taxon_count], input)
        topology_data = PostorderTopologyData(child_indices=node_child_indices)
        output = mapping(child_output, node_input, topology_data)
        return tf.nest.map_structure(lambda x, ta: ta.write(node_index, x), output, tas)

    if while_loop:
        for i in tf.range(taxon_count - 1):  # AutoGraph -> tf.while_loop
            tensorarrays = body(i, tensorarrays)
    else:
        # Static node count -> Python-unrolled loop (no while_loop).
        for i in range(static_taxon_count(topology) - 1):
            tensorarrays = body(i, tensorarrays)
    return tf.nest.map_structure(lambda x: x.stack(), tensorarrays)
