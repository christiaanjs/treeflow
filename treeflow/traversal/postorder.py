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


def _node_vjp(mapping, primal_children, node_input, topology_data, cotangent):
    """Vector-Jacobian product of ``mapping`` at a single node.

    Uses TF autodiff *of the user mapping function* (the "input function") to push the
    output ``cotangent`` back onto the mapping's differentiable arguments. Returns the
    cotangents ``(g_children, g_input)`` w.r.t. ``primal_children`` (the stacked child
    outputs for a postorder node, or the parent output for a preorder node) and
    ``node_input``. ``topology_data is None`` selects the two-argument preorder mapping
    signature; otherwise the three-argument postorder one. ``None`` gradients (an
    argument the mapping does not use) are returned as explicit zeros so callers can
    accumulate unconditionally."""
    co_flat = tf.nest.flatten(primal_children)
    ni_flat = tf.nest.flatten(node_input)
    sources = co_flat + ni_flat
    with tf.GradientTape() as tape:
        tape.watch(sources)
        children_r = tf.nest.pack_sequence_as(primal_children, co_flat)
        input_r = tf.nest.pack_sequence_as(node_input, ni_flat)
        if topology_data is None:
            out = mapping(children_r, input_r)
        else:
            out = mapping(children_r, input_r, topology_data)
    grads = tape.gradient(out, sources, output_gradients=cotangent)
    grads = [tf.zeros_like(s) if g is None else g for g, s in zip(grads, sources)]
    n_co = len(co_flat)
    g_children = tf.nest.pack_sequence_as(primal_children, grads[:n_co])
    g_input = tf.nest.pack_sequence_as(node_input, grads[n_co:])
    return g_children, g_input


def _traversal_custom_gradient(input, init, forward, backward):
    """Wrap a traversal ``forward`` pass in ``tf.custom_gradient``.

    ``forward(input, init) -> result`` runs the ordinary traversal (a ``tf.nest``
    structure matching ``init``, stacked over nodes). ``backward(result, input, init,
    dy) -> (d_input, d_init)`` computes the input/seed gradients from the output
    cotangent ``dy`` using a complementary traversal (see ``_*_backward_*``). Only the
    tensor leaves of ``input`` and ``init`` are exposed as differentiable arguments, so
    the custom backward replaces autodiff through the traversal graph entirely."""
    flat_input = tf.nest.flatten(input)
    flat_init = tf.nest.flatten(init)
    n_in = len(flat_input)

    @tf.custom_gradient
    def run(*flat_args):
        input_r = tf.nest.pack_sequence_as(input, list(flat_args[:n_in]))
        init_r = tf.nest.pack_sequence_as(init, list(flat_args[n_in:]))
        result = forward(input_r, init_r)

        def grad(*dy_flat, variables=None):
            if variables:
                raise NotImplementedError(
                    "Traversal custom_gradient does not support a `mapping` that "
                    "closes over tf.Variables; pass differentiable tensors through "
                    "`input`/`init` instead."
                )
            dy = tf.nest.pack_sequence_as(init, list(dy_flat))
            d_input, d_init = backward(result, input_r, init_r, dy)
            return tf.nest.flatten(d_input) + tf.nest.flatten(d_init)

        return tf.nest.flatten(result), grad

    flat_out = run(*(flat_input + flat_init))
    return tf.nest.pack_sequence_as(init, list(flat_out))


def postorder_node_traversal(
    topology: TensorflowTreeTopology,
    mapping: tp.Callable[
        [TOutputStructure, TInputStructure, PostorderTopologyData], TOutputStructure
    ],
    input: TInputStructure,
    leaf_init: TOutputStructure,
    xla_compatible: bool = False,
    unroll: str = "auto",
    custom_gradient: bool = False,
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
    custom_gradient
        When ``True`` the traversal is wrapped in ``tf.custom_gradient``: the forward
        pass is unchanged, but gradients w.r.t. ``input`` and ``leaf_init`` are computed
        by a complementary **preorder** reverse-accumulation rather than by autodiff
        through the traversal graph. Each node's local vector-Jacobian product is taken
        with ``tf.GradientTape`` over ``mapping`` itself, and the cotangents are
        propagated from parents to children. Because the backward is then itself a
        *forward-style* traversal (no autodiff through a ``TensorArray`` backward), it
        composes cleanly with the TensorArray modes and keeps their value+gradient
        XLA-compilable with ``xla_compatible=True``. The ``mapping`` must not close over
        ``tf.Variable``\\ s under this option (route any differentiable tensors through
        ``input``/``leaf_init`` instead).
    """
    post_static = tf.get_static_value(topology.postorder_node_indices)
    child_static = tf.get_static_value(topology.child_indices)
    values_static = post_static is not None and child_static is not None
    count_static = static_taxon_count(topology) is not None

    mode = _resolve_unroll_mode(unroll, values_static, count_static)

    def forward(input, leaf_init):
        if mode == "unrolled":
            return _postorder_unrolled(
                post_static, child_static, mapping, input, leaf_init
            )
        return _postorder_tensorarray(
            topology, mapping, input, leaf_init, xla_compatible,
            while_loop=(mode == "while_loop"),
        )

    if not custom_gradient:
        return forward(input, leaf_init)

    def backward(result, input, leaf_init, dy):
        if mode == "unrolled":
            return _postorder_backward_unrolled(
                post_static, child_static, mapping, input, result, dy
            )
        return _postorder_backward_tensorarray(
            topology, mapping, input, result, dy, xla_compatible,
            while_loop=(mode == "while_loop"),
        )

    return _traversal_custom_gradient(input, leaf_init, forward, backward)


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


def _postorder_backward_unrolled(post_np, child_np, mapping, input, result, dy):
    """Reverse-accumulating preorder sweep: the gradient of the postorder forward.

    Visits nodes parents-before-children (reverse postorder). ``gbar[n]`` is node ``n``'s
    accumulated output cotangent, seeded from ``dy[n]`` (the direct contribution of
    node ``n`` to the stacked output) and completed by its parent before ``n`` is
    processed. At each internal node the mapping's VJP splits ``gbar`` into the
    ``node_input`` gradient and per-child contributions, which are added onto the
    children's ``gbar``. Leaf gradients are the leaves' final ``gbar`` (leaves seed the
    forward via identity)."""
    node_count = int(child_np.shape[0])
    n_internal = int(post_np.shape[0])
    taxon_count = node_count - n_internal

    gbar = [tf.nest.map_structure(lambda x, n=n: x[n], dy) for n in range(node_count)]
    d_input_nodes = [None] * n_internal

    for node_index in reversed(post_np.tolist()):
        children = child_np[node_index]
        child_output = tf.nest.map_structure(
            lambda r: tf.stack([r[int(c)] for c in children], axis=0), result
        )
        node_input = tf.nest.map_structure(
            lambda x: x[node_index - taxon_count], input
        )
        topology_data = PostorderTopologyData(child_indices=tf.constant(children))
        g_child, g_input = _node_vjp(
            mapping, child_output, node_input, topology_data, gbar[node_index]
        )
        d_input_nodes[node_index - taxon_count] = g_input
        for j, c in enumerate(children):
            ci = int(c)
            contribution = tf.nest.map_structure(lambda x, j=j: x[j], g_child)
            gbar[ci] = tf.nest.map_structure(
                lambda a, b: a + b, gbar[ci], contribution
            )

    d_input = tf.nest.map_structure(lambda *xs: tf.stack(xs, axis=0), *d_input_nodes)
    d_leaf = tf.nest.map_structure(
        lambda *xs: tf.stack(xs, axis=0), *gbar[:taxon_count]
    )
    return d_input, d_leaf


@tf.function
def _postorder_backward_tensorarray(
    topology, mapping, input, result, dy, xla_compatible, while_loop
):
    """TensorArray form of :func:`_postorder_backward_unrolled` for dynamic topologies.

    Carries the per-node accumulated cotangent ``gbar`` (seeded from ``dy``) and the
    per-internal-node ``input`` gradient as TensorArrays, sweeping nodes in reverse
    postorder (parents before children). ``xla_compatible`` selects ``unstack``/``read``
    over ``gather`` for the child reads so the whole value+gradient stays XLA-compilable.
    """
    taxon_count = topology.taxon_count
    node_count = 2 * taxon_count - 1
    postorder_node_indices = tf.convert_to_tensor(topology.postorder_node_indices)
    child_indices = tf.convert_to_tensor(topology.child_indices)
    num_children = child_indices.shape[-1]
    n_internal = taxon_count - 1

    def init_gbar(d):
        ta = tf.TensorArray(
            dtype=d.dtype,
            size=node_count,
            element_shape=d.shape[1:],
            clear_after_read=False,
        )
        return ta.unstack(d)

    gbar = tf.nest.map_structure(init_gbar, dy)

    d_input_ta = tf.nest.map_structure(
        lambda x: tf.TensorArray(
            dtype=x.dtype,
            size=n_internal,
            element_shape=x.shape[1:],
            clear_after_read=False,
        ),
        input,
    )

    if xla_compatible:
        result_ta = tf.nest.map_structure(
            lambda r: tf.TensorArray(
                dtype=r.dtype,
                size=node_count,
                element_shape=r.shape[1:],
                clear_after_read=False,
            ).unstack(r),
            result,
        )

    def read_children(node_child_indices):
        if xla_compatible:
            return tf.nest.map_structure(
                lambda ta: tf.stack(
                    [ta.read(node_child_indices[c]) for c in range(num_children)],
                    axis=0,
                ),
                result_ta,
            )
        return tf.nest.map_structure(
            lambda r: tf.gather(r, node_child_indices, axis=0), result
        )

    def body(i, gbar, d_input_ta):
        node_index = postorder_node_indices[i]
        node_child_indices = child_indices[node_index]
        child_output = read_children(node_child_indices)
        node_input = tf.nest.map_structure(
            lambda x: x[node_index - taxon_count], input
        )
        gbar_node = tf.nest.map_structure(lambda ta: ta.read(node_index), gbar)
        topology_data = PostorderTopologyData(child_indices=node_child_indices)
        g_child, g_input = _node_vjp(
            mapping, child_output, node_input, topology_data, gbar_node
        )
        d_input_ta = tf.nest.map_structure(
            lambda ta, x: ta.write(node_index - taxon_count, x), d_input_ta, g_input
        )

        def add_children(ta, gc):
            for c in range(num_children):
                idx = node_child_indices[c]
                ta = ta.write(idx, ta.read(idx) + gc[c])
            return ta

        gbar = tf.nest.map_structure(add_children, gbar, g_child)
        return gbar, d_input_ta

    if while_loop:
        for j in tf.range(n_internal):  # AutoGraph -> tf.while_loop
            gbar, d_input_ta = body(n_internal - 1 - j, gbar, d_input_ta)
    else:
        static_n_internal = static_taxon_count(topology) - 1
        for j in range(static_n_internal):
            gbar, d_input_ta = body(static_n_internal - 1 - j, gbar, d_input_ta)

    d_input = tf.nest.map_structure(lambda ta: ta.stack(), d_input_ta)
    d_leaf = tf.nest.map_structure(lambda ta: ta.stack()[:taxon_count], gbar)
    return d_input, d_leaf
