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
    xla_compatible: bool = False,
    unroll: tp.Union[bool, str] = "auto",
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
        Only affects the dynamic (``TensorArray``) path. How children are read each
        step and TensorArrays are initialized.

        XLA compiling the TensorArray-based traversal may result in quadratic scaling,
        so this is not actually recommended.
    unroll
        Whether to unroll the traversal into a straight-line graph for the (static)
        topology instead of running a ``tf.while_loop`` over a ``TensorArray``:

        - ``"auto"`` (default): unroll iff the topology index tensors are statically
          known (``tf.get_static_value`` succeeds — true when the topology is a
          constant/eager tensor rather than a ``tf.function`` placeholder).
        - ``True``/``False``: force on/off (``True`` raises if the topology is not
          static).

        The unrolled graph is much faster (linear, no per-step loop overhead) and is
        differentiated by a single reverse sweep, at the cost of a per-topology
        trace/compile whose size grows with the node count. Prefer it for a fixed
        topology evaluated many times (e.g. VI/MCMC on one tree); the dynamic path is
        the right choice when the topology varies per call.
    """

    post_np = tf.get_static_value(topology.postorder_node_indices)
    child_np = tf.get_static_value(topology.child_indices)
    static = post_np is not None and child_np is not None
    if unroll is True and not static:
        raise ValueError(
            "unroll=True requires a statically known topology, but "
            "tf.get_static_value could not fold the topology index tensors "
            "(is the topology a tf.function argument rather than a constant?)."
        )
    if static and unroll is not False:
        return _postorder_unrolled(post_np, child_np, mapping, input, leaf_init)
    return _postorder_tensorarray(topology, mapping, input, leaf_init, xla_compatible)


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
def _postorder_tensorarray(topology, mapping, input, leaf_init, xla_compatible):
    """Dynamic-topology traversal: a bounded `tf.while_loop` over a TensorArray.
    Decorated with `tf.function` so that AutoGraph works.
    """

    taxon_count = topology.taxon_count
    node_count = 2 * taxon_count - 1

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
        else:
            return ta.scatter(tf.range(taxon_count), x)

    tensorarrays = tf.nest.map_structure(init_ta, leaf_init)
    postorder_node_indices = topology.postorder_node_indices
    child_indices = topology.child_indices
    num_children = child_indices.shape[-1]

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
        tas = tf.nest.map_structure(lambda x, ta: ta.write(node_index, x), output, tas)
        return tas

    for i in range(taxon_count - 1):
        # Autograph is proven faster than tf.while_loop here
        tensorarrays = body(i, tensorarrays)
    return tf.nest.map_structure(lambda x: x.stack(), tensorarrays)
