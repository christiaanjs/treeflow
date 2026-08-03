"""A normalising flow on the per-node coordinates of a fixed tree topology.

The flow acts on the vector of unconstrained per-internal-node coordinates that
sits *underneath* the node-height ratio transform -- i.e. on the same
``[..., internal_node]`` vector that
:class:`~treeflow.bijectors.node_height_ratio_bijector.NodeHeightRatioChainBijector`
maps to node heights -- so a variational approximation can put a flexible,
tree-structured distribution on node heights while every constraint (ordering,
positivity) is still enforced by the existing ratio machinery.

Structure: a **traversal sandwich**, repeated ``num_layers`` times.

1. a *postorder* (tip-to-root) affine map -- each node reads its children's
   outputs;
2. a *learnable elementwise nonlinearity* -- shared across the tree, with
   per-node affine conditioning, and no tree traversal at all;
3. a *preorder* (root-to-tip) affine map -- each node reads its parent's output.

Each affine map is triangular in its own traversal order (log-det-Jacobian
``sum_i log scale[i]``, inverse by a single gather), and the nonlinearity is
elementwise and strictly monotone, so the composition is a bijection with a
cheap exact density.

The nonlinearity is pluggable (``nonlinearity=``, see
:mod:`treeflow.bijectors.elementwise_node_flow`): a monotone rational-quadratic
``"spline"`` (the default -- most flexible, identity tails), ``"sinh_arcsinh"``
(unbounded, reshapes the tails rather than the bulk), or ``"affine"`` -- no
nonlinearity at all, which reduces the flow to a composition of triangular
affine maps, i.e. a Gaussian with tree-structured covariance in
``O(internal_node)`` parameters. That last one is both a useful family in its own
right and the ablation that separates what the tree structure buys from what the
nonlinearity buys.

Why postorder first (the default). The two orders are not equivalent, and the
difference is exactly which pairs of nodes end up coupled. The postorder map
makes a node's output depend on its *descendants*; the preorder map makes it
depend on its *ancestors*. Composing them up-then-down gives, for node ``i``,
dependence on the descendants of the ancestors of ``i`` -- and since the root is
an ancestor of everything, that is **every** node, after a single layer. The
other way round, down-then-up couples ``i`` only to its own ancestors and
descendants; two cousins, whose posterior heights are coupled through their
common ancestor, are left independent until a second layer is stacked on.
(``test_sandwich_dependence_structure`` checks both patterns.) The up-then-down
schedule is the same collect-then-distribute pass that exact belief propagation
on a tree uses, and for the same reason.

Both orders are valid bijections and ``order="preorder_first"`` remains
available -- with ``num_layers >= 2`` it too couples every pair of nodes, and it
ends with the tip-to-root map, which is the direction the node-height ratio
transform downstream of the flow does not itself provide.

Invertibility. Three conditions, all enforced by construction rather than
assumed:

* every affine ``scale`` is a **softplus** output, hence strictly positive, so
  each triangular map is non-singular (and its log-det-Jacobian is finite);
* the nonlinearity is strictly increasing and analytically invertible, whichever
  of the three is chosen;
* the recursion weights are bounded -- ``parent_weight = tanh(raw)`` (so
  ``|w| < 1`` and the root-to-tip recursion cannot amplify along a deep path)
  and ``child_weight = tanh(raw) / num_children`` (so the weights at a node sum
  to less than 1 in absolute value going up the tree). These bounds are not
  needed for invertibility -- the triangular structure guarantees that -- but
  they keep the map numerically well conditioned on large trees, where an
  unbounded recursion coefficient compounds over the tree's depth.

Conditioning. The per-node parameters of every layer are the sum of

* a free per-node base parameter block (the flow's own parameters);
* an optional contribution from **auxiliary variables** -- e.g. the
  unconstrained clock rate, population size or substitution-model parameters --
  so the tree distribution can depend on the rest of the model (this is what
  makes a flow *conditional*: for fixed auxiliary values the map is still a
  bijection in the tree coordinates);
* an optional contribution from **per-node variables** -- e.g. per-branch
  relaxed-clock rates -- shared across nodes by a small network, so per-branch
  parameters can inform the node they belong to.

At initialisation every base parameter gives scale 1, shift 0, zero recursion
weights and an identity nonlinearity, and both conditioners' output weights are
zero, so the flow starts as the **exact identity**: dropping it into an existing
approximation changes nothing until it is trained.
"""

import typing as tp

import tensorflow as tf
from tensorflow_probability.python.bijectors import Bijector, Chain
from tensorflow_probability.python.math import softplus_inverse

from treeflow import DEFAULT_FLOAT_DTYPE_TF
from treeflow.bijectors.elementwise_node_flow import (
    DEFAULT_BOUNDS,
    DEFAULT_NONLINEARITY,
    DEFAULT_NUM_BINS,
    ElementwiseNodeFlow,
    build_nonlinearity,
    check_nonlinearity,
    shared_parameter_shapes,
)
from treeflow.bijectors.tree_affine_bijector import (
    PostorderAffineBijector,
    PreorderAffineBijector,
)
from treeflow.tree.topology.tensorflow_tree_topology import TensorflowTreeTopology

ORDERS = ("postorder_first", "preorder_first")
DEFAULT_ORDER = "postorder_first"

# Offsets into a layer's per-node raw parameter block. The trailing
# ``num_children`` entries are the postorder map's child weights.
_PRE_SCALE, _PRE_SHIFT, _PRE_WEIGHT = 0, 1, 2
_MID_SCALE, _MID_SHIFT = 3, 4
_POST_SCALE, _POST_SHIFT = 5, 6
_NUM_SCALAR_PARAMS = 7

DEFAULT_CONDITIONER_UNITS = 8


class TreeFlowLayerParameters(tp.NamedTuple):
    """Constrained per-node parameters of one sandwich layer."""

    pre_scale: tf.Tensor
    pre_shift: tf.Tensor
    parent_weight: tf.Tensor
    mid_scale: tf.Tensor
    mid_shift: tf.Tensor
    post_scale: tf.Tensor
    post_shift: tf.Tensor
    child_weight: tf.Tensor
    #: Raw parameters of the layer's shared nonlinearity, keyed by name.
    nonlinearity: tp.Dict[str, tf.Tensor]


class TreeFlowParameters(tf.Module):
    """The trainable parameters of a :class:`TreeNormalizingFlowBijector`.

    Held separately from the bijector so that the same parameters can be reused
    by bijectors built with different conditioning inputs (see
    :meth:`TreeNormalizingFlowBijector.copy_with_inputs`) -- as a variational
    approximation must, since the auxiliary values change with every sample.

    Parameters
    ----------
    node_count
        Number of internal nodes (``taxon_count - 1``).
    num_children
        Children per internal node (2 for a bifurcating tree).
    num_layers
        Number of sandwich layers.
    num_auxiliary, num_node_features
        Widths of the two optional conditioning inputs; 0 disables a conditioner
        (and creates none of its variables).
    conditioner_units
        Hidden width of each conditioner's single hidden layer.
    nonlinearity
        Which shared nonlinearity each layer uses (see
        :mod:`treeflow.bijectors.elementwise_node_flow`); decides which shared
        parameters are created.
    num_bins
        Bins of the shared spline, when ``nonlinearity="spline"``.
    """

    def __init__(
        self,
        node_count: int,
        num_children: int = 2,
        num_layers: int = 1,
        num_auxiliary: int = 0,
        num_node_features: int = 0,
        conditioner_units: int = DEFAULT_CONDITIONER_UNITS,
        nonlinearity: str = DEFAULT_NONLINEARITY,
        num_bins: int = DEFAULT_NUM_BINS,
        dtype=DEFAULT_FLOAT_DTYPE_TF,
        name: str = "tree_flow_parameters",
    ):
        super().__init__(name=name)
        self.nonlinearity = check_nonlinearity(nonlinearity)
        self.node_count = int(node_count)
        self.num_children = int(num_children)
        self.num_layers = int(num_layers)
        self.num_auxiliary = int(num_auxiliary)
        self.num_node_features = int(num_node_features)
        self.num_bins = int(num_bins)
        self.dtype = dtype
        self.num_parameters = _NUM_SCALAR_PARAMS + self.num_children

        shape = (self.num_layers, self.node_count, self.num_parameters)
        softplus_inv1 = tf.cast(softplus_inverse(tf.constant(1.0, tf.float64)), dtype)
        # Identity initialisation: unit scales (softplus_inverse(1) in the three
        # scale slots), zero shifts, zero recursion weights.
        scale_slots = tf.reduce_sum(
            tf.one_hot(
                [_PRE_SCALE, _MID_SCALE, _POST_SCALE],
                self.num_parameters,
                dtype=dtype,
            ),
            axis=0,
        )
        self.base = tf.Variable(
            tf.broadcast_to(softplus_inv1 * scale_slots, shape), name="tree_flow_base"
        )

        # The shared nonlinearity's raw parameters, one set per layer (the
        # "affine" nonlinearity has none); zeros give the identity.
        self.nonlinearity_parameters = {
            parameter_name: tf.Variable(
                tf.zeros((self.num_layers,) + parameter_shape, dtype=dtype),
                name=f"tree_flow_nonlinearity_{parameter_name}",
            )
            for parameter_name, parameter_shape in shared_parameter_shapes(
                self.nonlinearity, self.num_bins
            ).items()
        }

        if self.num_auxiliary > 0:
            self.auxiliary_hidden_kernel = tf.Variable(
                tf.random.stateless_normal(
                    (self.num_auxiliary, conditioner_units),
                    seed=(0, 1),
                    dtype=dtype,
                )
                / tf.cast(tf.sqrt(float(self.num_auxiliary)), dtype),
                name="tree_flow_auxiliary_hidden_kernel",
            )
            self.auxiliary_hidden_bias = tf.Variable(
                tf.zeros((conditioner_units,), dtype=dtype),
                name="tree_flow_auxiliary_hidden_bias",
            )
            # Zero output weights => the conditioner starts as a no-op, so the
            # flow starts as the identity whatever the auxiliary values are.
            self.auxiliary_output_kernel = tf.Variable(
                tf.zeros((conditioner_units,) + shape, dtype=dtype),
                name="tree_flow_auxiliary_output_kernel",
            )

        if self.num_node_features > 0:
            self.node_hidden_kernel = tf.Variable(
                tf.random.stateless_normal(
                    (self.num_node_features, conditioner_units),
                    seed=(0, 2),
                    dtype=dtype,
                )
                / tf.cast(tf.sqrt(float(self.num_node_features)), dtype),
                name="tree_flow_node_hidden_kernel",
            )
            self.node_hidden_bias = tf.Variable(
                tf.zeros((conditioner_units,), dtype=dtype),
                name="tree_flow_node_hidden_bias",
            )
            self.node_output_kernel = tf.Variable(
                tf.zeros(
                    (conditioner_units, self.num_layers, self.num_parameters),
                    dtype=dtype,
                ),
                name="tree_flow_node_output_kernel",
            )

    def raw_parameters(
        self,
        auxiliary_input: tp.Optional[tf.Tensor] = None,
        node_input: tp.Optional[tf.Tensor] = None,
    ) -> tf.Tensor:
        """Unconstrained per-node parameters, shape ``[..., layer, node, param]``.

        ``auxiliary_input`` has shape ``[..., num_auxiliary]`` and
        ``node_input`` shape ``[..., internal_node, num_node_features]``; their
        (broadcastable) batch dimensions end up as the leading dimensions of the
        result, which is what lets the flow's parameters vary from sample to
        sample with the rest of the model.
        """
        raw = self.base
        if auxiliary_input is not None:
            if self.num_auxiliary == 0:
                raise ValueError(
                    "This TreeFlowParameters was built without an auxiliary "
                    "conditioner (num_auxiliary=0)"
                )
            auxiliary_input = tf.cast(auxiliary_input, self.dtype)
            hidden = tf.nn.tanh(
                tf.linalg.matvec(
                    self.auxiliary_hidden_kernel, auxiliary_input, transpose_a=True
                )
                + self.auxiliary_hidden_bias
            )
            raw = raw + tf.einsum("...h,hlnp->...lnp", hidden, self.auxiliary_output_kernel)
        if node_input is not None:
            if self.num_node_features == 0:
                raise ValueError(
                    "This TreeFlowParameters was built without a per-node "
                    "conditioner (num_node_features=0)"
                )
            node_input = tf.cast(node_input, self.dtype)
            # The hidden layer is shared across nodes: a per-branch parameter
            # informs its own node, without a private network per node.
            hidden = tf.nn.tanh(
                tf.einsum("...nf,fh->...nh", node_input, self.node_hidden_kernel)
                + self.node_hidden_bias
            )
            raw = raw + tf.einsum("...nh,hlp->...lnp", hidden, self.node_output_kernel)
        return raw

    def layer_parameters(
        self, raw_parameters: tf.Tensor, layer: int
    ) -> TreeFlowLayerParameters:
        """Constrained parameters of one layer, from :meth:`raw_parameters`."""
        raw = raw_parameters[..., layer, :, :]
        num_children = tf.cast(self.num_children, self.dtype)
        return TreeFlowLayerParameters(
            pre_scale=tf.math.softplus(raw[..., _PRE_SCALE]),
            pre_shift=raw[..., _PRE_SHIFT],
            # |tanh| < 1: the root-to-tip recursion cannot amplify along a path.
            parent_weight=tf.math.tanh(raw[..., _PRE_WEIGHT]),
            mid_scale=tf.math.softplus(raw[..., _MID_SCALE]),
            mid_shift=raw[..., _MID_SHIFT],
            post_scale=tf.math.softplus(raw[..., _POST_SCALE]),
            post_shift=raw[..., _POST_SHIFT],
            # Divided by the number of children so a node's weights sum to less
            # than 1 in absolute value going up the tree.
            child_weight=tf.math.tanh(raw[..., _NUM_SCALAR_PARAMS:]) / num_children,
            nonlinearity={
                parameter_name: variable[layer]
                for parameter_name, variable in self.nonlinearity_parameters.items()
            },
        )

    @property
    def variables_dict(self) -> tp.Dict[str, tf.Variable]:
        """The flow's variables keyed by name (for VI traces / warm starts)."""
        return {variable.name: variable for variable in self.trainable_variables}


class TreeNormalizingFlowBijector(Bijector):
    """Tree-structured normalising flow on per-internal-node coordinates.

    Parameters
    ----------
    topology
        The fixed tree topology the flow is structured by.
    auxiliary_input
        Optional tensor with shape ``[..., num_auxiliary]``: values of the
        variables the tree distribution should depend on (clock rate, population
        size, substitution parameters, ...), typically in unconstrained space
        and drawn from the same sample as the coordinates being transformed.
    node_input
        Optional tensor with shape ``[..., internal_node, num_node_features]``:
        per-node/per-branch variables.
    flow_parameters
        An existing :class:`TreeFlowParameters` to reuse. When omitted one is
        built, sized from the topology and the conditioning inputs.
    num_layers
        How many times the sandwich is repeated.
    order
        ``"postorder_first"`` (default) or ``"preorder_first"`` -- which affine
        traversal runs first; see the module docstring.
    nonlinearity
        Which elementwise nonlinearity sits between the two affine maps:
        ``"spline"`` (default), ``"sinh_arcsinh"`` or ``"affine"`` (none, making
        the flow a tree-structured Gaussian). See
        :mod:`treeflow.bijectors.elementwise_node_flow`.
    use_native, unroll
        Forwarded to the affine tree maps: the native C++ ops vs the
        pure-TensorFlow traversal, and how the latter is unrolled.
    """

    def __init__(
        self,
        topology: TensorflowTreeTopology,
        auxiliary_input: tp.Optional[tf.Tensor] = None,
        node_input: tp.Optional[tf.Tensor] = None,
        flow_parameters: tp.Optional[TreeFlowParameters] = None,
        num_layers: int = 1,
        order: str = DEFAULT_ORDER,
        nonlinearity: str = DEFAULT_NONLINEARITY,
        num_bins: int = DEFAULT_NUM_BINS,
        bounds: float = DEFAULT_BOUNDS,
        conditioner_units: int = DEFAULT_CONDITIONER_UNITS,
        use_native: tp.Union[bool, str] = "auto",
        unroll: tp.Union[bool, str] = "auto",
        dtype=DEFAULT_FLOAT_DTYPE_TF,
        name: str = "TreeNormalizingFlowBijector",
        validate_args: bool = False,
    ):
        parameters = dict(locals())
        if order not in ORDERS:
            raise ValueError(f"order must be one of {ORDERS}; got {order!r}")
        self.topology = topology
        self.order = order
        self.bounds = bounds
        self.use_native = use_native
        self.unroll = unroll
        self._auxiliary_input = auxiliary_input
        self._node_input = node_input

        if flow_parameters is None:
            node_count = int(tf.get_static_value(topology.taxon_count)) - 1
            num_children = int(topology.node_child_indices.shape[-1])
            flow_parameters = TreeFlowParameters(
                node_count=node_count,
                num_children=num_children,
                num_layers=num_layers,
                num_auxiliary=(
                    0 if auxiliary_input is None else int(auxiliary_input.shape[-1])
                ),
                num_node_features=(
                    0 if node_input is None else int(node_input.shape[-1])
                ),
                conditioner_units=conditioner_units,
                nonlinearity=nonlinearity,
                num_bins=num_bins,
                dtype=dtype,
            )
        self.flow_parameters = flow_parameters

        super().__init__(
            forward_min_event_ndims=1,
            inverse_min_event_ndims=1,
            dtype=flow_parameters.dtype,
            validate_args=validate_args,
            parameters=parameters,
            name=name,
        )

    @property
    def num_layers(self) -> int:
        return self.flow_parameters.num_layers

    @property
    def nonlinearity(self) -> str:
        return self.flow_parameters.nonlinearity

    @property
    def auxiliary_input(self) -> tp.Optional[tf.Tensor]:
        return self._auxiliary_input

    @property
    def node_input(self) -> tp.Optional[tf.Tensor]:
        return self._node_input

    def copy_with_inputs(
        self,
        auxiliary_input: tp.Optional[tf.Tensor] = None,
        node_input: tp.Optional[tf.Tensor] = None,
    ) -> "TreeNormalizingFlowBijector":
        """A flow with the same (shared) parameters and new conditioning inputs."""
        return TreeNormalizingFlowBijector(
            self.topology,
            auxiliary_input=auxiliary_input,
            node_input=node_input,
            flow_parameters=self.flow_parameters,
            order=self.order,
            bounds=self.bounds,
            use_native=self.use_native,
            unroll=self.unroll,
            name=self.name,
            validate_args=self.validate_args,
        )

    def _layer_bijectors(
        self, layer_parameters: TreeFlowLayerParameters
    ) -> tp.List[Bijector]:
        """One layer's bijectors, in the order they are *applied*."""
        preorder = PreorderAffineBijector(
            self.topology,
            layer_parameters.pre_scale,
            layer_parameters.pre_shift,
            layer_parameters.parent_weight,
            use_native=self.use_native,
            unroll=self.unroll,
        )
        elementwise = ElementwiseNodeFlow(
            layer_parameters.mid_scale,
            layer_parameters.mid_shift,
            build_nonlinearity(
                self.nonlinearity,
                layer_parameters.nonlinearity,
                bounds=self.bounds,
                validate_args=self.validate_args,
            ),
            validate_args=self.validate_args,
        )
        postorder = PostorderAffineBijector(
            self.topology,
            layer_parameters.post_scale,
            layer_parameters.post_shift,
            layer_parameters.child_weight,
            use_native=self.use_native,
            unroll=self.unroll,
        )
        if self.order == "preorder_first":
            return [preorder, elementwise, postorder]
        return [postorder, elementwise, preorder]

    def chain(self) -> Chain:
        """The flow as a plain :class:`Chain`, for the current inputs.

        Rebuilt on every call so that gradients reach the flow's variables (and
        so that a change in the conditioning inputs is picked up).
        """
        raw = self.flow_parameters.raw_parameters(
            self._auxiliary_input, self._node_input
        )
        applied = [
            bijector
            for layer in range(self.num_layers)
            for bijector in self._layer_bijectors(
                self.flow_parameters.layer_parameters(raw, layer)
            )
        ]
        # Chain applies its last bijector first.
        return Chain(list(reversed(applied)))

    def _forward(self, x):
        return self.chain().forward(x)

    def _inverse(self, y):
        return self.chain().inverse(y)

    def _forward_log_det_jacobian(self, x):
        return self.chain().forward_log_det_jacobian(x, event_ndims=1)

    def _inverse_log_det_jacobian(self, y):
        return self.chain().inverse_log_det_jacobian(y, event_ndims=1)


__all__ = [
    "TreeNormalizingFlowBijector",
    "TreeFlowParameters",
    "TreeFlowLayerParameters",
    "ORDERS",
]
