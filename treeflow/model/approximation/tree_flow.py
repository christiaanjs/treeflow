"""A variational approximation whose tree block is a tree normalising flow.

The unconstrained coordinates split into two blocks:

* the **parameter block** -- every non-tree model variable -- approximated by a
  mean-field or full-rank Gaussian, or by an inverse autoregressive flow;
* the **tree block** -- the ``taxon_count - 1`` unconstrained node-height
  coordinates -- pushed through a
  :class:`~treeflow.bijectors.tree_normalizing_flow.TreeNormalizingFlowBijector`
  and then an elementwise affine, before the existing node-height ratio
  machinery turns them into node heights.

The two blocks are coupled one way: the flow is *conditioned* on the parameter
block's (already transformed) values, so the tree distribution can depend on the
clock rate, population size and substitution parameters, and optionally on
per-branch parameters through the flow's per-node conditioner. The Jacobian is
block-triangular, so the log-density stays exact and cheap.

At initialisation the flow is the identity, so the approximation is exactly the
mean-field (or full-rank / IAF) one it is built on top of, initialised at
``init_loc`` -- the flow can only improve on that starting point.
"""

from functools import partial
import typing as tp

import tensorflow as tf
import tensorflow_probability.python.bijectors as tfb
import tensorflow_probability.python.distributions as tfd

from treeflow import DEFAULT_FLOAT_DTYPE_TF
from treeflow.bijectors.elementwise_node_flow import DEFAULT_NONLINEARITY
from treeflow.bijectors.tree_normalizing_flow import (
    DEFAULT_ORDER,
    TreeFlowParameters,
    TreeNormalizingFlowBijector,
)
from treeflow.model.approximation.full_rank import _FullRankAffineBijector
from treeflow.model.approximation.iaf import (
    DEFAULT_N_HIDDEN_LAYERS,
    DEFAULT_N_IAF_BIJECTORS,
)
from treeflow.model.approximation.root_full_rank import _MeanFieldAffineBijector
from treeflow.model.event_shape_bijector import (
    event_shape_fn as default_event_shape_fn,
    get_event_shape_and_space_bijector,
    get_fixed_topology_event_shape,
    get_fixed_topology_joint_bijector,
    get_unconstrained_init_values,
)
from treeflow.tree.topology.tensorflow_tree_topology import TensorflowTreeTopology

PARAMETER_APPROXIMATIONS = ("mean_field", "full_rank", "iaf")

DEFAULT_IAF_HIDDEN_UNITS = 32


class _TreeFlowCouplingBijector(tfb.Bijector):
    """Couples a parameter block to a conditional tree block.

    ``forward([x_p, x_t]) = [f(x_p), g_{f(x_p)}(x_t)]``, where ``f`` is the
    parameter block's bijector and ``g_a`` the tree block's bijector built with
    auxiliary input ``a``. The Jacobian is block-triangular, so its
    log-determinant is the sum of the two blocks'.
    """

    def __init__(
        self,
        parameter_bijector: tfb.Bijector,
        tree_bijector_fn: tp.Callable[[tf.Tensor], tfb.Bijector],
        parameter_size: int,
        tree_size: int,
        name: str = "TreeFlowCoupling",
        validate_args: bool = False,
    ):
        parameters = dict(locals())
        self.parameter_bijector = parameter_bijector
        self.tree_bijector_fn = tree_bijector_fn
        self.parameter_size = parameter_size
        self.tree_size = tree_size
        super().__init__(
            forward_min_event_ndims=1,
            inverse_min_event_ndims=1,
            validate_args=validate_args,
            parameters=parameters,
            name=name,
        )

    def _split(self, x):
        return x[..., : self.parameter_size], x[..., self.parameter_size :]

    def _forward(self, x):
        x_parameters, x_tree = self._split(x)
        y_parameters = self.parameter_bijector.forward(x_parameters)
        y_tree = self.tree_bijector_fn(y_parameters).forward(x_tree)
        return tf.concat([y_parameters, y_tree], axis=-1)

    def _inverse(self, y):
        y_parameters, y_tree = self._split(y)
        x_parameters = self.parameter_bijector.inverse(y_parameters)
        x_tree = self.tree_bijector_fn(y_parameters).inverse(y_tree)
        return tf.concat([x_parameters, x_tree], axis=-1)

    def _forward_log_det_jacobian(self, x):
        x_parameters, x_tree = self._split(x)
        y_parameters = self.parameter_bijector.forward(x_parameters)
        return self.parameter_bijector.forward_log_det_jacobian(
            x_parameters, event_ndims=1
        ) + self.tree_bijector_fn(y_parameters).forward_log_det_jacobian(
            x_tree, event_ndims=1
        )

    def _inverse_log_det_jacobian(self, y):
        y_parameters, y_tree = self._split(y)
        return self.parameter_bijector.inverse_log_det_jacobian(
            y_parameters, event_ndims=1
        ) + self.tree_bijector_fn(y_parameters).inverse_log_det_jacobian(
            y_tree, event_ndims=1
        )


def _node_features(
    value: tf.Tensor, size: int, node_count: int, taxon_count: int, name: str
) -> tf.Tensor:
    """Lay a model variable's unconstrained value out along the node axis.

    Two layouts are recognised, covering the per-node and per-branch parameters
    a phylogenetic model actually has:

    * one entry per internal node (``node_count``) -- used as is;
    * one entry per branch (``2 * taxon_count - 2``, indexed by the node below
      the branch) -- the branches above the internal nodes are taken, and the
      root, which has no branch above it, gets a zero.
    """
    if size == node_count:
        return value[..., tf.newaxis]
    if size == 2 * taxon_count - 2:
        internal = value[..., taxon_count:]  # branch above each non-root internal node
        root_pad = tf.zeros_like(value[..., :1])
        return tf.concat([internal, root_pad], axis=-1)[..., tf.newaxis]
    raise ValueError(
        f"Variable {name!r} has {size} unconstrained coordinates; a per-node "
        f"flow feature needs either {node_count} (one per internal node) or "
        f"{2 * taxon_count - 2} (one per branch)"
    )


def _softplus_inverse_one(dtype):
    """``softplus_inverse(1)``, so a raw scale of this value means scale 1."""
    return tf.cast(tf.math.log(tf.exp(tf.ones([], dtype=tf.float32)) - 1.0), dtype)


def get_tree_flow_approximation(
    model: tfd.JointDistribution,
    topology_pins: tp.Dict[str, TensorflowTreeTopology],
    init_loc=None,
    dtype=DEFAULT_FLOAT_DTYPE_TF,
    joint_bijector_func: tp.Callable[[tfd.JointDistribution], tfb.Composition] = None,
    event_shape_fn=None,
    tree_vars: tp.Iterable[str] = ("tree",),
    parameter_approximation: str = "mean_field",
    auxiliary_vars: tp.Optional[tp.Iterable[str]] = None,
    node_feature_vars: tp.Iterable[str] = (),
    num_layers: int = 1,
    order: str = DEFAULT_ORDER,
    nonlinearity: str = DEFAULT_NONLINEARITY,
    num_bins: int = 8,
    bounds: float = 3.0,
    conditioner_units: int = 8,
    iaf_hidden_units: int = DEFAULT_IAF_HIDDEN_UNITS,
    n_hidden_layers: int = DEFAULT_N_HIDDEN_LAYERS,
    n_iaf_bijectors: int = DEFAULT_N_IAF_BIJECTORS,
    use_native: tp.Union[bool, str] = "auto",
    unroll: tp.Union[bool, str] = "auto",
    seed=None,
) -> tp.Tuple[tfd.Distribution, tp.Dict[str, tf.Variable]]:
    """Build the tree normalising flow approximation.

    Parameters
    ----------
    topology_pins
        The pinned topologies; exactly one is required (the flow is structured
        by it).
    parameter_approximation
        How the non-tree variables are approximated: ``"mean_field"``
        (default), ``"full_rank"``, or ``"iaf"`` (an inverse autoregressive
        flow, as in :mod:`treeflow.model.approximation.iaf`).
    auxiliary_vars
        Names of the non-tree variables whose unconstrained values condition the
        flow. Defaults to all of them; pass ``()`` for an unconditional flow, or
        a subset to keep the conditioner small when the model has a variable
        whose dimension grows with the tree.
    node_feature_vars
        Names of non-tree variables to feed the flow's *per-node* conditioner --
        per-branch or per-internal-node quantities such as a relaxed clock's
        branch rates. They may also appear in ``auxiliary_vars``, though for a
        variable of tree-scale dimension the per-node route is the meaningful
        one.
    num_layers, order, nonlinearity, num_bins, bounds, conditioner_units
        Passed to :class:`TreeNormalizingFlowBijector`. ``nonlinearity="affine"``
        turns the tree block into a tree-structured Gaussian (no nonlinearity),
        which is the natural ablation against the spline flow.
    """
    if parameter_approximation not in PARAMETER_APPROXIMATIONS:
        raise ValueError(
            f"parameter_approximation must be one of {PARAMETER_APPROXIMATIONS}; "
            f"got {parameter_approximation!r}"
        )
    if len(topology_pins) != 1:
        raise ValueError(
            "get_tree_flow_approximation requires exactly one pinned topology; "
            f"got {sorted(topology_pins)}"
        )
    (topology,) = topology_pins.values()

    if joint_bijector_func is None:
        from treeflow.model.event_shape_bijector import (
            get_default_event_space_bijector,
        )

        joint_bijector_func = get_default_event_space_bijector
    if event_shape_fn is None:
        event_shape_fn = default_event_shape_fn

    (
        event_shape_and_space_bijector,
        base_event_shape,
    ) = get_event_shape_and_space_bijector(
        model,
        joint_bijector_func=joint_bijector_func,
        event_shape_fn=event_shape_fn,
    )
    init_loc_1d = get_unconstrained_init_values(
        model,
        event_shape_and_space_bijector,
        event_shape_fn=event_shape_fn,
        init=init_loc,
    )

    # `Restructure`/`Split` route data through `tf.nest`, which flattens dicts in
    # *sorted-key* order (see the same note in root_full_rank.py).
    names = sorted(base_event_shape.keys())
    tree_names = [name for name in names if name in tree_vars]
    if len(tree_names) != 1:
        raise ValueError(
            "get_tree_flow_approximation requires exactly one tree variable "
            f"among {list(tree_vars)}; found {tree_names} in model variables "
            f"{names}"
        )
    tree_name = tree_names[0]
    parameter_names = [name for name in names if name != tree_name]

    sizes = {name: base_event_shape[name].num_elements() for name in names}
    parameter_size = sum(sizes[name] for name in parameter_names)
    tree_size = sizes[tree_name]
    total_dim = parameter_size + tree_size

    taxon_count = int(tf.get_static_value(topology.taxon_count))
    node_count = taxon_count - 1
    if tree_size != node_count:
        raise ValueError(
            f"Tree variable {tree_name!r} has {tree_size} unconstrained "
            f"coordinates; the flow expects one per internal node ({node_count})"
        )

    if auxiliary_vars is None:
        auxiliary_vars = parameter_names
    auxiliary_names = [name for name in parameter_names if name in auxiliary_vars]
    node_feature_names = [name for name in parameter_names if name in node_feature_vars]
    unknown = sorted(
        (set(auxiliary_vars) | set(node_feature_vars)) - set(parameter_names)
    )
    if unknown:
        raise ValueError(
            f"{unknown} are not non-tree model variables; the model's non-tree "
            f"variables are {parameter_names}"
        )

    # Offset of each parameter variable within the (concatenated) parameter block.
    parameter_offsets = {}
    offset = 0
    for name in parameter_names:
        parameter_offsets[name] = offset
        offset += sizes[name]

    def slice_of(y_parameters, name):
        start = parameter_offsets[name]
        return y_parameters[..., start : start + sizes[name]]

    def auxiliary_input_fn(y_parameters):
        if not auxiliary_names:
            return None
        return tf.concat(
            [slice_of(y_parameters, name) for name in auxiliary_names], axis=-1
        )

    def node_input_fn(y_parameters):
        if not node_feature_names:
            return None
        return tf.concat(
            [
                _node_features(
                    slice_of(y_parameters, name),
                    sizes[name],
                    node_count,
                    taxon_count,
                    name,
                )
                for name in node_feature_names
            ],
            axis=-1,
        )

    def loc_init_for(keys):
        pieces = [
            tf.zeros(sizes[key], dtype=dtype)
            if init_loc_1d[key] is None
            else tf.cast(tf.reshape(init_loc_1d[key], [-1]), dtype)
            for key in keys
        ]
        if not pieces:
            return tf.zeros(0, dtype=dtype)
        return tf.concat(pieces, axis=0)

    softplus_inv1 = _softplus_inverse_one(dtype)

    # ---- Parameter block ----
    created_variables: tp.List[tf.Variable] = []
    parameter_loc = tf.Variable(
        loc_init_for(parameter_names), name="tree_flow_parameter_loc"
    )
    created_variables.append(parameter_loc)
    if parameter_approximation == "full_rank":
        parameter_scale_raw = tf.Variable(
            tf.linalg.diag(tf.fill([parameter_size], softplus_inv1)),
            name="tree_flow_parameter_scale_raw",
        )
        parameter_bijector = _FullRankAffineBijector(
            parameter_loc, parameter_scale_raw, name="TreeFlowParameterFullRank"
        )
    else:
        parameter_scale_raw = tf.Variable(
            tf.fill([parameter_size], softplus_inv1),
            name="tree_flow_parameter_scale_raw",
        )
        parameter_bijector = _MeanFieldAffineBijector(
            parameter_loc, parameter_scale_raw, name="TreeFlowParameterMeanField"
        )
    created_variables.append(parameter_scale_raw)

    iaf_bijectors = []
    if parameter_approximation == "iaf":
        iaf_bijectors = [
            tfb.Invert(
                tfb.MaskedAutoregressiveFlow(
                    shift_and_log_scale_fn=tfb.AutoregressiveNetwork(
                        params=2,
                        hidden_units=n_hidden_layers * [iaf_hidden_units],
                        activation="relu",
                        dtype=dtype,
                        event_shape=(parameter_size,),
                        # Small init so the IAF starts near the identity.
                        kernel_initializer=tf.keras.initializers.TruncatedNormal(
                            stddev=0.01
                        ),
                    )
                )
            )
            for _ in range(n_iaf_bijectors)
        ]
        # The affine is applied last, as in `iaf.py`, so the block starts as the
        # mean-field approximation at `init_loc`.
        parameter_bijector = tfb.Chain([parameter_bijector] + iaf_bijectors)

    # ---- Tree block: flow first (on standardised coordinates), then affine ----
    # The flow sees roughly standard-normal inputs, which is the range its
    # spline covers; the affine then moves them to the initial location/scale.
    tree_loc = tf.Variable(loc_init_for([tree_name]), name="tree_flow_tree_loc")
    tree_scale_raw = tf.Variable(
        tf.fill([tree_size], softplus_inv1), name="tree_flow_tree_scale_raw"
    )
    created_variables += [tree_loc, tree_scale_raw]
    tree_affine_bijector = _MeanFieldAffineBijector(
        tree_loc, tree_scale_raw, name="TreeFlowTreeAffine"
    )

    num_children = int(topology.node_child_indices.shape[-1])
    flow_parameters = TreeFlowParameters(
        node_count=node_count,
        num_children=num_children,
        num_layers=num_layers,
        num_auxiliary=sum(sizes[name] for name in auxiliary_names),
        num_node_features=len(node_feature_names),
        conditioner_units=conditioner_units,
        nonlinearity=nonlinearity,
        num_bins=num_bins,
        dtype=dtype,
    )
    flow_template = TreeNormalizingFlowBijector(
        topology,
        flow_parameters=flow_parameters,
        order=order,
        bounds=bounds,
        use_native=use_native,
        unroll=unroll,
        dtype=dtype,
    )
    created_variables += list(flow_parameters.trainable_variables)

    def tree_bijector_fn(y_parameters):
        flow = flow_template.copy_with_inputs(
            auxiliary_input=auxiliary_input_fn(y_parameters),
            node_input=node_input_fn(y_parameters),
        )
        return tfb.Chain([tree_affine_bijector, flow])

    coupling_bijector = _TreeFlowCouplingBijector(
        parameter_bijector, tree_bijector_fn, parameter_size, tree_size
    )

    # The coupling emits the parameter block then the tree block; `Split` needs
    # each variable's contiguous slice in `names` order. `Permute` reorders the
    # coordinates (volume-preserving, so no log-det contribution).
    source_offsets = dict(parameter_offsets)
    source_offsets[tree_name] = parameter_size
    permutation = [
        i
        for name in names
        for i in range(source_offsets[name], source_offsets[name] + sizes[name])
    ]
    assert sorted(permutation) == list(range(total_dim))
    permute_bijector = tfb.Permute(permutation=tf.constant(permutation, dtype=tf.int32))

    restructure_bijector = tfb.Restructure(
        output_structure=tf.nest.pack_sequence_as(
            base_event_shape, range(len(base_event_shape))
        ),
    )
    flat_sizes = restructure_bijector.inverse_event_shape(base_event_shape)
    split_bijector = tfb.Split(tf.cast(tf.concat(flat_sizes, axis=0), tf.int32))

    chain_bijector = tfb.Chain(
        [
            event_shape_and_space_bijector,
            restructure_bijector,
            split_bijector,
            permute_bijector,
            coupling_bijector,
        ]
    )
    base_dist = tfd.Sample(
        tfd.Normal(tf.constant(0.0, dtype=dtype), tf.constant(1.0, dtype=dtype)),
        total_dim,
    )
    distribution = tfd.TransformedDistribution(base_dist, chain_bijector)

    if iaf_bijectors:
        distribution.sample(seed=seed)  # build the autoregressive networks
        created_variables += [
            weight
            for bijector in iaf_bijectors
            for weight in bijector.bijector._shift_and_log_scale_fn._network.weights
        ]

    variables_dict = {variable.name: variable for variable in created_variables}
    return distribution, variables_dict


def get_fixed_topology_tree_flow_approximation(
    model: tfd.JointDistribution,
    topology_pins: tp.Dict[str, TensorflowTreeTopology],
    init_loc=None,
    dtype=DEFAULT_FLOAT_DTYPE_TF,
    use_native: tp.Union[bool, str] = "auto",
    unroll: tp.Union[bool, str] = "auto",
    **kwargs,
) -> tp.Tuple[tfd.Distribution, tp.Dict[str, tf.Variable]]:
    """:func:`get_tree_flow_approximation` for a model with a pinned topology.

    Matches the ``ApproximationBuilder`` protocol, so it can be passed straight
    to
    :func:`~treeflow.vi.fixed_topology_advi.fit_fixed_topology_variational_approximation`
    as ``approx_fn``.
    """
    bijector_func = partial(
        get_fixed_topology_joint_bijector,
        topology_pins=topology_pins,
        use_native=use_native,
        unroll=unroll,
    )
    event_shape_fn = partial(
        get_fixed_topology_event_shape, topology_pins=topology_pins
    )
    return get_tree_flow_approximation(
        model,
        topology_pins=topology_pins,
        init_loc=init_loc,
        dtype=dtype,
        joint_bijector_func=bijector_func,
        event_shape_fn=event_shape_fn,
        use_native=use_native,
        unroll=unroll,
        **kwargs,
    )


__all__ = [
    "get_tree_flow_approximation",
    "get_fixed_topology_tree_flow_approximation",
    "PARAMETER_APPROXIMATIONS",
]
