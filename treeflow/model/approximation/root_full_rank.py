from functools import partial
import typing as tp
import tensorflow as tf
import tensorflow_probability.python.distributions as tfd
import tensorflow_probability.python.bijectors as tfb
from treeflow import DEFAULT_FLOAT_DTYPE_TF
from treeflow.tree.topology.tensorflow_tree_topology import TensorflowTreeTopology
from treeflow.model.event_shape_bijector import (
    get_event_shape_and_space_bijector,
    get_unconstrained_init_values,
    event_shape_fn as default_event_shape_fn,
    get_fixed_topology_joint_bijector,
    get_fixed_topology_event_shape,
)
from treeflow.model.approximation.full_rank import _FullRankAffineBijector

# Keys used for the two pieces the tree variable's unconstrained representation
# is split into. They are not model variable names, so they are prefixed to
# avoid any possible collision with one.
_ROOT_KEY = "__tree_root__"
_RATIOS_KEY = "__tree_ratios__"


class _MeanFieldAffineBijector(tfb.Bijector):
    """Elementwise affine (shift + softplus-scale) transform for independent
    mean-field coordinates. Stores the raw variables directly (as
    ``_FullRankAffineBijector`` does) and recomputes the transform on every
    call, rather than passing a precomputed ``tf.math.softplus(raw_scale)``
    tensor into ``tfb.Scale`` -- which freezes that tensor's value at
    bijector-construction time and breaks gradient flow to ``raw_scale``.
    """

    def __init__(self, loc, raw_scale, name="MeanFieldAffine"):
        self.loc = loc
        self.raw_scale = raw_scale
        super().__init__(forward_min_event_ndims=1, name=name)

    def _scale(self):
        return tf.math.softplus(self.raw_scale)

    def _forward(self, x):
        return self.loc + self._scale() * x

    def _inverse(self, y):
        return (y - self.loc) / self._scale()

    def _forward_log_det_jacobian(self, x):
        return tf.reduce_sum(tf.math.log(self._scale()))


def get_root_full_rank_approximation(
    model: tfd.JointDistribution,
    topology_pins: tp.Dict[str, TensorflowTreeTopology],
    init_loc=None,
    dtype=DEFAULT_FLOAT_DTYPE_TF,
    joint_bijector_func: tp.Callable[[tfd.JointDistribution], tfb.Composition] = None,
    event_shape_fn=None,
    tree_vars: tp.Iterable[str] = ("tree",),
    mean_field_vars: tp.Iterable[str] = (),
) -> tp.Tuple[tfd.Distribution, tp.Dict[str, tf.Variable]]:
    """A hybrid approximation: the tree's root height and (by default) every
    non-tree model variable share one full-rank Gaussian block -- capturing the
    correlations between e.g. clock rate, population size and root height that
    a mean-field approximation misses -- while the tree's other (non-root)
    node-height ratios are approximated independently (mean-field). This
    targets the specific optimisation/approximation difficulty a full
    covariance over the whole (high-dimensional) set of node heights runs into,
    while keeping the correlations that matter most for the scalar model
    parameters.

    Requires a single fixed-topology tree variable, identified as the one name
    in ``tree_vars`` present in the model.

    Parameters
    ----------
    mean_field_vars
        Names of additional (non-tree) model variables to exclude from the
        full-covariance block and approximate independently. The full-rank
        block's parameter count is quadratic in its dimension, so variables
        whose dimension grows with the tree -- a per-branch relaxed clock rate,
        or the per-branch ``kappa`` of the carnivores lineage-variation model --
        make the full-rank block as expensive and as hard to fit as a
        whole-tree full-rank approximation. Naming them here keeps the block
        confined to the (few) parameters that are shared across the tree.
    """
    if joint_bijector_func is None:
        from treeflow.model.event_shape_bijector import get_default_event_space_bijector

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

    # `Restructure`/`Split` below route data through `tf.nest`, which flattens
    # dicts in *sorted-key* order -- NOT `base_event_shape`'s own iteration
    # order (`model._flat_resolve_names()`, which need not be alphabetical).
    # `full_rank.py` avoids this trap by building its flat vector via
    # `tf.nest.flatten` directly; we instead need an explicit name ordering
    # (to split the tree variable out), so it must match tf.nest's sorted
    # convention exactly, confirmed via `tf.nest.flatten({...})`.
    names = sorted(base_event_shape.keys())
    tree_names = [n for n in names if n in tree_vars]
    if len(tree_names) != 1:
        raise ValueError(
            "get_root_full_rank_approximation requires exactly one tree "
            f"variable among {list(tree_vars)}; found {tree_names} in model "
            f"variables {names}"
        )
    tree_name = tree_names[0]

    mean_field_names = [n for n in names if n in mean_field_vars]
    if tree_name in mean_field_names:
        raise ValueError(
            f"The tree variable {tree_name!r} cannot be listed in "
            "mean_field_vars; its non-root coordinates are always mean-field "
            "and its root is always in the full-rank block"
        )
    unknown = sorted(set(mean_field_vars) - set(names))
    if unknown:
        raise ValueError(
            f"mean_field_vars {unknown} are not model variables; model "
            f"variables are {names}"
        )
    full_rank_names = [
        n for n in names if n != tree_name and n not in mean_field_names
    ]

    # `NodeHeightRatioBijector`'s unconstrained representation is always
    # (taxon_count - 2) ratio coordinates followed by 1 root coordinate, so
    # the tree variable's total unconstrained size is enough to recover the
    # ratio/root split -- no need to look anything up by name in
    # `topology_pins` (whose keys are each tree distribution's own `tree_name`
    # attribute, not necessarily the same as the model's variable name used in
    # `base_event_shape`).
    tree_total_size = base_event_shape[tree_name].num_elements()
    if tree_total_size < 2:
        raise ValueError(
            f"Tree variable {tree_name!r} has only {tree_total_size} "
            "unconstrained coordinate(s); expected at least 2 (root plus at "
            "least one ratio)"
        )
    n_ratios = tree_total_size - 1

    sizes = {n: base_event_shape[n].num_elements() for n in names}
    sizes[_ROOT_KEY] = 1
    sizes[_RATIOS_KEY] = n_ratios

    # Unconstrained initial values, keyed the same way. The root is the *last*
    # unconstrained coordinate of the tree variable (see
    # `NodeHeightRatioBijector._inverse`), the ratios are everything before it.
    tree_init = init_loc_1d[tree_name]
    inits = dict(init_loc_1d)
    inits[_ROOT_KEY] = None if tree_init is None else tree_init[..., -1:]
    inits[_RATIOS_KEY] = None if tree_init is None else tree_init[..., :-1]

    # The two blocks, as ordered lists of the pieces each is built from.
    full_rank_keys = full_rank_names + [_ROOT_KEY]
    mean_field_keys = mean_field_names + [_RATIOS_KEY]
    full_rank_size = sum(sizes[k] for k in full_rank_keys)
    mean_field_size = sum(sizes[k] for k in mean_field_keys)
    total_dim = full_rank_size + mean_field_size

    def loc_init_for(keys):
        pieces = [
            tf.zeros(sizes[k], dtype=dtype)
            if inits[k] is None
            else tf.cast(tf.reshape(inits[k], [-1]), dtype)
            for k in keys
        ]
        return tf.concat(pieces, axis=0)

    # softplus_inverse(1) = log(exp(1) - 1) ~= 0.541, so the initial scale is 1
    # on the diagonal and 0 off it.
    softplus_inv1 = tf.cast(
        tf.math.log(tf.exp(tf.ones([], dtype=tf.float32)) - 1.0), dtype
    )

    # ---- Full-rank block: the shared model parameters plus the tree root ----
    fr_loc_var = tf.Variable(loc_init_for(full_rank_keys), name="root_full_rank_loc")
    fr_raw_var = tf.Variable(
        tf.linalg.diag(tf.fill([full_rank_size], softplus_inv1)),
        name="root_full_rank_scale_raw",
    )
    full_rank_bijector = _FullRankAffineBijector(
        fr_loc_var, fr_raw_var, name="RootFullRankAffine"
    )

    # ---- Mean-field block: the tree's ratios, plus any excluded variables ----
    mf_loc_var = tf.Variable(
        loc_init_for(mean_field_keys), name="root_full_rank_mean_field_loc"
    )
    mf_raw_var = tf.Variable(
        tf.fill([mean_field_size], softplus_inv1),
        name="root_full_rank_mean_field_scale_raw",
    )
    mean_field_bijector = _MeanFieldAffineBijector(mf_loc_var, mf_raw_var)

    blockwise_bijector = tfb.Blockwise(
        [full_rank_bijector, mean_field_bijector],
        block_sizes=[full_rank_size, mean_field_size],
    )

    # `blockwise_bijector` emits its pieces in block order (all of the
    # full-rank block, then all of the mean-field block). `Split` (below)
    # instead needs each variable's whole contiguous slice laid out in `names`
    # order, with the tree's slice being its ratios followed by its root
    # (`NodeHeightRatioBijector`'s convention). `Permute` reorders coordinates
    # to fix this up (`forward(x)[i] = x[permutation[i]]`, verified
    # empirically); it is volume-preserving, so it does not contribute to the
    # log-det-Jacobian.
    source_offsets = {}
    offset = 0
    for key in full_rank_keys + mean_field_keys:
        source_offsets[key] = offset
        offset += sizes[key]

    target_keys = []
    for n in names:
        if n == tree_name:
            target_keys += [_RATIOS_KEY, _ROOT_KEY]
        else:
            target_keys.append(n)
    permutation = [
        i
        for key in target_keys
        for i in range(source_offsets[key], source_offsets[key] + sizes[key])
    ]
    assert sorted(permutation) == list(range(total_dim))
    permute_bijector = tfb.Permute(
        permutation=tf.constant(permutation, dtype=tf.int32)
    )

    # Build the split+restructure chain (same pattern as full_rank.py) to go
    # from the single (now correctly ordered) flat vector to the dict of
    # per-variable unconstrained tensors.
    restructure_bijector = tfb.Restructure(
        output_structure=tf.nest.pack_sequence_as(
            base_event_shape, range(len(base_event_shape))
        ),
    )
    flat_sizes = restructure_bijector.inverse_event_shape(base_event_shape)
    flat_sizes_tensor = tf.cast(tf.concat(flat_sizes, axis=0), tf.int32)
    split_bijector = tfb.Split(flat_sizes_tensor)

    chain_bijector = tfb.Chain(
        [
            event_shape_and_space_bijector,
            restructure_bijector,
            split_bijector,
            permute_bijector,
            blockwise_bijector,
        ]
    )
    base_dist = tfd.Sample(
        tfd.Normal(
            tf.constant(0.0, dtype=dtype),
            tf.constant(1.0, dtype=dtype),
        ),
        total_dim,
    )
    distribution = tfd.TransformedDistribution(base_dist, chain_bijector)
    # Collected explicitly rather than from `distribution.trainable_variables`
    # so the returned dict is guaranteed complete regardless of how tf.Module's
    # attribute-based variable tracking traverses the bijector chain.
    created_variables = [fr_loc_var, fr_raw_var, mf_loc_var, mf_raw_var]
    variables_dict = {v.name: v for v in created_variables}
    return distribution, variables_dict


def get_fixed_topology_root_full_rank_approximation(
    model: tfd.JointDistribution,
    topology_pins: tp.Dict[str, TensorflowTreeTopology],
    init_loc=None,
    dtype=DEFAULT_FLOAT_DTYPE_TF,
    use_native="auto",
    unroll="auto",
    tree_vars: tp.Iterable[str] = ("tree",),
    mean_field_vars: tp.Iterable[str] = (),
) -> tp.Tuple[tfd.Distribution, tp.Dict[str, tf.Variable]]:
    bijector_func = partial(
        get_fixed_topology_joint_bijector,
        topology_pins=topology_pins,
        use_native=use_native,
        unroll=unroll,
    )
    event_shape_fn = partial(
        get_fixed_topology_event_shape, topology_pins=topology_pins
    )
    return get_root_full_rank_approximation(
        model,
        topology_pins=topology_pins,
        init_loc=init_loc,
        dtype=dtype,
        joint_bijector_func=bijector_func,
        event_shape_fn=event_shape_fn,
        tree_vars=tree_vars,
        mean_field_vars=mean_field_vars,
    )


__all__ = [
    "get_root_full_rank_approximation",
    "get_fixed_topology_root_full_rank_approximation",
]
