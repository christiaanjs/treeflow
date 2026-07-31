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


class _RatioMeanFieldBijector(tfb.Bijector):
    """Elementwise affine (shift + softplus-scale) transform for independent
    mean-field coordinates. Stores the raw variables directly (as
    ``_FullRankAffineBijector`` does) and recomputes the transform on every
    call, rather than passing a precomputed ``tf.math.softplus(raw_scale)``
    tensor into ``tfb.Scale`` -- which freezes that tensor's value at
    bijector-construction time and breaks gradient flow to ``raw_scale``.
    """

    def __init__(self, loc, raw_scale, name="RatioMeanField"):
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
    joint_bijector_func: tp.Callable[
        [tfd.JointDistribution], tfb.Composition
    ] = None,
    event_shape_fn=None,
    tree_vars: tp.Iterable[str] = ("tree",),
) -> tp.Tuple[tfd.Distribution, tp.Dict[str, tf.Variable]]:
    """A hybrid approximation: the tree's root height and every non-tree model
    variable share one full-rank Gaussian block (so the correlations between
    e.g. clock rate, population size and root height that a mean-field
    approximation misses can be captured), while the tree's other
    (non-root) node-height ratios are approximated independently
    (mean-field). This targets the specific optimisation/approximation
    difficulty a full covariance over the whole (high-dimensional) set of
    node heights can run into, while keeping the correlations that matter
    most for the scalar model parameters.

    Requires a single fixed-topology tree variable, identified as the one
    name in ``tree_vars`` present in the model.
    """
    if joint_bijector_func is None:
        from treeflow.model.event_shape_bijector import get_default_event_space_bijector

        joint_bijector_func = get_default_event_space_bijector
    if event_shape_fn is None:
        event_shape_fn = default_event_shape_fn

    event_shape_and_space_bijector, base_event_shape = get_event_shape_and_space_bijector(
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
    tree_position = names.index(tree_name)
    other_names = [n for n in names if n != tree_name]
    other_sizes = [base_event_shape[n].num_elements() for n in other_names]

    # `NodeHeightRatioBijector`'s unconstrained representation is always
    # (taxon_count - 2) ratio coordinates followed by 1 root coordinate, so
    # the tree variable's total unconstrained size (already computed above)
    # is enough to recover the ratio/root split -- no need to look anything
    # up by name in `topology_pins` (whose keys are each tree distribution's
    # own `tree_name` attribute, not necessarily the same as the model's
    # variable name used in `base_event_shape`).
    tree_total_size = base_event_shape[tree_name].num_elements()
    if tree_total_size < 2:
        raise ValueError(
            f"Tree variable {tree_name!r} has only {tree_total_size} "
            "unconstrained coordinate(s); expected at least 2 (root plus at "
            "least one ratio)"
        )
    n_ratios = tree_total_size - 1

    full_rank_size = sum(other_sizes) + 1  # non-tree variables + the tree root
    total_dim = full_rank_size + n_ratios

    # ---- Full-rank block: every non-tree variable, plus the tree root ----
    # `NodeHeightRatioBijector` puts the root as the *last* unconstrained
    # coordinate of the tree variable (see its `_inverse`), so this is
    # `init_loc_1d[tree_name][..., -1:]`.
    other_flat_inits = [init_loc_1d[n] for n in other_names]
    root_init = None if init_loc_1d[tree_name] is None else init_loc_1d[tree_name][..., -1:]
    fr_loc_pieces = [
        tf.zeros(sz, dtype=dtype) if val is None else tf.cast(tf.reshape(val, [-1]), dtype)
        for val, sz in zip(other_flat_inits + [root_init], other_sizes + [1])
    ]
    fr_loc_init = tf.concat(fr_loc_pieces, axis=0)
    fr_loc_var = tf.Variable(fr_loc_init, name="root_full_rank_loc")

    softplus_inv1 = tf.cast(
        tf.math.log(tf.exp(tf.ones([], dtype=tf.float32)) - 1.0), dtype
    )
    fr_raw_scale_init = tf.linalg.diag(tf.fill([full_rank_size], softplus_inv1))
    fr_raw_var = tf.Variable(fr_raw_scale_init, name="root_full_rank_scale_raw")
    full_rank_bijector = _FullRankAffineBijector(
        fr_loc_var, fr_raw_var, name="RootFullRankAffine"
    )

    # ---- Mean-field block: the tree's other (ratio) coordinates ----
    ratio_init = None if init_loc_1d[tree_name] is None else init_loc_1d[tree_name][..., :-1]
    ratio_loc_init = (
        tf.zeros(n_ratios, dtype=dtype)
        if ratio_init is None
        else tf.cast(tf.reshape(ratio_init, [-1]), dtype)
    )
    ratio_loc_var = tf.Variable(ratio_loc_init, name="root_full_rank_ratio_loc")
    ratio_raw_scale_var = tf.Variable(
        tf.fill([n_ratios], softplus_inv1), name="root_full_rank_ratio_scale_raw"
    )
    ratio_bijector = _RatioMeanFieldBijector(ratio_loc_var, ratio_raw_scale_var)

    blockwise_bijector = tfb.Blockwise(
        [full_rank_bijector, ratio_bijector], block_sizes=[full_rank_size, n_ratios]
    )

    # `blockwise_bijector` outputs, in order: [other_names[0], ..., other_names[-1],
    # tree_root, tree_ratios]. `Split` (below) instead needs, in `names` order,
    # each variable's whole contiguous slice -- i.e. the tree's slice
    # (ratios then root, matching NodeHeightRatioBijector's convention) at
    # `tree_position`, not at the end. `Permute` reorders coordinates
    # (forward(x)[i] = x[permutation[i]], verified empirically) to fix this up;
    # it is volume-preserving, so it does not affect the log-det-Jacobian.
    offset_before_tree = sum(base_event_shape[n].num_elements() for n in names[:tree_position])
    S = full_rank_size - 1
    permutation = []
    for i in range(total_dim):
        if i < offset_before_tree:
            permutation.append(i)
        elif i < offset_before_tree + n_ratios:
            permutation.append(S + 1 + (i - offset_before_tree))
        elif i == offset_before_tree + n_ratios:
            permutation.append(S)
        else:
            permutation.append(i - n_ratios - 1)
    permute_bijector = tfb.Permute(permutation=tf.constant(permutation, dtype=tf.int32))

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
    # Built explicitly (rather than from `distribution.trainable_variables`):
    # `tfb.Scale(tf.math.softplus(ratio_raw_scale_var))` passes a *computed*
    # tensor as the bijector's `scale` parameter rather than the variable
    # itself, so `ratio_raw_scale_var` is not discoverable via tf.Module's
    # attribute-based variable tracking (unlike `_FullRankAffineBijector`,
    # which stores `loc`/`raw_scale` as direct attributes). It still receives
    # gradients correctly -- this only affects which variables get collected.
    created_variables = [
        fr_loc_var,
        fr_raw_var,
        ratio_loc_var,
        ratio_raw_scale_var,
    ]
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
    )


__all__ = [
    "get_root_full_rank_approximation",
    "get_fixed_topology_root_full_rank_approximation",
]
