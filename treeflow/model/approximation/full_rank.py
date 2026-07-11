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


class _FullRankAffineBijector(tfb.Bijector):
    """Affine bijector L @ x + loc with constrained-positive diagonal of L.

    Stores a raw DxD matrix; the lower triangular part is extracted via
    ``tf.linalg.band_part`` and the diagonal is passed through ``softplus``
    to ensure positivity.  Upper-triangular elements receive zero gradient
    and are never updated.
    """

    def __init__(self, loc, raw_scale, name="FullRankAffine"):
        self.loc = loc
        self.raw_scale = raw_scale
        super().__init__(forward_min_event_ndims=1, name=name)

    def _get_scale_tril(self):
        L = tf.linalg.band_part(self.raw_scale, -1, 0)
        return tf.linalg.set_diag(L, tf.math.softplus(tf.linalg.diag_part(L)))

    def _forward(self, x):
        return self.loc + tf.linalg.matvec(self._get_scale_tril(), x)

    def _inverse(self, y):
        L = self._get_scale_tril()
        return tf.squeeze(
            tf.linalg.triangular_solve(L, tf.expand_dims(y - self.loc, -1)), -1
        )

    def _forward_log_det_jacobian(self, x):
        return tf.reduce_sum(
            tf.math.log(tf.linalg.diag_part(self._get_scale_tril()))
        )


def get_full_rank_approximation(
    model: tfd.JointDistribution,
    init_loc=None,
    dtype=DEFAULT_FLOAT_DTYPE_TF,
    joint_bijector_func: tp.Callable[
        [tfd.JointDistribution], tfb.Composition
    ] = None,
    event_shape_fn=None,
) -> tp.Tuple[tfd.Distribution, tp.Dict[str, tf.Variable]]:
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

    # Build split+restructure chain (same as IAF) to go flat-vector → dict
    restructure_bijector = tfb.Restructure(
        output_structure=tf.nest.pack_sequence_as(
            base_event_shape, range(len(base_event_shape))
        ),
    )
    flat_sizes = restructure_bijector.inverse_event_shape(base_event_shape)
    flat_sizes_tensor = tf.cast(tf.concat(flat_sizes, axis=0), tf.int32)
    split_bijector = tfb.Split(flat_sizes_tensor)
    total_dim = int(tf.reduce_sum(flat_sizes_tensor).numpy())

    # Concatenate init values in the same (alphabetical) key order as flat_sizes
    flat_inits = tf.nest.flatten(init_loc_1d)
    flat_sizes_list = tf.nest.flatten(
        restructure_bijector.inverse_event_shape(base_event_shape)
    )
    loc_pieces = [
        tf.zeros(sz, dtype=dtype) if val is None
        else tf.cast(tf.reshape(val, [-1]), dtype)
        for val, sz in zip(flat_inits, flat_sizes_list)
    ]
    loc_init = tf.concat(loc_pieces, axis=0)

    loc_var = tf.Variable(loc_init, name="full_rank_loc")

    # Initialise raw_scale so softplus(diagonal) ≈ 1 and off-diagonal = 0.
    # softplus_inverse(1) = log(exp(1) - 1) ≈ 0.541
    softplus_inv1 = tf.cast(
        tf.math.log(tf.exp(tf.ones([], dtype=tf.float32)) - 1.0), dtype
    )
    raw_scale_init = tf.linalg.diag(
        tf.fill([total_dim], softplus_inv1)
    )
    raw_var = tf.Variable(raw_scale_init, name="full_rank_scale_raw")

    full_rank_bijector = _FullRankAffineBijector(loc_var, raw_var)

    chain_bijector = tfb.Chain(
        [
            event_shape_and_space_bijector,
            restructure_bijector,
            split_bijector,
            full_rank_bijector,
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
    variables_dict = {v.name: v for v in distribution.trainable_variables}
    return distribution, variables_dict


def get_fixed_topology_full_rank_approximation(
    model: tfd.JointDistribution,
    topology_pins: tp.Dict[str, TensorflowTreeTopology],
    init_loc=None,
    dtype=DEFAULT_FLOAT_DTYPE_TF,
    use_native="auto",
    unroll="auto",
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
    return get_full_rank_approximation(
        model,
        init_loc=init_loc,
        dtype=dtype,
        joint_bijector_func=bijector_func,
        event_shape_fn=event_shape_fn,
    )


__all__ = [
    "get_full_rank_approximation",
    "get_fixed_topology_full_rank_approximation",
]
