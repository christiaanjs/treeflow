from collections import defaultdict
from functools import partial
import typing as tp
from numpy import reshape
import tensorflow as tf
import tensorflow_probability.python.distributions as tfd
import tensorflow_probability.python.bijectors as tfb
from tensorflow_probability.python.experimental.vi.util import (
    build_trainable_linear_operator_block,
)
from tensorflow_probability.python.distributions.joint_distribution import (
    _DefaultJointBijector,
)
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.experimental.util import make_trainable
from treeflow import DEFAULT_FLOAT_DTYPE_TF
from treeflow.tree.topology.tensorflow_tree_topology import TensorflowTreeTopology
from treeflow.bijectors.tree_ratio_bijector import TreeRatioBijector
from treeflow.bijectors.fixed_topology_bijector import FixedTopologyRootedTreeBijector
from treeflow.distributions.tree.base_tree_distribution import BaseTreeDistribution
import tensorflow.python.util.nest as tf_nest


def get_base_distribution(flat_event_size, dtype=DEFAULT_FLOAT_DTYPE_TF):
    base_standard_dist = tfd.JointDistributionSequential(
        [
            tfd.Sample(
                tfd.Normal(
                    loc=tf.constant(0.0, dtype=dtype),
                    scale=tf.constant(1.0, dtype=dtype),
                ),
                s,
            )
            for s in flat_event_size
        ]
    )
    return base_standard_dist


def get_mean_field_operator_classes(flat_event_size):
    return tuple(
        [tf.linalg.LinearOperatorDiag for _ in flat_event_size]
    )  # TODO: Bijector?


def get_trainable_shift_bijector(
    flat_event_size, init_loc_unconstrained, dtype=DEFAULT_FLOAT_DTYPE_TF
):
    return tfb.JointMap(
        tf.nest.map_structure(
            lambda s, init: tfb.Shift(
                tf.Variable(
                    tf.random.uniform((s,), minval=-2.0, maxval=2.0, dtype=dtype)
                    if init is None
                    else init
                )
            ),
            flat_event_size,
            init_loc_unconstrained,
        )
    )


def inverse_with_nones(bijector: tfb.Bijector, val):
    if val is None:
        return None
    else:
        return bijector.inverse(val)


def joint_inverse_with_nones(bijector: tfb.Composition, val):
    return bijector._walk_inverse(inverse_with_nones, val)


def get_default_event_space_bijector(model: tfd.JointDistribution) -> tfb.Composition:
    return model.experimental_default_event_space_bijector()


def event_shape_fn(model: tfd.JointDistribution):
    return model.event_shape_tensor()


def get_mean_field_approximation(
    model: tfd.JointDistribution,
    init_loc=None,
    dtype=DEFAULT_FLOAT_DTYPE_TF,
    joint_bijector_func: tp.Callable[
        [tfd.JointDistribution], tfb.Composition
    ] = get_default_event_space_bijector,
    event_shape_fn: tp.Callable[[tfd.JointDistribution], object] = event_shape_fn,
) -> tfd.Distribution:
    event_space_bijector = joint_bijector_func(model)
    event_shape = event_shape_fn(model)
    flat_event_shape = model._model_flatten(event_shape)
    flat_model_event_shape = model._model_flatten(model.event_shape)
    # Some bijectors (e.g. SoftmaxCentered) change event shape, but we need to handle trees
    unconstrained_event_shape = [
        (
            bijector.inverse_event_shape(shape)
            if isinstance(model_shape, tf.TensorShape)
            else shape
        )
        for bijector, shape, model_shape in zip(
            event_space_bijector.bijectors, flat_event_shape, flat_model_event_shape
        )
    ]
    flat_event_size = tf_nest.map_structure(tf.reduce_prod, unconstrained_event_shape)
    # operator_classes = get_mean_field_operator_classes(flat_event_size)
    # linear_operator_block = build_trainable_linear_operator_block(
    #     operator_classes, flat_event_size, dtype=dtype
    # )
    # scale_bijector = tfb.ScaleMatvecLinearOperatorBlock(linear_operator_block)

    if init_loc is None:
        init_loc = tf_nest.map_structure(lambda _: None, event_shape)
    else:
        init_loc = defaultdict(lambda: None, init_loc)  # TODO: Handle nesting

    unflatten_bijector = tfb.Restructure(
        model._model_unflatten(range(len(unconstrained_event_shape)))
    )
    reshape_bijector = tfb.JointMap(
        tf_nest.map_structure(
            lambda event_shape_element, event_size_element: tfb.Reshape(
                event_shape_element, tf.expand_dims(event_size_element, 0)
            )
            if tensorshape_util.rank(event_shape_element) > 0
            else tfb.Identity(),
            unconstrained_event_shape,
            flat_event_size,
        )
    )
    base_event_shape = reshape_bijector.inverse_event_shape(unconstrained_event_shape)

    init_loc_unconstrained = joint_inverse_with_nones(event_space_bijector, init_loc)
    init_loc_flat = unflatten_bijector.inverse(init_loc_unconstrained)
    init_loc_1d = joint_inverse_with_nones(reshape_bijector, init_loc_flat)
    # loc_bijector = get_trainable_shift_bijector(
    #     flat_event_size, init_loc_1d, dtype=dtype
    # )

    # base_standard_dist = get_base_distribution(flat_event_size, dtype=dtype)
    base_standard_dist = tfd.JointDistributionSequential(
        tf.nest.map_structure(
            lambda s, init: tfd.Independent(
                make_trainable(
                    tfd.Normal,
                    initial_parameters=(None if init is None else dict(loc=init)),
                    batch_and_event_shape=s,
                    parameter_dtype=dtype,
                ),
                reinterpreted_batch_ndims=tensorshape_util.rank(s),
            ),
            base_event_shape,
            init_loc_1d,
        )
    )

    chain_bijector = tfb.Chain(
        [
            event_space_bijector,
            unflatten_bijector,
            reshape_bijector,
            # loc_bijector,
            # scale_bijector,
        ]
    )
    distribution = tfd.TransformedDistribution(base_standard_dist, chain_bijector)
    return distribution


def get_fixed_topology_bijector(
    dist: tfd.Distribution, topology_pins=tp.Dict[str, TensorflowTreeTopology]
):
    if hasattr(dist, "tree_name") and getattr(dist, "tree_name") in topology_pins:
        topology = topology_pins[dist.tree_name]
        tree_bijector: TreeRatioBijector = (
            dist.experimental_default_event_space_bijector(topology=topology)
        )
        return FixedTopologyRootedTreeBijector(
            topology,
            tree_bijector.bijectors.node_heights,
            sampling_times=dist.sampling_times,  # TODO: Make sure dist has fixed sampling times
        )
    else:
        return dist.experimental_default_event_space_bijector()


def get_fixed_topology_joint_bijector(
    model: tfd.JointDistribution, topology_pins=tp.Dict[str, TensorflowTreeTopology]
) -> tfb.Composition:
    bijector_fn = partial(get_fixed_topology_bijector, topology_pins=topology_pins)
    return _DefaultJointBijector(model, bijector_fn=bijector_fn)


def get_fixed_topology_event_shape(
    model: tfd.JointDistribution, topology_pins=tp.Dict[str, TensorflowTreeTopology]
):
    pinned_topologies = set(topology_pins.keys())
    single_sample_distributions = model._get_single_sample_distributions()
    event_shape_tensors = model._model_flatten(model.event_shape_tensor())
    pinned_event_shape_tensors = [
        (
            event_shape_tensor.node_heights
            if hasattr(dist, "tree_name")
            and getattr(dist, "tree_name") in topology_pins
            else event_shape_tensor
        )
        for event_shape_tensor, dist in zip(
            event_shape_tensors, single_sample_distributions
        )
    ]
    return model._model_unflatten(pinned_event_shape_tensors)


def get_fixed_topology_mean_field_approximation(
    model: tfd.JointDistribution,
    topology_pins: tp.Dict[str, TensorflowTreeTopology],
    init_loc=None,
    dtype=DEFAULT_FLOAT_DTYPE_TF,
) -> tfd.Distribution:
    bijector_func = partial(
        get_fixed_topology_joint_bijector, topology_pins=topology_pins
    )
    event_shape_fn = partial(
        get_fixed_topology_event_shape, topology_pins=topology_pins
    )
    return get_mean_field_approximation(
        model,
        init_loc=init_loc,
        dtype=dtype,
        joint_bijector_func=bijector_func,
        event_shape_fn=event_shape_fn,
    )
