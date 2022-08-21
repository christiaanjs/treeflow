import typing as tp
from functools import partial
from collections import defaultdict
import tensorflow as tf
import tensorflow.python.util.nest as tf_nest
import tensorflow_probability.python.distributions as tfd
import tensorflow_probability.python.bijectors as tfb
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.distributions.joint_distribution import (
    _DefaultJointBijector,
)
from treeflow.bijectors.tree_ratio_bijector import TreeRatioBijector
from treeflow.bijectors.fixed_topology_bijector import FixedTopologyRootedTreeBijector
from treeflow.tree.topology.tensorflow_tree_topology import TensorflowTreeTopology


def inverse_with_nones(bijector: tfb.Bijector, val):
    if isinstance(bijector, tfb.Composition):
        return bijector._walk_inverse(inverse_with_nones, val)
    if val is None:
        return None
    else:
        return bijector.inverse(val)


def get_default_event_space_bijector(model: tfd.JointDistribution) -> tfb.Composition:
    return model.experimental_default_event_space_bijector()


def event_shape_fn(model: tfd.JointDistribution):
    return model.event_shape_tensor()


def get_event_shape_and_space_bijector(
    model: tfd.JointDistribution,
    joint_bijector_func: tp.Callable[
        [tfd.JointDistribution], tfb.Composition
    ] = get_default_event_space_bijector,
    event_shape_fn: tp.Callable[[tfd.JointDistribution], object] = event_shape_fn,
) -> tp.Tuple[tfb.Bijector, tp.Dict[str, tf.Tensor]]:
    event_space_bijector = joint_bijector_func(model)
    event_shape = event_shape_fn(model)
    names = model._flat_resolve_names()
    flat_event_shape = model._model_flatten(event_shape)
    flat_model_event_shape = model._model_flatten(model.event_shape)
    # Some bijectors (e.g. SoftmaxCentered) change event shape, but we need to handle trees
    flat_unconstrained_event_shape = [
        (
            bijector.inverse_event_shape(shape)
            if isinstance(model_shape, tf.TensorShape)
            else shape
        )
        for bijector, shape, model_shape in zip(
            event_space_bijector.bijectors, flat_event_shape, flat_model_event_shape
        )
    ]
    unconstrained_event_shape = dict(zip(names, flat_unconstrained_event_shape))
    flat_event_size = tf_nest.map_structure(tf.reduce_prod, unconstrained_event_shape)

    unflatten_bijector = tfb.Restructure(
        output_structure=model._model_unflatten(range(len(unconstrained_event_shape))),
        input_structure=dict(
            zip(
                names,
                range(len(names)),
            )
        ),
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
    chain_bijector = tfb.Chain(
        [
            event_space_bijector,
            unflatten_bijector,
            reshape_bijector,
        ]
    )
    return chain_bijector, base_event_shape


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
    model: tfd.JointDistribution, topology_pins: tp.Dict[str, TensorflowTreeTopology]
):
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


def get_fixed_topology_event_shape_and_space_bijector(
    model: tfd.JointDistribution,
    topology_pins: tp.Dict[str, TensorflowTreeTopology],
):
    bijector_func = partial(
        get_fixed_topology_joint_bijector, topology_pins=topology_pins
    )
    event_shape_fn = partial(
        get_fixed_topology_event_shape, topology_pins=topology_pins
    )
    return get_event_shape_and_space_bijector(
        model, joint_bijector_func=bijector_func, event_shape_fn=event_shape_fn
    )


def get_unconstrained_init_values(
    model: tfd.JointDistribution,
    event_shape_and_space_bijector: tfb.Bijector,
    event_shape_fn: tp.Callable[[tfd.JointDistribution], object] = event_shape_fn,
    init: tp.Optional[object] = None,
):
    event_shape = event_shape_fn(model)
    if init is None:
        init = tf_nest.map_structure(lambda _: None, event_shape)
    else:
        init = defaultdict(lambda: None, init)  # TODO: Handle nesting

    init_loc_1d = inverse_with_nones(event_shape_and_space_bijector, init)
    return init_loc_1d
