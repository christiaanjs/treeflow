import typing as tp
from collections import namedtuple
from functools import partial
import tensorflow as tf
from tensorflow.keras.optimizers import Optimizer
import tensorflow.python.util.nest as tf_nest
from tensorflow_probability.python.distributions import Distribution
from tensorflow_probability.python.bijectors import Bijector
from tensorflow_probability.python.math import MinimizeTraceableQuantities
from tensorflow_probability.python.math.minimize import (
    minimize,
)

from tensorflow_probability.python.optimizer.convergence_criteria import (
    ConvergenceCriterion,
    LossNotDecreasing,
)
import treeflow
from treeflow.tree.topology.tensorflow_tree_topology import TensorflowTreeTopology
from treeflow.model.event_shape_bijector import (
    get_fixed_topology_event_shape_and_space_bijector,
    get_fixed_topology_event_shape,
    get_unconstrained_init_values,
)

MLResults = namedtuple("MLResults", ["log_likelihood", "unconstrained_params"])


def default_ml_trace_fn(
    traceable_quantities: MinimizeTraceableQuantities,
    variables_dict: tp.Dict[str, tf.Variable],
):
    return MLResults(-traceable_quantities.loss, variables_dict)


def fit_fixed_topology_maximum_likelihood_sgd(
    model: Distribution,
    topologies: tp.Dict[str, TensorflowTreeTopology],
    num_steps: int = 10000,
    optimizer: Optimizer = None,
    trace_fn: tp.Optional[tp.Callable[[MinimizeTraceableQuantities], object]] = None,
    convergence_criterion: tp.Optional[ConvergenceCriterion] = None,
    init: tp.Optional[object] = None,
    return_full_length_trace: bool = False,
    dtype=treeflow.DEFAULT_FLOAT_DTYPE_TF,
    **opt_kwargs,
):
    """
    Returns
    -------
    variables
        Maximum likelihood estimate for model variables
    trace
        Optimization trace
    bijector
        Bijector to transform unconstrained parameters to model variables
    """
    bijector, base_event_shape = get_fixed_topology_event_shape_and_space_bijector(
        model, topologies
    )
    init_unconstrained = get_unconstrained_init_values(
        model,
        bijector,
        event_shape_fn=partial(
            get_fixed_topology_event_shape, topology_pins=topologies
        ),
        init=init,
    )
    param_variables = {
        name: tf.Variable(
            init_unconstrained[name]
            if init_unconstrained[name] is not None
            else tf.zeros(shape, dtype=dtype),
            name=name,
        )
        for name, shape in base_event_shape.items()
    }

    def negative_log_likelihood():
        variables = bijector.forward(param_variables)
        return -model.unnormalized_log_prob(variables)

    if convergence_criterion is None:
        convergence_criterion = LossNotDecreasing(atol=1e-3)
    if optimizer is None:
        optimizer = tf.optimizers.Adam()

    if trace_fn is None:
        trace_fn = partial(default_ml_trace_fn, variables_dict=param_variables)

    trace = minimize(
        negative_log_likelihood,
        num_steps,
        optimizer,
        convergence_criterion=convergence_criterion,
        trace_fn=trace_fn,
        return_full_length_trace=return_full_length_trace,
        **opt_kwargs,
    )
    final_variables = bijector.forward(param_variables)
    return final_variables, trace, bijector


__all__ = ["fit_fixed_topology_maximum_likelihood_sgd"]
