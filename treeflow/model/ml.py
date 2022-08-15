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

MLResults = namedtuple("MLResults", ["log_likelihood", "flat_unconstrained_params"])


def default_ml_trace_fn(traceable_quantities: MinimizeTraceableQuantities):
    return MLResults(-traceable_quantities.loss, traceable_quantities.parameters)


def fit_fixed_topology_maximum_likelihood_sgd(
    model: Distribution,
    topologies: tp.Dict[str, TensorflowTreeTopology],
    num_steps: int = 10000,
    optimizer: Optimizer = None,
    trace_fn: tp.Callable[[MinimizeTraceableQuantities], object] = default_ml_trace_fn,
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
    param_variables = [
        tf.Variable(init if init is not None else tf.zeros(shape, dtype=dtype))
        for (shape, init) in zip(base_event_shape, init_unconstrained)
    ]

    def negative_log_likelihood():
        variables = bijector.forward(param_variables)
        return -model.unnormalized_log_prob(variables)

    if convergence_criterion is None:
        convergence_criterion = LossNotDecreasing(atol=1e-3)
    if optimizer is None:
        optimizer = tf.optimizers.Adam()

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
