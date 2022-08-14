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
    _truncate_at_has_converged,
    _trace_has_converged,
)

from tensorflow_probability.python.optimizer.convergence_criteria import (
    ConvergenceCriterion,
)
from treeflow.tree.topology.tensorflow_tree_topology import TensorflowTreeTopology
from treeflow.model.event_shape_bijector import (
    get_fixed_topology_event_shape_and_space_bijector,
    get_fixed_topology_event_shape,
    get_unconstrained_init_values,
)

MLResults = namedtuple("MLResults", ["log_likelihood", "flat_unconstrained_params"])


def default_ml_trace_fn(traceable_quantities: MinimizeTraceableQuantities):
    return (-traceable_quantities.loss, traceable_quantities.parameters)


def fit_fixed_topology_maximum_likelihood_sgd(
    model: Distribution,
    topologies: tp.Dict[str, TensorflowTreeTopology],
    optimizer: Optimizer,
    num_steps: int,
    trace_fn: tp.Callable[[MinimizeTraceableQuantities], object] = default_ml_trace_fn,
    convergence_criterion: tp.Optional[ConvergenceCriterion] = None,
    init: tp.Optional[object] = None,
    return_full_length_trace: bool = True,
    **vi_kwargs
):
    bijector, base_event_shape = get_fixed_topology_event_shape_and_space_bijector(
        model, topologies
    )
    init_loc_1d = get_unconstrained_init_values(
        model,
        bijector,
        event_shape_fn=get_fixed_topology_event_shape,
        init=init,
    )
