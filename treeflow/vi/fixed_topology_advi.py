from functools import partial
import typing as tp
import tensorflow as tf
from tensorflow.keras.optimizers import Optimizer
from tensorflow_probability.python.distributions import Distribution
from tensorflow_probability.python.vi import fit_surrogate_posterior
from tensorflow_probability.python.optimizer.convergence_criteria import (
    ConvergenceCriterion,
)
from tensorflow_probability.python.math import MinimizeTraceableQuantities
from tensorflow_probability.python.math.minimize import (
    _truncate_at_has_converged,
    _trace_has_converged,
)
from treeflow.tree.topology.tensorflow_tree_topology import TensorflowTreeTopology
from treeflow.model.approximation import get_fixed_topology_mean_field_approximation
from treeflow.vi.util import default_vi_trace_fn


def fit_fixed_topology_variational_approximation(
    model: Distribution,
    topologies: tp.Dict[str, TensorflowTreeTopology],
    optimizer: Optimizer,
    num_steps: int,
    trace_fn: tp.Optional[tp.Callable[[MinimizeTraceableQuantities], object]] = None,
    convergence_criterion: tp.Optional[ConvergenceCriterion] = None,
    init_loc: tp.Optional[object] = None,
    return_full_length_trace: bool = True,
    **vi_kwargs,
) -> tp.Tuple[Distribution, object]:
    approximation, variables_dict = get_fixed_topology_mean_field_approximation(
        model, init_loc=init_loc, topology_pins=topologies
    )

    if trace_fn is None:
        trace_fn = partial(default_vi_trace_fn, variables_dict=variables_dict)

    if return_full_length_trace:
        augmented_trace_fn = trace_fn
    else:
        augmented_trace_fn = _trace_has_converged(trace_fn, tf.reduce_all)

    trace = fit_surrogate_posterior(
        model.unnormalized_log_prob,
        approximation,
        optimizer,
        num_steps,
        convergence_criterion=convergence_criterion,
        trace_fn=augmented_trace_fn,
        **vi_kwargs,
    )

    if return_full_length_trace:
        opt_res = trace
    else:
        opt_res = _truncate_at_has_converged(trace)

    return (approximation, opt_res)


__all__ = ["fit_fixed_topology_variational_approximation"]
