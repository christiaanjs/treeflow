from functools import partial
import typing as tp
from typing_extensions import Protocol
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
from treeflow.vi.progress_bar import make_progress_bar_trace_fn, ProgressBarFunc


class ApproximationBuilder(Protocol):
    def __call__(
        self,
        model: Distribution,
        topology_pins: tp.Dict[str, TensorflowTreeTopology],
        init_loc: tp.Optional[object] = None,
        **kwargs,
    ) -> tp.Tuple[Distribution, tp.Dict[str, tf.Variable]]:
        ...


def fit_fixed_topology_variational_approximation(
    model: Distribution,
    topologies: tp.Dict[str, TensorflowTreeTopology],
    optimizer: Optimizer,
    num_steps: int,
    trace_fn: tp.Optional[tp.Callable[[MinimizeTraceableQuantities], object]] = None,
    convergence_criterion: tp.Optional[ConvergenceCriterion] = None,
    init_loc: tp.Optional[object] = None,
    return_full_length_trace: bool = True,
    progress_bar: tp.Union[bool, ProgressBarFunc] = False,
    progress_bar_step: int = 10,
    approx_fn: ApproximationBuilder = get_fixed_topology_mean_field_approximation,
    approx_kwargs: tp.Optional[tp.Dict[str, object]] = None,
    **vi_kwargs,
) -> tp.Tuple[Distribution, object]:
    if approx_kwargs is None:
        approx_kwargs = {}

    approximation, variables_dict = approx_fn(
        model, init_loc=init_loc, topology_pins=topologies, **approx_kwargs
    )

    if trace_fn is None:
        trace_fn = partial(default_vi_trace_fn, variables_dict=variables_dict)

    if return_full_length_trace:
        augmented_trace_fn = trace_fn
    else:
        augmented_trace_fn = _trace_has_converged(trace_fn, tf.reduce_all)

    with make_progress_bar_trace_fn(
        augmented_trace_fn, num_steps, progress_bar, update_step=progress_bar_step
    ) as progress_trace_fn:
        trace = fit_surrogate_posterior(
            model.unnormalized_log_prob,
            approximation,
            optimizer,
            num_steps,
            convergence_criterion=convergence_criterion,
            trace_fn=progress_trace_fn,
            **vi_kwargs,
        )

    if return_full_length_trace:
        opt_res = trace
    else:
        opt_res = _truncate_at_has_converged(trace)

    return (approximation, opt_res)


__all__ = ["fit_fixed_topology_variational_approximation"]
