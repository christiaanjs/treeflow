import typing as tp
from collections import namedtuple
import numpy as np
import tensorflow as tf
from tensorflow_probability.python.math import MinimizeTraceableQuantities

VIResults = namedtuple("VIResults", ("loss", "parameters", "convergence_criterion_state"))
VIResults.__new__.__defaults__ = (None,)


def default_vi_trace_fn(
    traceable_quantities: MinimizeTraceableQuantities,
    variables_dict: tp.Dict[str, tf.Variable],
) -> VIResults:
    return VIResults(  # TODO: Name parameters
        loss=traceable_quantities.loss,
        parameters=variables_dict,
        convergence_criterion_state=traceable_quantities.convergence_criterion_state,
    )


def get_sampled_vi_trace_fn(
    variables_dict: tp.Dict[str, tf.Variable],
    max_trace_coords: int = 50,
    seed: tp.Optional[int] = None,
) -> tp.Callable[[MinimizeTraceableQuantities], VIResults]:
    """Build a trace_fn that records only up to ``max_trace_coords`` randomly
    selected flattened coordinates of each variable, instead of its full
    tensor, at every step.

    ``default_vi_trace_fn`` stores the entire value of every variable at
    every step, so its trace is O(num_steps * param_size). For approximations
    with large variables (e.g. a full-rank scale matrix, which is D x D in
    the number of free model dimensions D), that trace can dominate memory
    use and make long runs infeasible. Sampling a fixed set of coordinates
    once up front keeps the trace O(num_steps * max_trace_coords) regardless
    of how large any individual variable is, at the cost of a coarser trace
    for diagnostics (e.g. ``treeflow.vi.plotting.plot_parameter_traces``).
    Variables with at most ``max_trace_coords`` elements are traced in full.
    """
    rng = np.random.default_rng(seed)
    sample_indices: tp.Dict[str, tp.Optional[tf.Tensor]] = {}
    for name, variable in variables_dict.items():
        flat_size = int(np.prod(variable.shape))
        if flat_size <= max_trace_coords:
            sample_indices[name] = None
        else:
            sample_indices[name] = tf.constant(
                rng.choice(flat_size, size=max_trace_coords, replace=False),
                dtype=tf.int32,
            )

    def trace_fn(traceable_quantities: MinimizeTraceableQuantities) -> VIResults:
        parameters = {
            name: (
                tf.identity(variable)
                if sample_indices[name] is None
                else tf.gather(tf.reshape(variable, [-1]), sample_indices[name])
            )
            for name, variable in variables_dict.items()
        }
        return VIResults(
            loss=traceable_quantities.loss,
            parameters=parameters,
            convergence_criterion_state=traceable_quantities.convergence_criterion_state,
        )

    return trace_fn


__all__ = ["VIResults", "default_vi_trace_fn", "get_sampled_vi_trace_fn"]
