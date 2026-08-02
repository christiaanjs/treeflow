import typing as tp
from collections import namedtuple
import numpy as np
import tensorflow as tf
from tensorflow_probability.python.math import MinimizeTraceableQuantities

VIResults = namedtuple(
    "VIResults",
    ("loss", "parameters", "convergence_criterion_state", "parameter_coords"),
)
# `parameter_coords` is attached to the stacked results after optimisation
# rather than traced per step, so it defaults to the empty structure `()` (as
# `convergence_criterion_state` does when there is no criterion): `None` is not
# a traceable leaf, and a per-step value would be stacked `num_steps` times.
VIResults.__new__.__defaults__ = (None, ())

TracedCoordinates = namedtuple("TracedCoordinates", ("indices", "size"))
TracedCoordinates.__doc__ = """Which coordinates of a variable a trace holds.

``indices`` is an ascending array of the variable's flattened coordinate
indices that were recorded, and ``size`` its full flattened size. A trace
recorded by ``get_sampled_vi_trace_fn`` holds only a subset of each large
variable's coordinates, so positions along the trailing axis of
``VIResults.parameters[name]`` are *not* coordinate indices; ``indices`` maps
them back. ``VIResults.parameter_coords`` is empty for a full trace, where the
two coincide.
"""


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

    Coordinates are sampled in ascending order, and the last coordinate of a
    variable is always included: for a node-height vector that is the root, the
    single most interesting coordinate to diagnose. Which coordinates were kept
    is recorded on the returned function as a ``traced_coordinates`` attribute
    (``name -> TracedCoordinates``), so that plots can label coordinates by
    their index in the variable rather than their position in the trace.
    """
    rng = np.random.default_rng(seed)
    sample_indices: tp.Dict[str, tp.Optional[tf.Tensor]] = {}
    traced_coordinates: tp.Dict[str, TracedCoordinates] = {}
    for name, variable in variables_dict.items():
        flat_size = int(np.prod(variable.shape))
        if flat_size <= max_trace_coords:
            sample_indices[name] = None
            indices = np.arange(flat_size)
        else:
            # Always keep the last coordinate (the root, for a node-height
            # vector); sample the rest of the budget from the others.
            indices = np.sort(
                np.append(
                    rng.choice(flat_size - 1, size=max_trace_coords - 1, replace=False),
                    flat_size - 1,
                )
            )
            sample_indices[name] = tf.constant(indices, dtype=tf.int32)
        traced_coordinates[name] = TracedCoordinates(indices=indices, size=flat_size)

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

    trace_fn.traced_coordinates = traced_coordinates
    return trace_fn


__all__ = [
    "VIResults",
    "TracedCoordinates",
    "default_vi_trace_fn",
    "get_sampled_vi_trace_fn",
]
