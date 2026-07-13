import typing as tp
from collections import namedtuple
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


__all__ = ["VIResults", "default_vi_trace_fn"]
