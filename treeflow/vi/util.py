import typing as tp
from collections import namedtuple
import tensorflow as tf
from tensorflow_probability.python.math import MinimizeTraceableQuantities

VIResults = namedtuple("VIResults", ("loss", "parameters"))


def default_vi_trace_fn(
    traceable_quantities: MinimizeTraceableQuantities,
    variables_dict: tp.Dict[str, tf.Variable],
) -> VIResults:
    return VIResults(  # TODO: Name parameters
        loss=traceable_quantities.loss,
        parameters=variables_dict,
    )


__all__ = ["VIResults", "default_vi_trace_fn"]
