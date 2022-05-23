from collections import namedtuple
from tensorflow_probability.python.math import MinimizeTraceableQuantities

VIResults = namedtuple("VIResults", ("loss", "parameters"))


def default_vi_trace_fn(traceable_quantities: MinimizeTraceableQuantities) -> VIResults:
    return VIResults(  # TODO: Name parameters
        loss=traceable_quantities.loss, parameters=traceable_quantities.parameters
    )


__all__ = ["VIResults", "default_vi_trace_fn"]
