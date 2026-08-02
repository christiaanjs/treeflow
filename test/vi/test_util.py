from collections import namedtuple

import numpy as np
import pytest
import tensorflow as tf

from treeflow.vi.util import get_sampled_vi_trace_fn

_Quantities = namedtuple(
    "_Quantities", ("loss", "convergence_criterion_state")
)


def _variables():
    return {
        "tree_loc:0": tf.Variable(np.arange(10, dtype=np.float64)),
        "clock_rate_loc:0": tf.Variable(np.float64(1.0)),
    }


def test_sampled_trace_fn_records_traced_coordinates():
    variables = _variables()
    trace_fn = get_sampled_vi_trace_fn(variables, max_trace_coords=4, seed=1)

    coords = trace_fn.traced_coordinates
    assert set(coords) == set(variables)

    tree_coords = coords["tree_loc:0"]
    assert tree_coords.size == 10
    assert len(tree_coords.indices) == 4
    # Ascending, distinct, and always including the last coordinate (the root of
    # a node-height vector), so a subsampled trace can still be labelled by
    # variable coordinate.
    assert list(tree_coords.indices) == sorted(set(tree_coords.indices))
    assert tree_coords.indices[-1] == 9

    # Variables at or below the cap are traced in full.
    scalar_coords = coords["clock_rate_loc:0"]
    assert scalar_coords.size == 1
    assert list(scalar_coords.indices) == [0]


def test_sampled_trace_fn_traces_the_recorded_coordinates():
    variables = _variables()
    trace_fn = get_sampled_vi_trace_fn(variables, max_trace_coords=4, seed=1)
    indices = trace_fn.traced_coordinates["tree_loc:0"].indices

    results = trace_fn(_Quantities(loss=tf.constant(1.0), convergence_criterion_state=None))

    traced = np.asarray(results.parameters["tree_loc:0"])
    # The variable holds its own coordinate index as its value, so the traced
    # values are exactly the recorded indices.
    np.testing.assert_array_equal(traced, np.asarray(indices, dtype=traced.dtype))
    # Empty per step: the map is attached to the stacked results after the fit.
    assert results.parameter_coords == ()


@pytest.mark.parametrize("max_trace_coords", [2, 5, 9])
def test_sampled_trace_fn_respects_budget(max_trace_coords):
    trace_fn = get_sampled_vi_trace_fn(
        _variables(), max_trace_coords=max_trace_coords, seed=3
    )
    indices = trace_fn.traced_coordinates["tree_loc:0"].indices
    assert len(indices) == max_trace_coords
    assert indices[-1] == 9
    assert np.all(indices < 10)
