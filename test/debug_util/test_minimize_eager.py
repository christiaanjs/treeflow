import tensorflow as tf
from tensorflow_probability.python.optimizer.convergence_criteria.loss_not_decreasing import (
    LossNotDecreasing,
)
import tensorflow as tf
from tensorflow_probability.python.math.minimize import minimize as tfp_minimize
from treeflow.debug.minimize_eager import minimize_eager
from treeflow_test_helpers.optimization_helpers import obj, vars, optimizer_builder
from numpy.testing import assert_allclose


def criterion_builder():
    return LossNotDecreasing(atol=1e-3, min_num_steps=5, window_size=3)


def test_minimize_eager_matches():
    optimizer = optimizer_builder()
    other_optimizer = optimizer_builder()

    _vars = vars()
    other_vars = vars("other_")

    _obj = obj(_vars)
    other_obj = obj(other_vars)

    trace_fn = lambda x: (x.step, x.loss, x.parameters)

    num_steps = 10
    eager_res = minimize_eager(_obj, num_steps, optimizer, trace_fn=trace_fn)
    other_res = tfp_minimize(other_obj, num_steps, other_optimizer, trace_fn=trace_fn)

    tf.nest.map_structure(assert_allclose, eager_res, other_res)


def test_minimize_eager_convergence():
    convergence_criterion = criterion_builder()
    trace_fn = lambda x: (x.loss, x.has_converged, x.convergence_criterion_state)
    num_steps = 1000

    optimizer = optimizer_builder()
    _vars = vars()
    _obj = obj(_vars)
    eager_res = minimize_eager(
        _obj,
        num_steps,
        optimizer,
        trace_fn=trace_fn,
        batch_convergence_reduce_fn=tf.reduce_any,
        convergence_criterion=convergence_criterion,
    )

    other_optimizer = optimizer_builder()
    other_vars = vars("other_")
    other_obj = obj(other_vars)
    other_res = tfp_minimize(
        other_obj,
        num_steps,
        other_optimizer,
        trace_fn=trace_fn,
        return_full_length_trace=False,
        batch_convergence_reduce_fn=tf.reduce_any,
        convergence_criterion=convergence_criterion,
    )

    tf.nest.map_structure(assert_allclose, eager_res, other_res)
