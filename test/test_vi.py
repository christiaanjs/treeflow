import tensorflow as tf
import pytest
import treeflow
import treeflow.vi
import tensorflow_probability as tfp
from numpy.testing import assert_allclose


def vars(prefix=""):
    return dict(
        a=tf.Variable(
            tf.convert_to_tensor(1.0, dtype=treeflow.DEFAULT_FLOAT_DTYPE_TF),
            name=f"{prefix}a",
        ),
        b=tf.Variable(
            tf.convert_to_tensor([1.2, 3.2], dtype=treeflow.DEFAULT_FLOAT_DTYPE_TF),
            name=f"{prefix}b",
        ),
    )


obs = tf.convert_to_tensor([0.8, 1.2], dtype=treeflow.DEFAULT_FLOAT_DTYPE_TF)


def obj(vars):
    return lambda: tf.math.square(vars["a"]) + tf.math.reduce_sum(
        tf.math.square(vars["a"] - vars["b"]) + tf.math.square(vars["b"] - obs)
    )


def optimizer_builder():
    return tf.optimizers.SGD(learning_rate=1e-2)


def criterion_builder():
    return tfp.optimizer.convergence_criteria.LossNotDecreasing(
        atol=1e-3, min_num_steps=5, window_size=3
    )


def test_minimize_eager_matches():
    optimizer = optimizer_builder()
    other_optimizer = optimizer_builder()

    _vars = vars()
    other_vars = vars("other_")

    _obj = obj(_vars)
    other_obj = obj(other_vars)

    trace_fn = lambda x: (x.step, x.loss, x.parameters)

    num_steps = 10
    eager_res = treeflow.vi.minimize_eager(
        _obj, num_steps, optimizer, trace_fn=trace_fn
    )
    other_res = tfp.math.minimize(
        other_obj, num_steps, other_optimizer, trace_fn=trace_fn
    )

    tf.nest.map_structure(assert_allclose, eager_res, other_res)


def test_minimize_eager_convergence():
    convergence_criterion = criterion_builder()
    trace_fn = lambda x: (x.loss, x.has_converged, x.convergence_criterion_state)
    num_steps = 1000

    optimizer = optimizer_builder()
    _vars = vars()
    _obj = obj(_vars)
    eager_res = treeflow.vi.minimize_eager(
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
    other_res = tfp.math.minimize(
        other_obj,
        num_steps,
        other_optimizer,
        trace_fn=trace_fn,
        return_full_length_trace=False,
        batch_convergence_reduce_fn=tf.reduce_any,
        convergence_criterion=convergence_criterion,
    )

    tf.nest.map_structure(assert_allclose, eager_res, other_res)
