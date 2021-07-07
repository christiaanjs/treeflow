from numpy.core.numeric import identity
import tensorflow_probability as tfp
from tensorflow_probability.python import math as tfp_math
from tensorflow_probability.python.math.minimize import (
    MinimizeTraceableQuantities,
    _trace_loss,
)
from tensorflow_probability.python.vi import csiszar_divergence
from tensorflow_probability.python.internal import nest_util
import numpy as np
import tensorflow as tf
import tqdm


def minimize_eager(
    loss_fn,
    num_steps,
    optimizer,
    trainable_variables=None,
    iter=range,
    convergence_criterion=None,
    trace_fn=_trace_loss,
    batch_convergence_reduce_fn=np.all,
):
    def convergence_detected(has_converged):
        return has_converged is not None and batch_convergence_reduce_fn(has_converged)

    def array_writer(arr, i, x):
        arr[i] = x.numpy()

    def optimizer_step(step):
        with tf.GradientTape(
            watch_accessed_variables=trainable_variables is None
        ) as tape:
            for v in trainable_variables or []:
                tape.watch(v)
            loss = loss_fn()
        watched_variables = tape.watched_variables()
        grads = tape.gradient(loss, watched_variables)
        optimizer.apply_gradients(zip(grads, watched_variables))
        parameters = [tf.identity(v) for v in watched_variables]
        return loss, grads, parameters

    step = 0
    initial_loss, initial_grads, initial_parameters = optimizer_step(step)
    has_converged = None
    initial_convergence_criterion_state = None
    if convergence_criterion is not None:
        has_converged = tf.zeros(tf.shape(initial_loss), dtype=tf.bool)
        initial_convergence_criterion_state = convergence_criterion.bootstrap(
            initial_loss, initial_grads, initial_parameters
        )
    initial_traced_values = trace_fn(
        MinimizeTraceableQuantities(
            loss=initial_loss,
            gradients=initial_grads,
            parameters=initial_parameters,
            step=step,
            has_converged=has_converged,
            convergence_criterion_state=initial_convergence_criterion_state,
        )
    )
    trace_arrays = tf.nest.map_structure(
        lambda t: np.array(shape=(num_steps,) + t.shape, dtype=t.dtype),
        initial_traced_values,
    )
    tf.nest.map_structure(
        lambda ta, x: array_writer(ta, step, x),
        trace_arrays,
        initial_traced_values,
    )

    for step in iter(1, num_steps):
        try:
            loss, grads, parameters = optimizer_step(step)
            if convergence_criterion is not None:
                (
                    has_converged,
                    convergence_criterion_state,
                ) = convergence_criterion.one_step(
                    step, loss, grads, parameters, convergence_criterion_state
                )
            traced_values = trace_fn(
                MinimizeTraceableQuantities(
                    loss=loss,
                    gradients=grads,
                    parameters=parameters,
                    step=step,
                    has_converged=has_converged,
                    convergence_criterion_state=initial_convergence_criterion_state,
                )
            )
            tf.nest.map_structure(
                lambda ta, x: array_writer(ta, step, x),
                trace_arrays,
                traced_values,
            )

            if convergence_detected(has_converged):
                break
        except KeyboardInterrupt:
            print("Exiting after {0} iterations".format(step))
            break
    return trace_arrays


def fit_surrogate_posterior(
    target_log_prob_fn,
    surrogate_posterior,
    optimizer,
    num_steps,
    seed=None,
    **min_kwargs
):
    def kl():
        q_samples = surrogate_posterior.sample(seed=seed)
        return surrogate_posterior.log_prob(q_samples) - nest_util.call_fn(
            target_log_prob_fn, q_samples
        )

    return minimize_eager(kl, num_steps=num_steps, optimizer=optimizer, **min_kwargs)


def _any_nonfinite(x):
    nonfinite = tf.logical_not(tf.math.is_finite(x))
    return tf.reduce_any(nonfinite)


class NonfiniteConvergenceCriterion(
    tfp.optimizer.convergence_criteria.ConvergenceCriterion
):
    def __init__(self, name="NonfiniteConvergenceCriterion"):
        super().__init__(min_num_steps=0, name=name)

    def _bootstrap(self, loss, grads, parameters):
        return None

    def _one_step(self, step, loss, grads, parameters, auxiliary_state):
        loss_nonfinite = _any_nonfinite(loss)
        grads_nonfinite = [_any_nonfinite(x) for x in grads]
        parameters_nonfinite = [_any_nonfinite(x) for x in parameters]
        has_converged = tf.reduce_any(
            [loss_nonfinite] + grads_nonfinite + parameters_nonfinite
        )
        return has_converged, None
