from tensorflow_probability.python.math.minimize import (
    MinimizeTraceableQuantities,
    _trace_loss,
)
from tensorflow_probability.python.vi import csiszar_divergence
import numpy as np
import tensorflow as tf
import functools


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
            step=tf.convert_to_tensor(step),
            has_converged=has_converged,
            convergence_criterion_state=initial_convergence_criterion_state,
        )
    )
    trace_arrays = tf.nest.map_structure(
        lambda t: np.zeros((num_steps,) + t.numpy().shape, dtype=t.numpy().dtype),
        initial_traced_values,
    )
    tf.nest.map_structure(
        lambda ta, x: array_writer(ta, step, x),
        trace_arrays,
        initial_traced_values,
    )

    convergence_criterion_state = initial_convergence_criterion_state
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
                    step=tf.convert_to_tensor(step),
                    has_converged=has_converged,
                    convergence_criterion_state=convergence_criterion_state,
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
    return tf.nest.map_structure(lambda ta: ta[: step + 1], trace_arrays)


def fit_surrogate_posterior(
    target_log_prob_fn,
    surrogate_posterior,
    optimizer,
    num_steps,
    variational_loss_fn=None,
    discrepancy_fn=csiszar_divergence.kl_reverse,
    sample_size=1,
    importance_sample_size=1,
    seed=None,
    **opt_kwargs
):
    if variational_loss_fn is None:
        variational_loss_fn = functools.partial(
            csiszar_divergence.monte_carlo_variational_loss,
            discrepancy_fn=discrepancy_fn,
            importance_sample_size=importance_sample_size,
            # Silent fallback to score-function gradients leads to
            # difficult-to-debug failures, so force reparameterization gradients by
            # default.
            gradient_estimator=(
                csiszar_divergence.GradientEstimators.REPARAMETERIZATION),
            )

    def complete_variational_loss_fn():
        return variational_loss_fn(
            target_log_prob_fn, surrogate_posterior, sample_size=sample_size, seed=seed
        )

    return minimize_eager(
        complete_variational_loss_fn,
        num_steps=num_steps,
        optimizer=optimizer,
        **opt_kwargs
    )
