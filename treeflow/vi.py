import tensorflow_probability as tfp
from tensorflow_probability.python import math as tfp_math
from tensorflow_probability.python.vi import csiszar_divergence

from tensorflow_probability.python.internal import nest_util
import numpy as np
import tensorflow as tf
import tqdm


def minimize_eager(
    loss_fn, num_steps, optimizer, trainable_variables=None, name="minimize", iter=range
):
    state = {"loss": np.zeros(num_steps), "vars": {}}
    for i in iter(num_steps):
        try:
            with tf.GradientTape(
                watch_accessed_variables=trainable_variables is None
            ) as tape:
                for v in trainable_variables or []:
                    tape.watch(v)
                loss = loss_fn()

            watched_variables = tape.watched_variables()

            if i == 0:
                for variable in watched_variables:
                    state["vars"][variable.name] = np.zeros(
                        (num_steps + variable.shape).as_list()
                    )
            state["loss"][i] = loss.numpy()
            for variable in watched_variables:
                state["vars"][variable.name][i] = variable.numpy()

            grads = tape.gradient(loss, watched_variables)
            optimizer.apply_gradients(zip(grads, watched_variables))
        except KeyboardInterrupt:
            print("Exiting after {0} iterations".format(i))
            break
    return state


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
