import tensorflow as tf
from tensorflow_probability.python.optimizer.convergence_criteria import (
    ConvergenceCriterion,
)


def _any_nonfinite(x):
    nonfinite = tf.logical_not(tf.math.is_finite(x))
    return tf.reduce_any(nonfinite)


class NonfiniteConvergenceCriterion(ConvergenceCriterion):
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


__all__ = ["NonfiniteConvergenceCriterion"]
