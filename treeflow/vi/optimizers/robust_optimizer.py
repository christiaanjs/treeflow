import tensorflow as tf


class RobustOptimizer:  # Pseudo-optimizer
    def __init__(self, inner, max_retries=100):  # TODO: Count number of failed steps
        self.inner = inner
        self.max_retries = max_retries
        self.retries = tf.Variable(0)

    def apply_gradients(self, grads_and_vars, **kwargs):
        """Apply gradients to variables, skipping the step if any gradient
        contains NaN.

        When NaN gradients are detected, all gradients are zeroed so that
        the inner optimizer is still called with its expected variables
        (required by Keras 3) but no parameters are updated, preserving
        the original skip-step semantics.
        """
        grads_and_vars = list(grads_and_vars)
        grads = [grad for grad, var in grads_and_vars]
        vars_ = [var for grad, var in grads_and_vars]
        any_nan = tf.reduce_any([tf.reduce_any(tf.math.is_nan(x)) for x in grads])

        # Zero all gradients if any NaN — effectively skips the step
        safe_grads = [tf.where(any_nan, tf.zeros_like(g), g) for g in grads]

        # Track consecutive NaN steps (assert before incrementing,
        # matching original semantics)
        tf.debugging.assert_less(
            self.retries,
            self.max_retries,
            message="Too many consecutive NaN gradient steps",
        )
        self.retries.assign(tf.where(any_nan, self.retries + 1, 0))

        return self.inner.apply_gradients(zip(safe_grads, vars_), **kwargs)


__all__ = ["RobustOptimizer"]
