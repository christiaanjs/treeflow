from collections import namedtuple
import tensorflow as tf
from tensorflow_probability.python.optimizer.convergence_criteria import (
    ConvergenceCriterion,
)

_State = namedtuple(
    "RelativeLossNotDecreasingState",
    ["previous_loss", "average_decrease", "rel_rate", "consecutive_below"],
)


class RelativeLossNotDecreasing(ConvergenceCriterion):
    """Convergence criterion based on loss-normalised per-step improvement.

    Tracks an EWMA of the per-step decrease in loss (matching TFP's
    `LossNotDecreasing`) and declares convergence when that EWMA falls below
    a threshold that is expressed relative to the current |loss| value:

        decrease[t]  = loss[t-1] - loss[t]
        ewma[t]      = decrease[t] + decay * (ewma[t-1] - decrease[t])
        decay        = 1 - 1 / window_size
        converged    = ewma[t] / |loss[t]| < rtol   (and/or ewma[t] < atol)

    Differences from `LossNotDecreasing(rtol=...)`:
    - The denominator is the *current* |loss|, not the EWMA at step
      `window_size`.  This is invariant to where optimisation started
      (large initial loss → large initial EWMA, but the threshold scales
      with the current ELBO, not that initial burst).
    - It is also invariant to dataset scale: a model with 10× the sites has
      a 10× larger ELBO, so the same `rtol` threshold applies.

    NaN loss values are handled gracefully:
    - The per-step decrease is treated as zero improvement when either loss
      value is non-finite (so NaN steps don't permanently corrupt the EWMA).
    - The state (previous_loss, ewma) is frozen on NaN steps (not updated)
      so the criterion resumes correctly on the next finite loss.
    - Convergence is never declared on a NaN step.
    """

    def __init__(
        self,
        rtol=None,
        atol=None,
        window_size=100,
        min_num_steps=500,
        min_consecutive=10,
        name=None,
    ):
        """
        Args:
            rtol: convergence threshold relative to |loss|.  Stops when
                ``ewma / |loss| < rtol``.  At least one of ``rtol`` and
                ``atol`` must be given.
            atol: absolute convergence threshold.  Stops when ``ewma < atol``.
                Can be combined with ``rtol``; either condition suffices.
            window_size: EWMA time-constant (in steps).
            min_num_steps: minimum steps before convergence can be declared.
            min_consecutive: number of consecutive steps that must satisfy the
                convergence condition before stopping.  Guards against transient
                single-step dips in ``rel_rate`` causing spurious early stopping.
            name: optional name for TF ops.
        """
        if rtol is None and atol is None:
            raise ValueError("Must specify at least one of `rtol` and `atol`.")
        self._rtol = rtol
        self._atol = atol
        self._window_size = window_size
        self._min_consecutive = min_consecutive
        super().__init__(
            min_num_steps=min_num_steps,
            name=name or "RelativeLossNotDecreasing",
        )

    @property
    def rtol(self):
        return self._rtol

    @property
    def atol(self):
        return self._atol

    @property
    def window_size(self):
        return self._window_size

    @property
    def min_consecutive(self):
        return self._min_consecutive

    def _bootstrap(self, loss, grads, parameters):
        return _State(
            previous_loss=loss,
            average_decrease=tf.zeros_like(loss),
            rel_rate=tf.zeros_like(loss),
            consecutive_below=tf.zeros_like(loss, dtype=tf.int32),
        )

    def _one_step(self, step, loss, grads, parameters, auxiliary_state):
        previous_loss, ewma, _, consecutive_below = auxiliary_state
        decay = 1.0 - 1.0 / tf.cast(self._window_size, loss.dtype)

        decrease = previous_loss - loss
        # Replace non-finite differences with zero.  This covers both NaN loss
        # steps and the first finite step after a NaN (where previous_loss is
        # the pre-NaN finite value but the difference itself may still be NaN
        # if either side is non-finite).
        safe_decrease = tf.where(
            tf.math.is_finite(decrease), decrease, tf.zeros_like(decrease)
        )
        new_ewma = safe_decrease + decay * (ewma - safe_decrease)

        # Keep previous_loss frozen on NaN steps so the next finite step's
        # decrease is measured from the last good loss, not from NaN.
        # We do NOT freeze the EWMA: letting it decay via safe_decrease=0 is
        # preferable to holding it at whatever transient value it had when the
        # NaN cluster started (which could cause spurious early convergence).
        loss_is_nan = ~tf.math.is_finite(loss)
        new_previous_loss = tf.where(loss_is_nan, previous_loss, loss)

        # Always compute rel_rate for tracing, even when rtol is not used for
        # convergence checking — it is a useful diagnostic in the traced state.
        rel_rate = new_ewma / tf.abs(new_previous_loss)

        # Determine whether this step satisfies the convergence condition
        # (independently of min_consecutive — that check happens below).
        # A negative EWMA means the loss is worsening (not converging), so
        # rel_rate < rtol would spuriously fire — guard against it.
        step_satisfies = loss_is_nan & False  # False tensor, matching loss shape

        if self._rtol is not None:
            rtol = tf.cast(self._rtol, loss.dtype)
            step_satisfies = step_satisfies | (
                tf.math.is_finite(rel_rate)
                & (new_ewma >= tf.zeros_like(new_ewma))
                & (rel_rate < rtol)
            )

        if self._atol is not None:
            atol = tf.cast(self._atol, loss.dtype)
            step_satisfies = step_satisfies | (
                (new_ewma >= tf.zeros_like(new_ewma)) & (new_ewma < atol)
            )

        # Never count NaN steps toward consecutive threshold (they reset the
        # counter) — a NaN step should never push us closer to convergence.
        step_satisfies = step_satisfies & ~loss_is_nan

        # Increment counter when satisfied, reset to 0 otherwise.
        new_consecutive = tf.where(
            step_satisfies,
            consecutive_below + tf.ones_like(consecutive_below),
            tf.zeros_like(consecutive_below),
        )

        has_converged = new_consecutive >= tf.cast(
            self._min_consecutive, new_consecutive.dtype
        )

        return has_converged, _State(
            previous_loss=new_previous_loss,
            average_decrease=new_ewma,
            rel_rate=rel_rate,
            consecutive_below=new_consecutive,
        )


__all__ = ["RelativeLossNotDecreasing"]
