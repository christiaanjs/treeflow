from collections import namedtuple
import tensorflow as tf
from tensorflow_probability.python.optimizer.convergence_criteria import (
    ConvergenceCriterion,
)

_State = namedtuple(
    "ParametersNotChangingState",
    ["previous_parameters", "average_change", "rel_change", "consecutive_below"],
)


def _global_rms(tensors, dtype):
    """Root-mean-square over every entry of a list of tensors.

    This is the global L2 norm divided by ``sqrt(d)`` (``d`` = total number of
    scalar entries), i.e. a *per-coordinate* scale.  Using the RMS rather than the
    raw L2 norm makes absolute thresholds independent of the number of
    parameters: adding more parameters of the same typical magnitude leaves the
    RMS unchanged.
    """
    if not tensors:
        return tf.zeros([], dtype=dtype)
    squared = [tf.reduce_sum(tf.square(tf.cast(t, dtype))) for t in tensors]
    count = tf.add_n([tf.cast(tf.size(t), dtype) for t in tensors])
    return tf.sqrt(tf.add_n(squared) / count)


class ParametersNotChanging(ConvergenceCriterion):
    """Convergence criterion based on how much the parameters are changing.

    Whereas :class:`RelativeLossNotDecreasing` watches the ELBO, this watches the
    optimisation variables themselves and declares convergence once they stop
    moving.  This is useful when the loss is noisy (stochastic ELBO estimates) but
    the underlying parameters have settled, or vice versa.

    At each step the per-coordinate root-mean-square (RMS) of the parameter
    update is computed across all parameters::

        change[t]   = rms(theta[t] - theta[t-1])
        ewma[t]     = change[t] + decay * (ewma[t-1] - change[t])
        decay       = 1 - 1 / window_size
        rel_change  = ewma[t] / rms(theta[t])
        converged   = rel_change < rtol   (and/or ewma[t] < atol)

    where ``rms(x) = ||x|| / sqrt(d)`` and ``d`` is the total number of scalar
    parameter entries.

    The change is smoothed with an EWMA (matching the loss criterion) so that the
    stochastic gradient noise in any single step does not trigger or prevent
    convergence.  Two scale invariances fall out of this definition:
    - ``rtol`` is invariant to the overall magnitude of the parameters (it divides
      by ``rms(theta)``).
    - ``atol`` is invariant to the *number* of parameters, because it tests a
      per-coordinate RMS rather than a raw L2 norm: a model with 10x the
      parameters (of similar magnitude) needs the same ``atol``.  This makes
      ``atol`` the natural choice when a parameter's optimum is at (or near) zero,
      where ``rms(theta) -> 0`` and the relative ``rtol`` test cannot settle.

    Note:
        All parameters are reduced into a single scalar change signal, so this
        criterion does not preserve batch dimensions (unlike the loss criterion).

    NaN/Inf parameter values are handled gracefully, mirroring
    :class:`RelativeLossNotDecreasing`:
    - The per-step change is treated as zero on a non-finite step (so the EWMA
      decays rather than being corrupted).
    - ``previous_parameters`` are frozen on a non-finite step so the next finite
      step measures its change from the last good values.
    - Convergence is never declared on a non-finite step.
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
            rtol: convergence threshold relative to ``rms(theta)``.  Stops when
                ``ewma / rms(theta) < rtol``.  At least one of ``rtol`` and
                ``atol`` must be given.
            atol: absolute convergence threshold on the (smoothed) per-coordinate
                RMS parameter update.  Stops when ``ewma < atol``.  Being a
                per-coordinate scale, a single ``atol`` (e.g. ``1e-4``) applies
                regardless of the number of parameters.  Can be combined with
                ``rtol``; either condition suffices.
            window_size: EWMA time-constant (in steps).
            min_num_steps: minimum steps before convergence can be declared.
            min_consecutive: number of consecutive steps that must satisfy the
                convergence condition before stopping.  Guards against transient
                single-step dips causing spurious early stopping.
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
            name=name or "ParametersNotChanging",
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
        dtype = loss.dtype
        return _State(
            previous_parameters=tuple(parameters),
            average_change=tf.zeros([], dtype=dtype),
            rel_change=tf.zeros([], dtype=dtype),
            consecutive_below=tf.zeros([], dtype=tf.int32),
        )

    def _one_step(self, step, loss, grads, parameters, auxiliary_state):
        previous_parameters, ewma, _, consecutive_below = auxiliary_state
        dtype = ewma.dtype
        decay = 1.0 - 1.0 / tf.cast(self._window_size, dtype)

        diffs = [p - q for p, q in zip(parameters, previous_parameters)]
        change = _global_rms(diffs, dtype)
        param_norm = _global_rms(list(parameters), dtype)

        # A non-finite change/parameter must not corrupt the EWMA or fire
        # convergence (this typically signals divergence, not convergence).
        change_nonfinite = ~tf.math.is_finite(change) | ~tf.math.is_finite(param_norm)
        safe_change = tf.where(change_nonfinite, tf.zeros_like(change), change)
        new_ewma = safe_change + decay * (ewma - safe_change)

        # Freeze the reference parameters on a non-finite step so the next finite
        # step's change is measured from the last good values.
        new_previous_parameters = tuple(
            tf.where(change_nonfinite, q, p)
            for p, q in zip(parameters, previous_parameters)
        )

        rel_change = new_ewma / param_norm

        # A change norm is always non-negative, so (unlike the loss decrease) no
        # sign guard is needed here.
        step_satisfies = change_nonfinite & False  # False scalar of matching shape

        if self._rtol is not None:
            rtol = tf.cast(self._rtol, dtype)
            step_satisfies = step_satisfies | (
                tf.math.is_finite(rel_change) & (rel_change < rtol)
            )

        if self._atol is not None:
            atol = tf.cast(self._atol, dtype)
            step_satisfies = step_satisfies | (new_ewma < atol)

        # Never count non-finite steps toward the consecutive threshold.
        step_satisfies = step_satisfies & ~change_nonfinite

        new_consecutive = tf.where(
            step_satisfies,
            consecutive_below + tf.ones_like(consecutive_below),
            tf.zeros_like(consecutive_below),
        )

        has_converged = new_consecutive >= tf.cast(
            self._min_consecutive, new_consecutive.dtype
        )

        return has_converged, _State(
            previous_parameters=new_previous_parameters,
            average_change=new_ewma,
            rel_change=rel_change,
            consecutive_below=new_consecutive,
        )


__all__ = ["ParametersNotChanging"]
