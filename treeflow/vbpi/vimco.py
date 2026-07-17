"""VIMCO multi-sample gradient estimator for VBPI.

VBPI maximises a ``K``-sample importance-weighted (IWAE) lower bound::

    L_hat = logmeanexp_k  log w_k ,   w_k = p(x, T_k, b_k) / q(T_k, b_k)

The branch lengths ``b_k`` are reparameterised, so their gradient flows through
``log w_k`` directly. The topology ``T_k`` is a *discrete* sample, so its
parameters get no pathwise gradient; VIMCO (Mnih & Rezende, 2016) supplies a
score-function estimator with a per-sample leave-one-out control variate that
sharply reduces variance relative to plain REINFORCE.

The estimator combines two pieces:

* a **pathwise** part -- the gradient of ``L_hat`` itself, which carries the
  reparameterised branch-length / model gradients (weighting each sample by its
  normalised importance weight ``w~_k``). This part also produces, for the
  topology parameters, the gradient of the ``-log q(T_k)`` term that appears
  *explicitly* inside ``log w_k``;
* a **score** part for the topology parameters,
  ``sum_k (L_hat - L_hat_{-k}) * grad log q(T_k)``, where ``L_hat_{-k}`` is
  ``L_hat`` with the ``k``-th log-weight replaced by the (log-space) mean of the
  other ``K-1`` -- the VIMCO control variate for the *sampling* dependence.

For the topology parameters the two pieces combine to the familiar VIMCO
multiplier ``(L_hat - L_hat_{-k} - w~_k)`` on ``grad log q(T_k)``: the ``-w~_k``
is precisely the pathwise gradient of the explicit ``-log q(T_k)`` term. Because
of this, ``log_weights`` must be passed **fully attached** (do *not* detach the
discrete ``log q(T_k)`` inside it); :func:`vimco_surrogate` returns a
differentiable surrogate whose gradient is the full VIMCO estimator, together
with the (equal) reported bound value.
"""
import typing as tp

import tensorflow as tf


class VimcoObjective(tp.NamedTuple):
    """Result of :func:`vimco_surrogate`.

    Attributes
    ----------
    elbo
        The reported ``K``-sample bound value ``L_hat`` (``[...]``); take its
        mean over any batch dimensions to monitor training.
    surrogate
        A differentiable scalar-per-batch quantity ``[...]`` whose gradient is
        the VIMCO estimator. Maximise it (equivalently minimise ``-surrogate``)
        to fit the variational parameters. Its *value* equals ``elbo``.
    """

    elbo: tf.Tensor
    surrogate: tf.Tensor


def reduce_logmeanexp(x: tf.Tensor, axis: int = -1) -> tf.Tensor:
    """``log( mean( exp(x) ) )`` along ``axis`` (numerically stable)."""
    k = tf.cast(tf.shape(x)[axis], x.dtype)
    return tf.reduce_logsumexp(x, axis=axis) - tf.math.log(k)


def _leave_one_out_bounds(log_weights: tf.Tensor) -> tf.Tensor:
    """``L_hat_{-k}`` for every ``k`` along the last (sample) axis.

    ``L_hat_{-k}`` is ``logmeanexp`` of the log-weight vector with entry ``k``
    replaced by the arithmetic mean (in log space) of the other ``K-1`` entries.
    Returns a tensor shaped like ``log_weights`` (``[..., K]``).
    """
    k = tf.shape(log_weights)[-1]
    kf = tf.cast(k, log_weights.dtype)
    # Log-space mean of the other K-1 entries, per position.
    loo_mean = (
        tf.reduce_sum(log_weights, axis=-1, keepdims=True) - log_weights
    ) / (kf - 1.0)
    # Matrix [..., K(row=k), K(col=j)] = log_weights broadcast over rows, then
    # the diagonal (j == k) replaced by loo_mean[k].
    tiled = tf.repeat(tf.expand_dims(log_weights, -2), k, axis=-2)
    replaced = tf.linalg.set_diag(tiled, loo_mean)
    return reduce_logmeanexp(replaced, axis=-1)  # [..., K]


def vimco_surrogate(
    log_weights: tf.Tensor,
    log_q_discrete: tf.Tensor,
) -> VimcoObjective:
    """Build the VIMCO surrogate objective.

    Parameters
    ----------
    log_weights
        Importance log-weights ``log w_k`` with shape ``[..., K]`` and ``K >=
        2``, passed **fully attached** to the graph: every differentiable
        contribution (the reparameterised branch/model terms *and* the explicit
        ``-log q(T_k)`` term) must stay attached so its pathwise gradient flows.
    log_q_discrete
        The discrete topology log-probabilities ``log q(T_k)``, shape
        ``[..., K]``, *differentiable* in the topology parameters. Supplies the
        score-function (sampling) gradient via the leave-one-out control
        variate.

    Returns
    -------
    VimcoObjective
        ``elbo`` (the reported bound) and ``surrogate`` (maximise for gradients).
    """
    log_weights = tf.convert_to_tensor(log_weights)
    log_q_discrete = tf.cast(
        tf.convert_to_tensor(log_q_discrete), log_weights.dtype
    )
    k_static = log_weights.shape[-1]
    if k_static is not None and k_static < 2:
        raise ValueError(
            "VIMCO requires at least 2 samples (K >= 2) for the leave-one-out "
            f"control variate; got K={k_static}"
        )

    l_hat = reduce_logmeanexp(log_weights, axis=-1)  # [...]
    l_hat_minus = _leave_one_out_bounds(log_weights)  # [..., K]
    # Per-sample learning signal with the leave-one-out baseline; detached so it
    # acts as a fixed multiplier on the score term.
    learning_signal = tf.stop_gradient(
        tf.expand_dims(l_hat, -1) - l_hat_minus
    )  # [..., K]

    # Zero-*value*, score-*gradient* term: (log_q - sg(log_q)) evaluates to 0 but
    # differentiates to grad log_q. The surrogate's value therefore equals the
    # reported bound, while its gradient adds the VIMCO score contribution.
    score_term = tf.reduce_sum(
        learning_signal * (log_q_discrete - tf.stop_gradient(log_q_discrete)),
        axis=-1,
    )
    surrogate = l_hat + score_term
    return VimcoObjective(elbo=l_hat, surrogate=surrogate)


__all__ = ["vimco_surrogate", "reduce_logmeanexp", "VimcoObjective"]
