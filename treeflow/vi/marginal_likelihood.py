import tensorflow as tf
import numpy as np
from tensorflow_probability.python.distributions import Distribution
from tensorflow_probability.python.util import SeedStream
import treeflow


def estimate_log_ml_importance_sampling(
    model: Distribution,
    approx: Distribution,
    n_samples=100,
    approx_samples=None,
    return_std=False,
    vectorise_log_prob=True,
    seed=None,
) -> tf.Tensor:
    """
    Estimate the log marginal likelihood using importance sampling
    This estimate can have high variance if the fit of the approximation
    is poor.
    Parameters
    ----------
    model
        The (pinned) distribution representing the prior and likelihood
    approx
        A fitted variational approximation
    n_samples
        The number of samples to use in the estimate
    """
    assert not (
        (not vectorise_log_prob) and (not approx_samples is None)
    ), "If samples are provided then vectorised log prob much be used"

    if vectorise_log_prob:
        if approx_samples is None:
            approx_samples = approx.sample(n_samples, seed=seed)
        model_log_probs = model.unnormalized_log_prob(approx_samples)
        approx_log_probs = approx.log_prob(approx_samples)
        estimates = model_log_probs - approx_log_probs
    else:

        seed = SeedStream(seed, salt="ml_estimate")
        estimates = np.zeros((n_samples,), dtype=treeflow.DEFAULT_FLOAT_DTYPE_NP)

        @tf.function
        def estimate_fn(seed):
            sample = approx.sample(seed=seed)
            model_log_prob = model.unnormalized_log_prob(sample)
            approx_log_prob = approx.log_prob(sample)
            return model_log_prob - approx_log_prob

        for i in range(n_samples):
            estimates[i] = estimate_fn(seed()).numpy()

    n_samples_float = tf.cast(n_samples, estimates.dtype)
    ml_estimate = tf.math.reduce_logsumexp(estimates) - tf.math.log(n_samples_float)
    if return_std:
        std = tf.math.reduce_std(estimates)
        ml_estimate_std = std / tf.sqrt(n_samples_float)
        return (ml_estimate, ml_estimate_std)
    else:
        return ml_estimate
