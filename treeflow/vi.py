from tensorflow_probability.python import math as tfp_math
from tensorflow_probability.python.vi import csiszar_divergence

from tensorflow_probability.python.internal import nest_util

_trace_loss = lambda loss, grads, variables: loss

def fit_surrogate_posterior(
    target_log_prob_fn,
    surrogate_posterior,
    optimizer,
    num_steps,
    trace_fn=_trace_loss,
    trainable_variables=None,
    seed=None,
    name='fit_surrogate_posterior'):
    def kl():
        q_samples = surrogate_posterior.sample(seed=seed)
        return surrogate_posterior.log_prob(q_samples) - nest_util.call_fn(target_log_prob_fn, q_samples)

    return tfp_math.minimize(kl,
        num_steps=num_steps,
        optimizer=optimizer,
		trace_fn=trace_fn,
		trainable_variables=trainable_variables,
		name=name) 
    
