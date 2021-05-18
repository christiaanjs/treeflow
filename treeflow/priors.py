from scipy.optimize import minimize
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from treeflow import DEFAULT_FLOAT_DTYPE_NP, DEFAULT_FLOAT_DTYPE_TF

tfd = tfp.distributions

REAL_BOUNDS = (None, None)
NONNEGATIVE_BOUNDS = (1e-3, 1e4)
NORMAL_BOUNDS = dict(loc=REAL_BOUNDS, scale=NONNEGATIVE_BOUNDS)
default_bounds = {
    tfd.Gamma: dict(concentration=NONNEGATIVE_BOUNDS, rate=NONNEGATIVE_BOUNDS),
    tfd.Normal: NORMAL_BOUNDS,
    tfd.LogNormal: NORMAL_BOUNDS,
}

NORMAL_INIT = dict(loc=0.0, scale=1.0)
default_init = {
    tfd.Gamma: dict(concentration=1.0, rate=1.0),
    tfd.Normal: NORMAL_INIT,
    tfd.LogNormal: NORMAL_INIT,
}


def get_params_for_quantiles(
    dist_class,
    quantiles,
    init_params_dict=None,
    bounds_dict=None,
    probs=[0.025, 0.975],
    **optim_kwargs
):
    """
    Assume all parameters are scalar
    """
    if init_params_dict is None:
        init_params_dict = default_init[dist_class]

    param_keys = list(init_params_dict.keys())
    init_par = np.array(
        [init_params_dict[key] for key in param_keys], dtype=DEFAULT_FLOAT_DTYPE_NP
    )
    quantiles = tf.cast(quantiles, dtype=DEFAULT_FLOAT_DTYPE_TF)

    @tf.function
    def obj(**par_dict):
        dist = dist_class(**par_dict)
        return tf.reduce_sum((quantiles - dist.quantile(probs)) ** 2.0)

    @tf.function
    def obj_and_grad(**par_dict):
        par_list = [par_dict[key] for key in param_keys]
        obj_val = obj(**par_dict)
        return obj_val, tf.gradients(obj_val, par_list)

    def obj_and_grad_flat(x):
        obj_val, grads = obj_and_grad(**{key: x[i] for i, key in enumerate(param_keys)})
        return obj_val.numpy(), np.array([x.numpy() for x in grads])

    if bounds_dict is None:
        bounds_dict = default_bounds[dist_class]

    bounds = [bounds_dict.get(key, (None, None)) for key in param_keys]

    res = minimize(
        obj_and_grad_flat,
        init_par,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        **optim_kwargs
    )
    return {key: res.x[i] for i, key in enumerate(param_keys)}, res


def get_params_for_quantiles_lognormal_conjugate(
    cov_quantiles, mean_quantiles, **kwargs
):
    cov_to_precision = lambda x: 1.0 / np.log(x ** 2.0 + 1)
    precision_quantiles = cov_to_precision(np.array(cov_quantiles))[::-1]
    precision_params, precision_res = get_params_for_quantiles(
        tfd.Gamma, precision_quantiles
    )

    lognormal_loc_from_mean = lambda mean, scale: np.log(mean) - scale ** 2.0 / 2.0
    loc_quantiles = lognormal_loc_from_mean(
        mean_quantiles, 1.0 / np.sqrt(precision_quantiles)
    )  # Increasing function of mean and precision
    loc_params, loc_res = get_params_for_quantiles(tfd.Normal, loc_quantiles)
    return (
        dict(loc=dict(normal=loc_params), precision=dict(gamma=precision_params)),
        dict(loc=loc_res, precision=precision_res),
    )
