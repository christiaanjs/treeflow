from scipy.optimize import minimize
import numpy as np
import tensorflow as tf

def get_params_for_quantiles(dist_class, quantiles, init_params_dict, bounds_dict={}, probs=[0.025, 0.975], **optim_kwargs):
    """
    Assume all parameters are scalr
    """
    param_keys = list(init_params_dict.keys() for key in init_params_dict)
    init_par = np.array([init_params_dict[key] for key in param_keys])

    def flatten_func(f):
        return lambda x: f(**{x[i] for i, key in enumerate(param_keys)})

    @tf.function
    def obj(par_dict):
        dist = dist_class(**par_dict)
        return tf.reduce_sum((quantiles - dist.quantile(probs)) ** 2.0)

    @tf.function
    def obj_and_grad(par_dict):
        par_list = [par_dict[key] for key in param_keys]
        return tf.gradients([obj(**par_dict)], [par_list])
        
    obj_and_grad_flat = flatten_func(obj_and_grad)

    bounds = [bounds_dict.get(key, default=(None, None) for key in param_keys)]

    res = minimize(obj_and_grad_flat, init_par, method="L-BFGS-B", jac=True, bounds=bounds, **optim_kwargs)
    return { key: res.x[i] for i, key in enumerate(param_keys) }

