import typing as tp
import numpy as np
import tensorflow as tf
from tensorflow_probability.python.distributions import JointDistribution
from tensorflow_probability.python.internal import prefer_static as ps


def flatten_tensor_to_1d_slices(name: str, x: tf.Tensor) -> tp.Dict[str, tf.Tensor]:
    """
    Returns
    """
    rank = ps.rank(x)
    if rank > 1:
        indices = np.ndindex(*ps.shape(x)[1:])
        x_t = tf.transpose(
            x,
            perm=tf.concat(
                [
                    tf.range(1, rank),
                    [0],
                ],
                axis=0,
            ),
        )
        return {
            f"{name}_{'_'.join([str(i) for i in index])}": tf.gather_nd(
                x_t,
                [index],
            )[0]
            for index in indices
        }
    else:
        return {name: x}


def flatten_samples_to_dict(
    samples: object, distribution: JointDistribution
) -> tp.Dict[str, tf.Tensor]:
    flat_tensors = distribution._model_flatten(samples)
    flat_names = distribution._flat_resolve_names()
    return {
        name: flat_tensor
        for var_name, tensor in zip(flat_names, flat_tensors)
        for name, flat_tensor in flatten_tensor_to_1d_slices(var_name, tensor).items()
    }


def write_samples_to_file(
    samples: object, distribution: JointDistribution, fname: str, sep=","
):
    samples_dict = flatten_samples_to_dict(samples, distribution)
    keys = list(samples_dict.keys())
    header = sep.join(keys)
    arr = np.stack([samples_dict[key].numpy() for key in keys], axis=-1)
    np.savetxt(fname, arr, delimiter=sep, header=header)
