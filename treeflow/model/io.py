import typing as tp
import numpy as np
import tensorflow as tf
from tensorflow_probability.python.distributions import JointDistribution
from tensorflow_probability.python.internal import prefer_static as ps
from treeflow.tree.rooted.tensorflow_rooted_tree import TensorflowRootedTree


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
) -> tp.Tuple[tp.Dict[str, tf.Tensor], tp.Dict[str, tp.List[str]]]:
    flat_tensors = distribution._model_flatten(samples)
    flat_names = distribution._flat_resolve_names()
    vars_to_flat_dicts = {
        var_name: flatten_tensor_to_1d_slices(var_name, tensor)
        for var_name, tensor in zip(flat_names, flat_tensors)
    }
    flat_dict = {
        name: flat_tensor
        for var_name, flat_dict in vars_to_flat_dicts.items()
        for name, flat_tensor in flat_dict.items()
    }
    key_mapping = {
        var_name: list(flat_dict.keys())
        for var_name, flat_dict in vars_to_flat_dicts.items()
    }
    return flat_dict, key_mapping


def calculate_tree_stats(
    name: str, tree: TensorflowRootedTree
) -> tp.Dict[str, tf.Tensor]:
    return {f"{name}_height": tree.root_height}


def write_samples_to_file(
    samples: object,
    distribution: JointDistribution,
    fname: str,
    sep=",",
    vars: tp.Optional[tp.Iterable[str]] = None,
    tree_vars: tp.Optional[tp.Mapping[str, TensorflowRootedTree]] = None,
):
    samples_dict, key_mapping = flatten_samples_to_dict(samples, distribution)
    if vars is None:
        vars = list(key_mapping.keys())
    keys = [key for var in vars for key in key_mapping[var]]

    if tree_vars is not None:
        tree_stats = {
            stat_name: stat
            for tree_name, tree in tree_vars.items()
            for stat_name, stat in calculate_tree_stats(tree_name, tree).items()
        }
        keys = keys + list(tree_stats.keys())
        samples_dict = dict(samples_dict, **tree_stats)

    header = sep.join(keys)
    arr = np.stack([samples_dict[key].numpy() for key in keys], axis=-1)
    np.savetxt(fname, arr, delimiter=sep, header=header, comments="")
