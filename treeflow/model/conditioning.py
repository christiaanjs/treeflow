from re import I
from black import out
from tensorflow_probability.python.distributions import (
    JointDistribution,
    TransformedDistribution,
)
from tensorflow_probability.python.bijectors import Restructure
import typing as tp
from treeflow.tree.rooted.tensorflow_rooted_tree import TensorflowRootedTree


def flatten_trees(
    model: JointDistribution, trees: tp.Iterable[str], name="FlattenedDistribution"
) -> JointDistribution:
    tree_set = set(trees)
    event_shape = model.event_shape

    i = 0
    input_structure: tp.Dict[str, tp.Union[int, TensorflowRootedTree]] = {}
    output_structure: tp.Dict[str, int] = {}
    for key, value in event_shape.items():
        if key in tree_set:
            assert isinstance(value, TensorflowRootedTree)
            node_height_index = i
            sampling_times_index = node_height_index + 1
            topology_index = sampling_times_index + 1

            input_structure[key] = TensorflowRootedTree(
                node_heights=node_height_index,
                sampling_times=sampling_times_index,
                topology=topology_index,
            )

            output_structure[f"{key}_node_heights"] = node_height_index
            output_structure[f"{key}_sampling_times"] = sampling_times_index
            output_structure[f"{key}_topology"] = topology_index

            i = topology_index + 1
        else:
            input_structure[key] = i
            output_structure[key] = i

            i += 1
    bijector = Restructure(output_structure, input_structure)
    return TransformedDistribution(model, bijector, name=name)
