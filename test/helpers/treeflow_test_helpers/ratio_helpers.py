from collections import namedtuple
import numpy as np
from treeflow.tree.topology.numpy_tree_topology import NumpyTreeTopology
from treeflow.tree.rooted.numpy_rooted_tree import NumpyRootedTree
from treeflow.tree.topology.tensorflow_tree_topology import (
    TensorflowTreeTopology,
    numpy_topology_to_tensor,
)

RatioTestData = namedtuple(
    "RatioTestData",
    [
        "heights",
        "node_parent_indices",
        "parent_indices",
        "preorder_node_indices",
        "ratios",
        "anchor_heights",
        "sampling_times",
    ],
)


def numpy_topology_from_ratio_test_data(
    ratio_test_data: RatioTestData,
) -> NumpyTreeTopology:
    return NumpyTreeTopology(parent_indices=ratio_test_data.parent_indices)


def topology_from_ratio_test_data(
    ratio_test_data: RatioTestData,
) -> TensorflowTreeTopology:
    return numpy_topology_to_tensor(
        numpy_topology_from_ratio_test_data(ratio_test_data)
    )


def numpy_tree_from_ratio_test_data(ratio_test_data: RatioTestData) -> NumpyRootedTree:
    return NumpyRootedTree(
        heights=np.concatenate(
            [ratio_test_data.sampling_times, ratio_test_data.heights], axis=-1
        ),
        topology=numpy_topology_from_ratio_test_data(ratio_test_data),
    )


__all__ = [
    RatioTestData.__name__,
    numpy_topology_from_ratio_test_data.__name__,
    topology_from_ratio_test_data.__name__,
    numpy_tree_from_ratio_test_data.__name__,
]
