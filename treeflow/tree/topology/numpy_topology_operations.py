import numpy as np
from numpy.typing import ArrayLike
import typing as tp


def _get_child_indices_flat(parent_indices: np.ndarray) -> np.ndarray:
    node_count = parent_indices.shape[-1] + 1
    child_indices = np.full((node_count, 2), -1)
    current_child = np.zeros(node_count, dtype=int)
    for i in range(node_count - 1):
        parent = parent_indices[i]
        if (
            current_child[parent] == 0
            or child_indices[parent, current_child[parent] - 1] < i
        ):
            child_indices[parent, current_child[parent]] = i
        else:  # Ensure last axis sorted
            child_indices[parent, current_child[parent]] = child_indices[
                parent, current_child[parent] - 1
            ]
            child_indices[parent, current_child[parent] - 1] = 1
        current_child[parent] += 1
    return child_indices


get_child_indices: tp.Callable[[ArrayLike], np.ndarray] = np.vectorize(
    _get_child_indices_flat, otypes=[np.int32], signature="(m)->(n,2)"
)


def _get_preorder_indices_flat(child_indices: np.ndarray):
    node_count = child_indices.shape[-2]

    def is_leaf(node_index):
        return child_indices[node_index, 0] == -1

    stack = np.zeros(node_count, dtype=int)
    stack[0] = len(child_indices) - 1
    stack_length = 1

    visited = np.zeros(node_count, dtype=int)
    visited_count = 0

    while stack_length:
        node_index = stack[stack_length - 1]
        stack_length -= 1
        if not is_leaf(node_index):
            for child_index in child_indices[node_index][::-1]:
                stack[stack_length] = child_index
                stack_length += 1
        visited[visited_count] = node_index
        visited_count += 1
    return visited


get_preorder_indices: tp.Callable[[ArrayLike], np.ndarray] = np.vectorize(
    _get_preorder_indices_flat, otypes=[np.int32], signature="(m,2)->(m)"
)

__all__ = ["get_child_indices", "get_preorder_indices"]
