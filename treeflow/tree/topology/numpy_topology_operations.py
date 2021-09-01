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

__all__ = ["get_child_indices"]
