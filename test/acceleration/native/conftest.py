import numpy as np
import pytest
import tensorflow as tf

from treeflow.acceleration.native import phylo_likelihood as native
from treeflow.tree.topology.numpy_tree_topology import NumpyTreeTopology
from treeflow.tree.topology.tensorflow_tree_topology import numpy_topology_to_tensor


def pytest_collection_modifyitems(items):
    """Tag every test in this package with the ``native`` marker."""
    for item in items:
        if "acceleration/native" in str(item.fspath).replace("\\", "/"):
            item.add_marker(pytest.mark.native)


@pytest.fixture(scope="session", autouse=True)
def _ensure_native_built():
    """Build the native op once per session; skip the module if it can't build."""
    import os

    if not os.path.exists(native.library_path()):
        try:
            from treeflow.acceleration.native.build import build

            build()
        except Exception as e:  # pragma: no cover - environment dependent
            pytest.skip(f"Could not build native op: {e}", allow_module_level=True)
    try:
        native.load_op_library()
    except Exception as e:  # pragma: no cover
        pytest.skip(f"Could not load native op: {e}", allow_module_level=True)


def _random_parent_indices(leaf_count: int, rng: np.random.Generator) -> np.ndarray:
    """Random binary topology with the children<parent index convention."""
    node_count = 2 * leaf_count - 1
    parent = np.full(node_count, -1, dtype=np.int32)
    active = list(range(leaf_count))
    next_node = leaf_count
    while len(active) > 1:
        i = rng.integers(len(active))
        a = active.pop(i)
        j = rng.integers(len(active))
        b = active.pop(j)
        parent[a] = next_node
        parent[b] = next_node
        active.append(next_node)
        next_node += 1
    return parent[:-1]  # drop root (-1), matching treeflow's parent_indices layout


def _random_stochastic(shape, rng):
    x = rng.uniform(0.1, 1.0, size=shape)
    return x / x.sum(axis=-1, keepdims=True)


def make_problem(leaf_count, state_count, site_count, seed=0, dtype=tf.float64):
    """Build a random phylogenetic-likelihood problem and its tensors."""
    rng = np.random.default_rng(seed)
    parent_indices = _random_parent_indices(leaf_count, rng)
    topology = numpy_topology_to_tensor(
        NumpyTreeTopology(parent_indices=parent_indices)
    )
    node_count = 2 * leaf_count - 1
    np_dtype = dtype.as_numpy_dtype

    # One-hot leaf sequences: [site, leaf, state]
    states = rng.integers(0, state_count, size=(site_count, leaf_count))
    sequences = np.eye(state_count, dtype=np_dtype)[states]

    # Per-node transition matrices broadcast over sites: [1, node, state, state]
    transition_probs = _random_stochastic(
        (1, node_count, state_count, state_count), rng
    ).astype(np_dtype)

    frequencies = _random_stochastic((state_count,), rng).astype(np_dtype)

    return dict(
        topology=topology,
        sequences=tf.constant(sequences),
        transition_probs=tf.constant(transition_probs),
        frequencies=tf.constant(frequencies),
        postorder_node_indices=topology.postorder_node_indices,
        node_child_indices=topology.node_child_indices,
        leaf_count=leaf_count,
        state_count=state_count,
        site_count=site_count,
    )


@pytest.fixture
def small_problem():
    return make_problem(leaf_count=6, state_count=4, site_count=12, seed=1)


@pytest.fixture
def make_large_problem():
    return make_problem
