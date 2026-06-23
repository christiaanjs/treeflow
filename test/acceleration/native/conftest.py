import numpy as np
import pytest
import tensorflow as tf

from treeflow.acceleration.native import phylo_likelihood as native
from treeflow.acceleration.native import node_height_ratio as native_ratio
from treeflow.acceleration.native import conditional_clade as native_clade
from treeflow.tree.topology.numpy_tree_topology import NumpyTreeTopology
from treeflow.tree.topology.tensorflow_tree_topology import numpy_topology_to_tensor


def pytest_collection_modifyitems(items):
    """Tag every test in this package with the ``native`` marker."""
    for item in items:
        if "acceleration/native" in str(item.fspath).replace("\\", "/"):
            item.add_marker(pytest.mark.native)


@pytest.fixture(scope="session", autouse=True)
def _ensure_native_built():
    """Ensure the native op is loadable for this session.

    Normally the op is built on demand and these tests are skipped if it cannot
    be built (e.g. no C++ compiler). Set ``TREEFLOW_REQUIRE_NATIVE=1`` to turn
    that skip into a hard failure -- used in CI to assert the op really is
    present and working in the built image, rather than silently skipping.
    """
    import os

    require = os.environ.get("TREEFLOW_REQUIRE_NATIVE") not in (None, "", "0")

    def unavailable(message):
        if require:
            pytest.fail(message, pytrace=False)
        pytest.skip(message, allow_module_level=True)

    from treeflow.acceleration.native.build import (
        build,
        build_node_height_ratio,
        build_conditional_clade,
    )

    if not os.path.exists(native.library_path()):
        try:
            build()
        except Exception as e:  # pragma: no cover - environment dependent
            unavailable(f"Could not build native op: {e}")
    if not os.path.exists(native_ratio.library_path()):
        try:
            build_node_height_ratio()
        except Exception as e:  # pragma: no cover - environment dependent
            unavailable(f"Could not build native ratio op: {e}")
    if not os.path.exists(native_clade.library_path()):
        try:
            build_conditional_clade()
        except Exception as e:  # pragma: no cover - environment dependent
            unavailable(f"Could not build native conditional clade op: {e}")
    try:
        native.load_op_library()
        native_ratio.load_op_library()
        native_clade.load_op_library()
    except Exception as e:  # pragma: no cover
        unavailable(f"Could not load native op: {e}")


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


def make_ratio_problem(leaf_count, batch_shape=(), seed=0, dtype=tf.float64):
    """Build a random node-height ratio transform problem and its tensors.

    Returns the preorder/parent index tensors (in internal-node space, as the
    bijector passes them to ``ratios_to_node_heights``) together with random
    ratios in ``[0, 1)`` (root entry a positive height) and random non-negative
    anchor heights, optionally with leading batch dimensions.
    """
    rng = np.random.default_rng(seed)
    parent_indices = _random_parent_indices(leaf_count, rng)
    topology = numpy_topology_to_tensor(
        NumpyTreeTopology(parent_indices=parent_indices)
    )
    np_dtype = dtype.as_numpy_dtype
    node_count = leaf_count - 1  # internal nodes

    taxon_count = leaf_count
    preorder = topology.preorder_node_indices - taxon_count
    parent = topology.parent_indices[taxon_count:] - taxon_count

    shape = tuple(batch_shape) + (node_count,)
    ratios = rng.uniform(0.05, 0.95, size=shape).astype(np_dtype)
    # The last internal node (root) carries an unconstrained height, not a ratio.
    ratios[..., -1] = rng.uniform(1.0, 3.0, size=ratios[..., -1].shape)
    anchor = rng.uniform(0.0, 0.5, size=(node_count,)).astype(np_dtype)

    return dict(
        topology=topology,
        preorder_node_indices=preorder,
        parent_indices=parent,
        ratios=tf.constant(ratios),
        anchor_heights=tf.constant(anchor),
        leaf_count=leaf_count,
        node_count=node_count,
    )


@pytest.fixture
def small_ratio_problem():
    return make_ratio_problem(leaf_count=8, seed=3)


@pytest.fixture
def make_ratio_problem_factory():
    return make_ratio_problem
