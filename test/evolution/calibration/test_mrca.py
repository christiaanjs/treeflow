import pytest
from numpy.testing import assert_allclose
from treeflow.tree.io import parse_newick
from treeflow.evolution.calibration.mrca import get_mrca_index


@pytest.mark.parametrize(
    ["taxa", "expected_mrca_height"],
    [
        (
            [
                "T14_1.3684210526315788",
                "T18_1.789473684210526",
                "T10_0.9473684210526315",
            ],
            1.79992,
        ),
        (["T6_0.5263157894736842", "T5_0.42105263157894735"], 2.05092),
    ],
)
def test_get_mrca_index(tree_sim_newick_file, taxa, expected_mrca_height):
    tree = parse_newick(tree_sim_newick_file, remove_zero_edges=False)
    mrca_index = get_mrca_index(tree.topology, taxa)
    assert_allclose(tree.heights[mrca_index], expected_mrca_height)
