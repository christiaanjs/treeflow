import treeflow.tree_processing
import numpy as np
from numpy.testing import assert_allclose


def test_parse_newick(hello_newick_file):
    tree, taxon_names_res = treeflow.tree_processing.parse_newick(hello_newick_file)
    expected_heights = np.array([0.0, 0.0, 0.0, 0.1, 0.3])
    expected_parent_indices = np.array([3, 3, 4, 4])
    expected_taxon_names = ["mars", "saturn", "jupiter"]
    assert all(
        [
            res == expected
            for res, expected in zip(taxon_names_res, expected_taxon_names)
        ]
    )
    assert_allclose(tree["heights"], expected_heights, atol=1e-16)
    assert_allclose(tree["topology"]["parent_indices"], expected_parent_indices)
