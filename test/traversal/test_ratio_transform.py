from numpy.testing import assert_allclose
import pytest
from treeflow.traversal.ratio_transform import ratios_to_node_heights
from treeflow_test_helpers.ratio_helpers import topology_from_ratio_test_data


@pytest.mark.parametrize("unroll", ["auto", "unrolled", "tensorarray", "while_loop"])
def test_ratios_to_node_heights(ratio_test_data, unroll):
    topology = topology_from_ratio_test_data(ratio_test_data)
    res = ratios_to_node_heights(
        topology,
        ratio_test_data.ratios,
        ratio_test_data.anchor_heights,
        unroll=unroll,
    )

    assert_allclose(res, ratio_test_data.heights)
