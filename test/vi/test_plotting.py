import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from treeflow.vi.plotting import plot_parameter_traces
from treeflow.vi.util import TracedCoordinates

NUM_STEPS = 7
NUM_COORDS = 10


def _trace(coord_indices):
    """A trace whose value at every step is the coordinate's index."""
    values = np.tile(np.asarray(coord_indices, dtype=float), (NUM_STEPS, 1))
    return {"tree_loc:0": values}


def _labelled_lines(ax):
    return {line.get_label(): line for line in ax.get_lines()}


def test_sample_labels_root_of_full_trace():
    ax = plot_parameter_traces(
        _trace(range(NUM_COORDS)), sample=True, tree_coords=3
    )
    lines = _labelled_lines(ax)
    assert "tree_loc[root]" in lines
    # For a full trace the root is the last coordinate.
    assert lines["tree_loc[root]"].get_ydata()[0] == NUM_COORDS - 1


def test_sample_labels_root_of_subsampled_trace():
    # A trace holding coordinates 2, 5 and 9 of a 10-coordinate variable: the
    # root (coordinate 9) is at position 2, and positions are not coordinates.
    indices = [2, 5, 9]
    ax = plot_parameter_traces(
        _trace(indices),
        sample=True,
        tree_coords=3,
        parameter_coords={
            "tree_loc:0": TracedCoordinates(indices=np.asarray(indices), size=NUM_COORDS)
        },
    )
    lines = _labelled_lines(ax)
    assert lines["tree_loc[root]"].get_ydata()[0] == 9
    assert set(lines) == {"tree_loc[root]", "tree_loc[node 2]", "tree_loc[node 5]"}


def test_sample_omits_root_when_not_traced():
    indices = [1, 4, 6]
    ax = plot_parameter_traces(
        _trace(indices),
        sample=True,
        tree_coords=3,
        parameter_coords={
            "tree_loc:0": TracedCoordinates(indices=np.asarray(indices), size=NUM_COORDS)
        },
    )
    labels = set(_labelled_lines(ax))
    assert not any("root" in label for label in labels)
    assert labels == {"tree_loc[node 1]", "tree_loc[node 4]", "tree_loc[node 6]"}


def test_sample_labels_non_tree_coordinates_by_index():
    indices = [3, 8]
    ax = plot_parameter_traces(
        {"full_rank_loc:0": np.tile(np.asarray(indices, dtype=float), (NUM_STEPS, 1))},
        sample=True,
        coords_per_var=2,
        parameter_coords={
            "full_rank_loc:0": TracedCoordinates(
                indices=np.asarray(indices), size=NUM_COORDS
            )
        },
    )
    assert set(_labelled_lines(ax)) == {"full_rank_loc[3]", "full_rank_loc[8]"}


def test_full_layout_title_reports_subsampling():
    indices = [2, 5, 9]
    axes = plot_parameter_traces(
        _trace(indices),
        parameter_coords={
            "tree_loc:0": TracedCoordinates(indices=np.asarray(indices), size=NUM_COORDS)
        },
    )
    assert axes[0].get_title() == "tree_loc:0  (3 of 10)"


def test_full_layout_title_for_full_trace():
    axes = plot_parameter_traces(_trace(range(NUM_COORDS)))
    assert axes[0].get_title() == f"tree_loc:0  ({NUM_COORDS})"


def test_mismatched_parameter_coords_raise():
    with pytest.raises(ValueError, match="coordinate indices"):
        plot_parameter_traces(
            _trace(range(NUM_COORDS)),
            sample=True,
            parameter_coords={
                "tree_loc:0": TracedCoordinates(indices=np.asarray([0, 1]), size=NUM_COORDS)
            },
        )
