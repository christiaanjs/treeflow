import os
import textwrap

import numpy as np
import pytest
from numpy.testing import assert_allclose

from treeflow.evolution.traitio import (
    DiscreteTraitData,
    TraitDataParseError,
    parse_trait_csv,
)


@pytest.fixture
def example_mapping():
    return {
        "taxon_A": "NY",
        "taxon_B": "HK",
        "taxon_C": "NZ",
        "taxon_D": "NY",
    }


def test_from_mapping_infers_sorted_states(example_mapping):
    data = DiscreteTraitData(trait_mapping=example_mapping)
    assert data.states == ("HK", "NY", "NZ")
    assert data.n_states == 3
    assert data.taxon_count == 4


def test_encoded_array_layout_and_onehot(example_mapping):
    data = DiscreteTraitData(trait_mapping=example_mapping)
    arr = data.get_encoded_trait_array(
        ["taxon_A", "taxon_B", "taxon_C", "taxon_D"]
    )
    assert arr.shape == (1, 4, 3)  # [sites=1, taxa, states]
    # HK=0, NY=1, NZ=2 after sorting
    expected = np.array(
        [
            [
                [0, 1, 0],  # taxon_A -> NY
                [1, 0, 0],  # taxon_B -> HK
                [0, 0, 1],  # taxon_C -> NZ
                [0, 1, 0],  # taxon_D -> NY
            ]
        ],
        dtype=np.float64,
    )
    assert_allclose(arr, expected)


def test_taxon_reordering(example_mapping):
    data = DiscreteTraitData(trait_mapping=example_mapping)
    a = data.get_encoded_trait_array(
        ["taxon_A", "taxon_B", "taxon_C", "taxon_D"]
    )
    b = data.get_encoded_trait_array(
        ["taxon_C", "taxon_A", "taxon_D", "taxon_B"]
    )
    # Row permutation by [2, 0, 3, 1]
    assert_allclose(b[0], a[0][[2, 0, 3, 1]])


def test_unknown_state_is_flat_partials():
    data = DiscreteTraitData(
        trait_mapping={"t1": "A", "t2": "B", "t3": "?"},
        states=("A", "B"),
    )
    arr = data.get_encoded_trait_array(["t1", "t2", "t3"])
    assert_allclose(arr[0, 0], [1.0, 0.0])
    assert_allclose(arr[0, 1], [0.0, 1.0])
    assert_allclose(arr[0, 2], [1.0, 1.0])


def test_explicit_states_ordering():
    data = DiscreteTraitData(
        trait_mapping={"t1": "NY", "t2": "HK", "t3": "NZ"},
        states=("NY", "HK", "NZ"),
    )
    assert data.states == ("NY", "HK", "NZ")
    arr = data.get_encoded_trait_array(["t1", "t2", "t3"])
    # NY=0, HK=1, NZ=2 under explicit ordering
    assert_allclose(arr[0, 0], [1.0, 0.0, 0.0])
    assert_allclose(arr[0, 1], [0.0, 1.0, 0.0])
    assert_allclose(arr[0, 2], [0.0, 0.0, 1.0])


def test_explicit_states_must_cover_observed():
    with pytest.raises(TraitDataParseError, match="not present"):
        DiscreteTraitData(
            trait_mapping={"t1": "A", "t2": "B", "t3": "C"},
            states=("A", "B"),
        )


def test_requires_at_least_two_states():
    with pytest.raises(TraitDataParseError, match="at least two"):
        DiscreteTraitData(trait_mapping={"t1": "A", "t2": "A"})


def test_missing_taxon_on_encode(example_mapping):
    data = DiscreteTraitData(trait_mapping=example_mapping)
    with pytest.raises(TraitDataParseError, match="not present"):
        data.get_encoded_trait_array(["taxon_A", "taxon_ZZZ"])


def test_mutually_exclusive_sources():
    with pytest.raises(ValueError, match="Exactly one"):
        DiscreteTraitData()
    with pytest.raises(ValueError, match="Exactly one"):
        DiscreteTraitData(csv_file="foo", trait_mapping={"t": "A"})


def test_csv_roundtrip(tmp_path):
    csv_path = tmp_path / "traits.csv"
    csv_path.write_text(
        textwrap.dedent(
            """\
            taxon,trait
            t1,NY
            t2,HK
            t3,NZ
            t4,?
            """
        )
    )
    data = DiscreteTraitData(csv_file=str(csv_path))
    assert data.states == ("HK", "NY", "NZ")
    arr = data.get_encoded_trait_array(["t1", "t2", "t3", "t4"])
    assert arr.shape == (1, 4, 3)
    assert_allclose(arr[0, 3], [1.0, 1.0, 1.0])  # unknown -> flat


def test_csv_custom_column_names(tmp_path):
    csv_path = tmp_path / "traits_custom.csv"
    csv_path.write_text("name,deme\nt1,A\nt2,B\n")
    data = DiscreteTraitData(
        csv_file=str(csv_path), taxon_column="name", trait_column="deme"
    )
    assert data.taxon_count == 2
    assert data.states == ("A", "B")


def test_csv_missing_column(tmp_path):
    csv_path = tmp_path / "traits_bad.csv"
    csv_path.write_text("taxon\nt1\nt2\n")
    with pytest.raises(TraitDataParseError, match="missing required column"):
        DiscreteTraitData(csv_file=str(csv_path))


def test_csv_duplicate_taxon(tmp_path):
    csv_path = tmp_path / "traits_dup.csv"
    csv_path.write_text("taxon,trait\nt1,A\nt1,B\n")
    with pytest.raises(TraitDataParseError, match="Duplicate"):
        DiscreteTraitData(csv_file=str(csv_path))


def test_encoded_tensor_dtype():
    import tensorflow as tf

    data = DiscreteTraitData(trait_mapping={"t1": "A", "t2": "B"})
    t = data.get_encoded_trait_tensor(["t1", "t2"])
    assert isinstance(t, tf.Tensor)
    # Default dtype should be treeflow's default float
    assert t.shape == (1, 2, 2)
