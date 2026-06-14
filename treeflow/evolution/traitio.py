from __future__ import annotations

import csv
import os
import typing as tp

import numpy as np
import tensorflow as tf

from treeflow import DEFAULT_FLOAT_DTYPE_TF

PathLikeType = tp.Union[str, bytes, os.PathLike]
TraitMappingType = tp.Mapping[str, str]

DEFAULT_UNKNOWN_TOKENS: tp.FrozenSet[str] = frozenset({"?", "-", "NA", "N/A", ""})


class TraitDataParseError(ValueError):
    pass


def parse_trait_csv(
    filename: PathLikeType,
    taxon_column: str = "taxon",
    trait_column: str = "trait",
) -> TraitMappingType:
    """Read a two-column CSV of (taxon, trait) and return a mapping.

    Parameters
    ----------
    filename : path-like
        Path to a CSV file with at least the ``taxon_column`` and
        ``trait_column`` headers.
    taxon_column : str
        Name of the column holding taxon labels.
    trait_column : str
        Name of the column holding discrete trait labels.

    Returns
    -------
    Mapping[str, str]
        Dictionary from taxon name to (string) trait label.
    """
    try:
        with open(filename, newline="") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None:
                raise TraitDataParseError(
                    f"Trait file {filename} is empty or has no header row"
                )
            missing = {taxon_column, trait_column} - set(reader.fieldnames)
            if missing:
                raise TraitDataParseError(
                    f"Trait file {filename} missing required column(s): "
                    f"{sorted(missing)}. Found: {reader.fieldnames}"
                )
            mapping: tp.Dict[str, str] = {}
            for row_num, row in enumerate(reader, start=2):
                taxon = row[taxon_column]
                trait = row[trait_column]
                if taxon in mapping:
                    raise TraitDataParseError(
                        f"Duplicate taxon {taxon!r} in trait file "
                        f"{filename} at row {row_num}"
                    )
                mapping[taxon] = trait
    except FileNotFoundError:
        raise
    except OSError as ex:
        raise TraitDataParseError(f"Error reading trait file {filename}: {ex}")
    return mapping


class DiscreteTraitData:
    """Discrete trait data at tips, for phylogeography / DTA.

    Stores a mapping from taxon label to trait state, together with the
    ordered set of possible trait states. Produces one-hot partials arrays
    aligned to a caller-supplied taxon order, matching the layout consumed
    by ``LeafCTMC`` and the phylogenetic likelihood machinery.

    Parameters
    ----------
    csv_file : Optional[PathLikeType]
        Path to a CSV file with at least ``taxon_column`` and
        ``trait_column`` headers. Mutually exclusive with ``trait_mapping``.
    trait_mapping : Optional[Mapping[str, str]]
        Mapping from taxon label to trait state label.
    states : Optional[Sequence[str]]
        Ordered sequence of the full set of possible trait states. If
        ``None``, the states are inferred from ``trait_mapping`` and sorted
        lexicographically for reproducibility.
    unknown_tokens : Iterable[str]
        Trait-state labels interpreted as "unknown" — these tips receive
        flat (uniform) partials rather than a one-hot vector. Defaults to
        the common placeholders ``?``, ``-``, ``NA``, ``N/A``, and ``""``.
    taxon_column, trait_column : str
        Column names in the CSV (only used when ``csv_file`` is provided).
    """

    _trait_mapping: TraitMappingType

    def __init__(
        self,
        csv_file: tp.Optional[PathLikeType] = None,
        trait_mapping: tp.Optional[TraitMappingType] = None,
        states: tp.Optional[tp.Sequence[str]] = None,
        unknown_tokens: tp.Iterable[str] = DEFAULT_UNKNOWN_TOKENS,
        taxon_column: str = "taxon",
        trait_column: str = "trait",
    ):
        if (csv_file is None) == (trait_mapping is None):
            raise ValueError(
                "Exactly one of `csv_file` or `trait_mapping` must be supplied"
            )
        if csv_file is not None:
            self._trait_mapping = parse_trait_csv(
                csv_file,
                taxon_column=taxon_column,
                trait_column=trait_column,
            )
            self.csv_file = csv_file
        else:
            assert trait_mapping is not None
            self._trait_mapping = dict(trait_mapping)
            self.csv_file = None

        self._unknown_tokens = frozenset(unknown_tokens)

        observed_states = {
            t for t in self._trait_mapping.values() if t not in self._unknown_tokens
        }
        if states is None:
            self._states: tp.Tuple[str, ...] = tuple(sorted(observed_states))
        else:
            extra = observed_states - set(states)
            if extra:
                raise TraitDataParseError(
                    f"Trait labels {sorted(extra)} not present in supplied "
                    f"`states` list {list(states)}"
                )
            self._states = tuple(states)

        if len(self._states) < 2:
            raise TraitDataParseError(
                f"Need at least two distinct trait states, got {self._states}"
            )

        self._state_index = {s: i for i, s in enumerate(self._states)}

    @property
    def states(self) -> tp.Tuple[str, ...]:
        return self._states

    @property
    def n_states(self) -> int:
        return len(self._states)

    @property
    def taxon_count(self) -> int:
        return len(self._trait_mapping)

    @property
    def site_count(self) -> int:
        """Always 1 — a discrete trait is a single-site "alignment" with K states.

        Exposed for duck-type compatibility with :class:`Alignment` in the
        phylogenetic likelihood pipeline.
        """
        return 1

    @property
    def trait_mapping(self) -> TraitMappingType:
        return dict(self._trait_mapping)

    def get_encoded_trait_array(
        self, taxon_names: tp.Iterable[str]
    ) -> np.ndarray:
        """Return ``[1, n_taxa, K]`` partials array in the supplied taxon order.

        Unknown states produce flat partials ``[1, 1, ..., 1]``; known states
        produce one-hot vectors. The leading singleton axis is the "site"
        axis, matching the ``[n_sites, n_taxa, n_states]`` layout produced
        by :func:`encode_sequence_mapping` for nucleotide alignments.
        """
        taxon_names = list(taxon_names)
        K = self.n_states
        flat = np.ones(K, dtype=np.float64)
        out = np.zeros((1, len(taxon_names), K), dtype=np.float64)
        for i, taxon in enumerate(taxon_names):
            if taxon not in self._trait_mapping:
                raise TraitDataParseError(
                    f"Taxon {taxon!r} requested but not present in trait data"
                )
            state = self._trait_mapping[taxon]
            if state in self._unknown_tokens:
                out[0, i, :] = flat
            else:
                out[0, i, self._state_index[state]] = 1.0
        return out

    def get_encoded_trait_tensor(
        self,
        taxon_names: tp.Iterable[str],
        dtype: tf.DType = DEFAULT_FLOAT_DTYPE_TF,
    ) -> tf.Tensor:
        return tf.constant(self.get_encoded_trait_array(taxon_names), dtype=dtype)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(taxon_count={self.taxon_count}, "
            f"n_states={self.n_states}, states={list(self._states)})"
        )


__all__ = [
    "DiscreteTraitData",
    "TraitDataParseError",
    "parse_trait_csv",
    "DEFAULT_UNKNOWN_TOKENS",
]
