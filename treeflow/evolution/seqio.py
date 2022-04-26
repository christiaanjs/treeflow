from __future__ import annotations

import typing as tp
import os
import numpy as np
import tensorflow as tf
from collections import Counter

from treeflow import DEFAULT_FLOAT_DTYPE_TF

init_partials_dict = {
    "A": [1.0, 0.0, 0.0, 0.0],
    "C": [0.0, 1.0, 0.0, 0.0],
    "G": [0.0, 0.0, 1.0, 0.0],
    "T": [0.0, 0.0, 0.0, 1.0],
    "U": [0.0, 0.0, 0.0, 1.0],
    "-": [1.0, 1.0, 1.0, 1.0],
    "?": [1.0, 1.0, 1.0, 1.0],
    "N": [1.0, 1.0, 1.0, 1.0],
    ".": [1.0, 1.0, 1.0, 1.0],
    # Note treating all degenerate bases as gaps to maintain agreement with BEAST.
    "B": [1.0, 1.0, 1.0, 1.0],
    "D": [1.0, 1.0, 1.0, 1.0],
    "H": [1.0, 1.0, 1.0, 1.0],
    "K": [1.0, 1.0, 1.0, 1.0],
    "M": [1.0, 1.0, 1.0, 1.0],
    "R": [1.0, 1.0, 1.0, 1.0],
    "S": [1.0, 1.0, 1.0, 1.0],
    "U": [1.0, 1.0, 1.0, 1.0],
    "V": [1.0, 1.0, 1.0, 1.0],
    "W": [1.0, 1.0, 1.0, 1.0],
    "Y": [1.0, 1.0, 1.0, 1.0],
}

PathLikeType = tp.Union[str, bytes, os.PathLike]


def parse_fasta(filename: PathLikeType) -> tp.Dict[str, str]:
    with open(filename) as f:
        text = f.read()

    def process_block(block):
        lines = block.split("\n")
        return lines[0], "".join(lines[1:])

    return dict([process_block(block) for block in text.split(">")[1:]])


SequenceMappingType = tp.Mapping[str, tp.Collection[str]]


def compress_sites(
    sequence_mapping: SequenceMappingType,
) -> tp.Tuple[tp.Dict[str, tp.Tuple[str]], tp.List[int]]:
    taxa = sorted(list(sequence_mapping.keys()))
    sequences = [sequence_mapping[taxon] for taxon in taxa]
    patterns = list(zip(*sequences))
    count_dict = Counter(patterns)
    pattern_ordering = sorted(list(count_dict.keys()))
    compressed_sequences = list(zip(*pattern_ordering))
    counts = [count_dict[pattern] for pattern in pattern_ordering]
    pattern_dict = dict(zip(taxa, compressed_sequences))
    return pattern_dict, counts


def encode_sequence_mapping(
    sequence_mapping: SequenceMappingType, taxon_names: tp.Iterable[str]
) -> np.ndarray:
    """Returns array with shape [sites, taxa, states, states]"""
    return np.moveaxis(
        np.array(
            [
                [init_partials_dict[char] for char in sequence_mapping[taxon_name]]
                for taxon_name in taxon_names
            ]
        ),
        1,
        0,
    )


class Alignment:
    sequence_mapping: SequenceMappingType

    def __init__(
        self,
        fasta_file: tp.Optional[PathLikeType] = None,
        sequence_mapping: tp.Optional[SequenceMappingType] = None,
    ):
        if sequence_mapping is not None:
            self.fasta_file = fasta_file
            self.sequence_mapping = sequence_mapping
        elif fasta_file is not None:
            self.fasta_file = fasta_file
            self.sequence_mapping = parse_fasta(fasta_file)
        else:
            raise ValueError(
                "Either `sequence_mapping` or `fasta_file` must be supplied"
            )
        sequence_lengths = set(len(x) for x in self.sequence_mapping.values())
        assert len(sequence_lengths) == 1
        self.site_count = next(iter(sequence_lengths))

    def get_encoded_sequence_array(self, taxon_names: tp.Iterable[str]) -> np.ndarray:
        return encode_sequence_mapping(self.sequence_mapping, taxon_names)

    def get_encoded_sequence_tensor(
        self, taxon_names, dtype=DEFAULT_FLOAT_DTYPE_TF
    ) -> tf.Tensor:
        return tf.constant(self.get_encoded_sequence_array(taxon_names), dtype=dtype)

    def get_compressed_alignment(self) -> WeightedAlignment:
        compressed_sequence_mapping, counts = compress_sites(self.sequence_mapping)
        return WeightedAlignment(compressed_sequence_mapping, counts)

    @property
    def taxon_count(self):
        return len(self.sequence_mapping)

    @property
    def pattern_count(self):
        return len(next(iter(self.sequence_mapping.values())))

    def __repr__(self):
        return f"{type(self).__name__}(taxon_count={self.taxon_count}, pattern_count={self.pattern_count})"


class WeightedAlignment(Alignment):
    def __init__(
        self, pattern_mapping: SequenceMappingType, weights: tp.Iterable[float]
    ):
        super().__init__(sequence_mapping=pattern_mapping)
        self.weights = weights

    def get_weights_array(self) -> np.ndarray:
        return np.array(self.weights)

    def get_weights_tensor(self, dtype=DEFAULT_FLOAT_DTYPE_TF) -> tf.Tensor:
        return tf.constant(self.get_weights_array(), dtype=dtype)


__all__ = ["Alignment"]
