"""TreeFlow: automatic differentiation and probabilistic modelling with phylogenetic trees"""

__version__ = "0.1"

import os

if os.getenv("TREEFLOW_SILENCE_TENSORFLOW", 1) == 1:
    print("Silencing TensorFlow...")
    from silence_tensorflow import silence_tensorflow

    silence_tensorflow()

from treeflow.tf_util import (
    DEFAULT_FLOAT_DTYPE_TF,
    DEFAULT_FLOAT_DTYPE_NP,
    float_constant,
)
from treeflow.tree import parse_newick, convert_tree_to_tensor, write_tensor_trees
from treeflow.evolution import (
    Alignment,
    WeightedAlignment,
    AlignmentType,
    AlignmentFormat,
)
from treeflow.model import PhyloModel, phylo_model_to_joint_distribution

__all__ = [
    "parse_newick",
    "convert_tree_to_tensor",
    "float_constant",
    "Alignment",
    "WeightedAlignment",
    "AlignmentType",
    "AlignmentFormat",
    "PhyloModel",
    "phylo_model_to_joint_distribution",
    "write_tensor_trees",
]
