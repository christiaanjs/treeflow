"""VBPI: variational Bayesian phylogenetic inference over tree topologies.

This subpackage implements the topology side of VBPI (Zhang & Matsen, 2019): a
subsplit Bayesian network / conditional clade distribution as the variational
family over rooted topologies, an amortised per-split branch-length
approximation, and the VIMCO multi-sample gradient estimator that trains the
discrete topology parameters.

Components
----------
:mod:`treeflow.vbpi.clade`
    Clade / subsplit primitives and decomposition of a rooted topology.
:mod:`treeflow.vbpi.support`
    :class:`SubsplitSupport` -- the pointer-array (CSR) data structure of clades
    and candidate child subsplits the SBN probabilities live on.
:mod:`treeflow.vbpi.sbn`
    :class:`SubsplitBayesianNetwork` -- differentiable ``log_prob`` and ancestral
    sampling (native C++ or NumPy) of topologies.
:mod:`treeflow.vbpi.branch_model`
    :class:`SplitLognormalBranchModel` -- per-split log-normal branch lengths.
:mod:`treeflow.vbpi.vimco`
    :func:`vimco_surrogate` -- the VIMCO gradient estimator.
"""
from treeflow.vbpi.branch_model import SplitLognormalBranchModel
from treeflow.vbpi.sbn import SampledTopologies, SubsplitBayesianNetwork
from treeflow.vbpi.support import SubsplitSupport, all_rooted_parent_indices
from treeflow.vbpi.vimco import VimcoObjective, vimco_surrogate

__all__ = [
    "SubsplitSupport",
    "all_rooted_parent_indices",
    "SubsplitBayesianNetwork",
    "SampledTopologies",
    "SplitLognormalBranchModel",
    "vimco_surrogate",
    "VimcoObjective",
]
