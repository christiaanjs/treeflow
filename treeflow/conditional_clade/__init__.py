"""Conditional clade (subsplit Bayesian network) distributions over topologies.

This subpackage provides:

* :mod:`treeflow.conditional_clade.clade` -- bitset-based clade and subsplit
  representations, with binary-vector views for embedding-based parametrisations.
* :mod:`treeflow.conditional_clade.support` -- enumeration of the conditional
  clade support and conversion to/from TreeFlow ``parent_indices`` topologies.
* :mod:`treeflow.conditional_clade.distribution` -- a differentiable distribution
  over rooted topologies, with sampling, log-probability, exhaustive enumeration
  and exact KL for small taxon sets.
* :mod:`treeflow.conditional_clade.estimators` -- gradient estimators (score
  function, leave-one-out / VIMCO, straight-through Gumbel-Softmax) and the
  "1/0 probability gradient" sampler.
"""

from treeflow.conditional_clade.clade import (
    Subsplit,
    enumerate_clade_subsplits,
    make_subsplit,
)
from treeflow.conditional_clade.support import (
    ConditionalCladeSupport,
    SubsplitAssignment,
)
from treeflow.conditional_clade.distribution import (
    ConditionalCladeDistribution,
    segment_log_softmax,
)
from treeflow.conditional_clade.tree_distribution import (
    ConditionalCladeTreeDistribution,
)

__all__ = [
    "Subsplit",
    "make_subsplit",
    "enumerate_clade_subsplits",
    "ConditionalCladeSupport",
    "SubsplitAssignment",
    "ConditionalCladeDistribution",
    "ConditionalCladeTreeDistribution",
    "segment_log_softmax",
]
