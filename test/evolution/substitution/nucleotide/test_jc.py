import typing as tp
import tensorflow as tf
from treeflow.evolution.substitution.nucleotide.jc import JC
from treeflow_test_helpers.substitution_helpers import (
    EigenSubstitutionModelHelper,
)


class TestJC(EigenSubstitutionModelHelper):
    ClassUnderTest = JC
    params: tp.Mapping[str, tf.Tensor] = dict(frequencies=JC.frequencies())
