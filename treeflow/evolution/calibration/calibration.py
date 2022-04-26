import typing as tp
import numpy as np
import scipy.stats
import tensorflow as tf
from treeflow import DEFAULT_FLOAT_DTYPE_TF
from treeflow.tree.rooted.numpy_rooted_tree import NumpyRootedTree
from treeflow.evolution.calibration.mrca import get_mrca_index

CalibrationDictType = tp.Dict[str, object]

DEFAULT_INTERVAL_MASS = 0.95


class MRCACalibration:
    def __init__(
        self,
        taxa: tp.Iterable[str],
        range: tp.Iterable[float],
        name: tp.Optional[str] = None,
    ):
        self.name = name
        self.taxa = taxa
        self.low, self.high = range

    def get_mrca_index(self, tree: NumpyRootedTree):
        return get_mrca_index(tree.topology, self.taxa)

    def get_normal_sd(self, interval_mass=DEFAULT_INTERVAL_MASS):
        """
        Get the standard deviation for a Normal distribution which places
        `interval_mass` probability mass on the interval between `low` and `high`
        """
        centre = self.get_normal_mean()
        low_boundary = self.low - centre
        probability = (1 - interval_mass) / 2.0
        return low_boundary / scipy.stats.norm.ppf(probability)

    def get_normal_mean(self):
        return (self.high + self.low) / 2.0


class MRCACalibrationSet:
    def __init__(self, calibration_dicts: tp.Iterable[CalibrationDictType]):
        self.calibrations = [MRCACalibration(**x) for x in calibration_dicts]

    def get_mrca_index_array(self, tree: NumpyRootedTree) -> np.ndarray:
        return np.array(
            [calibration.get_mrca_index(tree) for calibration in self.calibrations]
        )

    def get_mrca_index_tensor(self, tree: NumpyRootedTree) -> tf.Tensor:
        return tf.constant(self.get_mrca_index_array(tree), dtype=tf.int32)

    def get_normal_sd_array(self, interval_mass=DEFAULT_INTERVAL_MASS):
        return np.array(
            [
                calibration.get_normal_sd(interval_mass=interval_mass)
                for calibration in self.calibrations
            ]
        )

    def get_normal_sd_tensor(
        self, interval_mass=DEFAULT_INTERVAL_MASS, dtype=DEFAULT_FLOAT_DTYPE_TF
    ):
        return tf.constant(
            self.get_normal_sd_array(interval_mass=interval_mass), dtype=dtype
        )

    def get_normal_mean_array(self):
        return np.array(
            [calibration.get_normal_mean() for calibration in self.calibrations]
        )

    def get_normal_mean_tensor(self, dtype=DEFAULT_FLOAT_DTYPE_TF):
        return tf.constant(self.get_normal_mean_array(), dtype=dtype)
