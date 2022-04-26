from treeflow.evolution.calibration.calibration import MRCACalibration
import scipy.stats
from numpy.testing import assert_allclose


def test_MRCACalibration_get_normal_sample_sd():
    high = 6.4
    low = 3.2
    calibration = MRCACalibration(["a", "b"], (low, high))
    prob = 0.99
    alpha = (1 - prob) / 2.0
    sd = calibration.get_normal_sd(prob)
    loc = calibration.get_normal_mean()
    res = scipy.stats.norm.cdf([low, high], loc=loc, scale=sd)
    assert_allclose(res, [alpha, 1 - alpha])
