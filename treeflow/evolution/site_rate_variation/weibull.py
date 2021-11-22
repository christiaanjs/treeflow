from treeflow.evolution.site_rate_variation.base_site_model import BaseSiteModel
import tensorflow as tf


class WeibullSiteModel(BaseSiteModel):
    def quantile(self, p, lambd, k):
        return lambd * (-tf.math.log(1 - p)) ** (1 / k)
