from treeflow.evolution.site_rate_variation.base_site_model import BaseSiteModel
from tensorflow_probability.python.distributions.gamma import Gamma


class GammaSiteModel(BaseSiteModel):
    def quantile(self, p, a):
        return Gamma(a, a).quantile(p)
