import tensorflow as tf
import tensorflow_probability as tfp

class SiteModel():
    def __init__(self, category_count, normalise_rates=True):
        self.category_count = category_count
        self.normalise_rates = normalise_rates

    def weights(self):
        return tf.fill([self.category_count], 1.0/self.category_count)

    def p(self):
        return (2*tf.range(self.category_count) + 1) / (2 * self.category_count)
    
    def quantile(self, p, **params):
        raise NotImplementedError()

    def rates_weights(self, **params):
        weights = self.weights()
        p = self.p()
        rates = self.quantile(p, **params)
        return rates / tf.reduce_sum(rates * weights) if self.normalise_rates else rates, weights

class GammaSiteModel(SiteModel):
    def quantile(self, p, a):
        return tfp.distributions.Gamma(a, a).quantile(p)

class WeibullSiteModel(SiteModel):
    def quantile(self, p, lambd, k):
        return lambd * (-tf.math.log(1 - p)) ** (1 / k)