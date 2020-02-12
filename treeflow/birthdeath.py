import tensorflow as tf
import treeflow.tree

class BirthDeath(treeflow.tree.TreeDistribution):
    def __init__(self, taxon_count, birth_diff_rate, relative_death_rate, sample_probability=1.0,
               validate_args=False,
               allow_nan_stats=True,
               name='BirthDeath'):
        super(BirthDeath, self).__init__(
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            parameters = dict(locals()),
            name=name
        )
        self.r = birth_diff_rate
        self.a = relative_death_rate
        self.rho = sample_probability
        self.taxon_count = taxon_count

    def _log_prob_1d(self, x, r, a, rho):
        # TODO: Validate topology
        # TODO: Check sampling times?
        heights = x['heights'][self.taxon_count:]
        
        taxon_count = tf.cast(self.taxon_count, tf.float32)
        log_coeff = (taxon_count - 1)*tf.math.log(2.0) - tf.math.lgamma(taxon_count)
        tree_logp = log_coeff + (taxon_count - 1)*tf.math.log(r*rho) + taxon_count*tf.math.log(1 - a)
        
        mrhs = -r*heights
        zs = tf.math.log(rho + ((1 - rho) - a)*tf.math.exp(mrhs))
        ls = -2*zs + mrhs
        root_term = mrhs[-1] - zs[-1]
        
        return tree_logp + tf.reduce_sum(ls) + root_term
        
    def _log_prob_1d_flat(self, x_flat):
        x_dict = {
            'heights': x_flat[0],
            'topology': {
                'parent_indices': x_flat[1]
            }
        }
        r = x_flat[2]
        a = x_flat[3]
        rho = x_flat[4]
        return self._log_prob_1d(x_dict, r, a, rho)


    def _log_prob(self, x): # For now we assume that parameters aren't batched
        batch_shape = x['heights'].shape[:-1]
        r = tf.broadcast_to(self.r, batch_shape)
        a = tf.broadcast_to(self.a, batch_shape)
        rho = tf.broadcast_to(self.rho, batch_shape)
        x_flat = [x['heights'], x['topology']['parent_indices'], r, a, rho]
        return treeflow.tf_util.vectorize_1d_if_needed(self._log_prob_1d_flat, x_flat, batch_shape.rank)


    def _sample_n(self, n, seed=None):
        import warnings
        warnings.warn('Dummy sampling')
        #raise NotImplementedError('Coalescent simulator not yet implemented')
        return {
            'heights': tf.zeros([n, 2 * self.taxon_count - 1], dtype=self.dtype['heights']),
            'topology': {
                'parent_indices': tf.zeros([n, 2 * self.taxon_count - 2], dtype=self.dtype['topology']['parent_indices'])
            }
        }