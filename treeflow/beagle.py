import libsbn
import tensorflow as tf
import treeflow.substitution_model
import treeflow.libsbn
import numpy as np
from treeflow import DEFAULT_FLOAT_DTYPE_TF, DEFAULT_FLOAT_DTYPE_NP

def log_prob_conditioned_branch_only(fasta_file, subst_model, frequencies, rescaling=False, inst=None, newick_file=None, dated=True, **subst_model_params):
    if isinstance(subst_model, treeflow.substitution_model.JC):
        subst_model_string = 'JC69'
        param_updates = { }
    elif isinstance(subst_model, treeflow.substitution_model.GTR):
        subst_model_string = 'GTR'
        param_updates = {
            'GTR rates': np.array(subst_model_params['rates']),
            'frequencies': np.array(frequencies)
        }
    elif isinstance(subst_model, treeflow.substitution_model.HKY):
        subst_model_string = 'GTR'
        kappa = subst_model_params['kappa']
        rates = np.ones(6)
        rates[1] = kappa
        rates[4] = kappa
        param_updates = {
            'GTR rates': rates,
            'frequencies': np.array(frequencies)
        }
    else:
        raise ValueError('Unsupported substitution model')

    if inst is None:
        if newick_file is None:
            raise ValueError('Either a libsbn instance or Newick file must be supplied')
        inst = treeflow.libsbn.get_instance(newick_file, dated=dated)
    inst.read_fasta_file(fasta_file)
    inst.set_rescaling(rescaling)
    model_specification = libsbn.PhyloModelSpecification(subst_model_string, 'constant','strict')
    inst.prepare_for_phylo_likelihood(model_specification, 1)

    phylo_model_param_block_map = inst.get_phylo_model_param_block_map()
    phylo_model_param_block_map["clock rate"][:] = 1.0

    for key, value in param_updates.items():
        phylo_model_param_block_map[key][:] = value

    parent_id_vector = np.array(inst.tree_collection.trees[0].parent_id_vector())

    branch_lengths = np.array(inst.tree_collection.trees[0].branch_lengths, copy=False)

    def libsbn_func(x):
        """Wrapping likelihood and gradient evaluation via libsbn."""
        branch_lengths[:-1] = x
        gradient = inst.phylo_gradients()[0]
        grad_array = np.array(gradient.branch_lengths, dtype=DEFAULT_FLOAT_DTYPE_NP)[:-1]
        return np.array(gradient.log_likelihood, dtype=DEFAULT_FLOAT_DTYPE_NP), grad_array

    libsbn_func_vec = np.vectorize(libsbn_func, [DEFAULT_FLOAT_DTYPE_NP, DEFAULT_FLOAT_DTYPE_NP], signature='(n)->(),(n)')

    @tf.custom_gradient
    def libsbn_tf_func(x):
        logp, grad_val = tf.numpy_function(libsbn_func_vec, [x], [DEFAULT_FLOAT_DTYPE_TF, DEFAULT_FLOAT_DTYPE_TF])
        def grad(dlogp):
            return tf.expand_dims(dlogp, -1) * grad_val
        return logp, grad

    return libsbn_tf_func, inst