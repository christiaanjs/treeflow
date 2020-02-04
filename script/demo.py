import tensorflow as tf
import tensorflow_probability as tfp
import treeflow.model
import treeflow.vi

def run_demo():
    newick_file = 'data/analysis-tree.nwk'
    fasta_file = 'data/sim-seq.fasta'
    n_iter = 10

    log_prob = treeflow.model.construct_model_likelihood(newick_file, fasta_file)
    q = treeflow.model.construct_surrogate_posterior(newick_file)
    res = treeflow.vi.fit_surrogate_posterior(log_prob, q, tf.optimizers.Adam(), 10)
    print(res)



if __name__ == '__main__':
    run_demo()