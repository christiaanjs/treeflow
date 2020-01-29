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
    def kl():
        q_samples = q.sample()
        return q.log_prob(q_samples) - log_prob(**q_samples)

    with tf.GradientTape(watch_accessed_variables=True) as t:
        loss = kl()

    res = t.gradient(loss, t.watched_variables())
    print(res)



if __name__ == '__main__':
    run_demo()