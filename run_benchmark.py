from beagle_likelihood import BeagleLikelihood
from tensorflow_likelihood import TensorflowLikelihood
import numpy as np
import time

FASTA_FILE = 'data/sim-seq.fasta'
NEWICK_FILE = 'data/analysis-tree.nwk'

likelihood_classes = [BeagleLikelihood, TensorflowLikelihood]

n_points = 2
delta = 1e-3
rate = 0.03

def init_likelihood(cls):
    return cls(newick_file=NEWICK_FILE, fasta_file=FASTA_FILE)

print('Computing reference values')

reference = init_likelihood(BeagleLikelihood)
init_branch_lengths = reference.get_init_branch_lengths() * rate
branch_values = np.repeat(init_branch_lengths[np.newaxis, :], n_points * len(init_branch_lengths), 0)

print('Benchmarking on {0} values'.format(branch_values.shape[0]))

for i in range(n_points):
    for j in range(len(init_branch_lengths)):
        index = (i * len(init_branch_lengths)) + j
        branch_values[index, j] += delta * (i + 1)

reference_branch_lengths = branch_values[-1]
reference_likelihood = reference.compute_likelihood(reference_branch_lengths)
reference_gradient = reference.compute_gradient(reference_branch_lengths)

def benchmark_likelihood(cls):
    print('Benchmarking ' + str(cls))
    likelihood = init_likelihood(cls)
    likelihood_correct = np.allclose(reference_likelihood, likelihood.compute_likelihood(reference_branch_lengths))
    gradient_correct = np.allclose(reference_gradient, likelihood.compute_gradient(reference_branch_lengths))

    likelihood_values = np.zeros(branch_values.shape[0]) 
    
    start = time.time()
    for i in range(branch_values.shape[0]):
        likelihood_values[i] = likelihood.compute_likelihood(branch_values[i])
    end = time.time()
    likelihood_time = end - start

    gradient_values = np.zeros(branch_values.shape)
    for i in range(branch_values.shape[0]):
        gradient_values[i] = likelihood.compute_gradient(branch_values[i])
    end = time.time()
    gradient_time = end - start

    return {
        'name': str(cls),
        'likelihood_correct': likelihood_correct,
        'gradient_correct': gradient_correct,
        'likelihood_time': likelihood_time,
        'gradient_time': gradient_time
    }

results = [benchmark_likelihood(cls) for cls in likelihood_classes]
print(results)
