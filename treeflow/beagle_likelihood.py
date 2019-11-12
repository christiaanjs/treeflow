from base_likelihood import BaseLikelihood
import numpy as np

class BeagleLikelihood(BaseLikelihood):
    def __init__(self, *args, **kwargs):
       super(BeagleLikelihood, self).__init__(*args, **kwargs) 
       self.inst.make_beagle_instances(1)
       parent_indices = self.get_parent_indices()
       self.root_child_indices = np.nonzero(parent_indices == len(parent_indices))[0]
    
    def compute_likelihood(self, branch_lengths):
        self.branch_lengths[:-1] = branch_lengths
        return np.array(self.inst.log_likelihoods())

    def compute_gradient(self, branch_lengths):
        self.branch_lengths[:-1] = branch_lengths
        gradient = np.array(self.inst.branch_gradients()[0][1])[:-1]
        gradient[self.root_child_indices] = np.sum(gradient[self.root_child_indices])
        return gradient
        


