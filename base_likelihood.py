import sbn
import numpy as np

class BaseLikelihood:
    def __init__(self, newick_file, fasta_file):
        self.newick_file = newick_file
        self.fasta_file = fasta_file
        self.inst = sbn.instance('sbninstance')
        self.inst.read_fasta_file(fasta_file)
        self.inst.read_newick_file(newick_file)
        self.inst.print_status()
        
        self.branch_lengths = np.array(self.inst.tree_collection.trees[0].branch_lengths, copy=False)
        self.init_branch_lengths = np.array(self.branch_lengths)[:-1]

    def get_init_branch_lengths(self):
        return self.init_branch_lengths 

    def get_parent_indices(self):
        return np.array(self.inst.tree_collection.trees[0].parent_id_vector())

    def get_vertex_count(self):
        return len(self.init_branch_lengths) + 1

    def get_child_indices(self):
        parent_indices = self.get_parent_indices()
        child_indices = np.full((len(parent_indices) + 1, 2), -1)
        current_child = np.zeros(len(parent_indices) + 1, dtype=int)
        for i in range(len(parent_indices)):
            parent = parent_indices[i]
            child_indices[parent, current_child[parent]] = i
            current_child[parent] += 1
        return child_indices

    def get_sibling_indices(self):
        child_indices = self.get_child_indices()
        sibling_indices = np.full(child_indices.shape[0] - 1, -1)

        def is_leaf(node_index):
            return child_indices[node_index, 0] == -1

        for i in range(child_indices.shape[0]):
            if not is_leaf(i):
                left, right = child_indices[i]
                sibling_indices[left] = right
                sibling_indices[right] = left

    def get_postorder_node_traversal_indices(self):
        child_indices = self.get_child_indices()

        def is_leaf(node_index):
            return child_indices[node_index, 0] == -1

        stack = [len(child_indices) - 1]
        visited = []
        while len(stack) > 0:
            node_index = stack.pop()
            for child_index in child_indices[node_index]:
                if not is_leaf(child_index):
                    stack.append(child_index)
            visited.append(node_index)
        return visited

    def get_preorder_traversal_indices(self):
        child_indices = self.get_child_indices()
        
        def is_leaf(node_index):
            return child_indices[node_index, 0] == -1
        
        stack = [len(child_indices) - 1]
        visited = [] 
        while len(stack) > 0:
            node_index = stack.pop()
            if not is_leaf(node_index):
                for child_index in child_indices[node_index][::-1]:
                    stack.append(child_index)
            visited.append(node_index)
        return visited


    def compute_likelihood(self, branch_lengths):
        raise NotImplementedError()

    def compute_gradient(self, branch_lengths):
        raise NotImplementedError()


