import ete3
import numpy as np

def parse_newick(newick_file):
    """Return leaves followed by nodes (postorder)"""
    t = ete3.Tree(newick_file)
    ordered_nodes = sorted(t.traverse("postorder"), key=lambda n: not n.is_leaf())

    indices = { i: n for i, n in enumerate(ordered_nodes) }
    parent_indices = np.array([indices[n.up] for n in ordered_nodes[:-1]])

    root_distances = np.array([t.get_distance(n) for n in ordered_nodes]) # TODO: Optimise
    root_height = max(root_distances)
    heights = root_height - root_distances

    taxon_count = (len(ordered_nodes) + 1)/2
    taxon_names = [x.name for x in ordered_nodes[:taxon_count]]
    return { 'topology': { 'parent_indices': parent_indices }, 'heights': heights }, taxon_names

def get_postorder_node_indices(taxon_count):
    return np.arange(taxon_count, 2*taxon_count - 1)

def get_child_indices(parent_indices):
    child_indices = np.full((len(parent_indices) + 1, 2), -1)
    current_child = np.zeros(len(parent_indices) + 1, dtype=int)
    for i in range(len(parent_indices)):
        parent = parent_indices[i]
        child_indices[parent, current_child[parent]] = i
        current_child[parent] += 1
    return child_indices

def get_sibling_indices(child_indices):
    sibling_indices = np.full(child_indices.shape[0] - 1, -1)

    def is_leaf(node_index):
        return child_indices[node_index, 0] == -1

    for i in range(child_indices.shape[0]):
        if not is_leaf(i):
            left, right = child_indices[i]
            sibling_indices[left] = right
            sibling_indices[right] = left

    return sibling_indices

def get_preorder_indices(child_indices):        
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
    return np.array(visited)