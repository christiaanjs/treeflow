import ete3
import numpy as np

def parse_newick(newick_file):
    """Return leaves followed by nodes (postorder)"""
    t = ete3.Tree(newick_file)
    ordered_nodes = sorted(t.traverse("postorder"), key=lambda n: not n.is_leaf())

    indices = { n: i for i, n in enumerate(ordered_nodes) }
    parent_indices = np.array([indices[n.up] for n in ordered_nodes[:-1]])

    root_distances = np.array([t.get_distance(n) for n in ordered_nodes]) # TODO: Optimise
    root_height = max(root_distances)
    heights = root_height - root_distances

    taxon_count = (len(ordered_nodes) + 1)//2
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

def get_preorder_node_indices(child_indices):
    preorder_indices = get_preorder_indices(child_indices)
    taxon_count = (len(preorder_indices) + 1) // 2
    return preorder_indices[preorder_indices >= taxon_count]

def update_topology_dict(topology):
    parent_indices = topology['parent_indices']
    child_indices = get_child_indices(parent_indices)
    return dict(
        postorder_node_indices=get_postorder_node_indices(len(parent_indices)//2 + 1), # Parent indices for every vertex except root
        child_indices=child_indices,
        preorder_indices=get_preorder_indices(child_indices),
        preorder_node_indices=get_preorder_node_indices(child_indices),
        sibling_indices=get_sibling_indices(child_indices),
        **topology)

def get_node_anchor_heights(heights, postorder_node_indices, child_indices):
    taxon_count = len(postorder_node_indices) + 1
    anchor_heights = np.zeros_like(heights)
    anchor_heights[:taxon_count] = heights[:taxon_count]

    for i in postorder_node_indices:
        anchor_heights[i] = np.max(anchor_heights[child_indices[i]])

    return anchor_heights[taxon_count:]