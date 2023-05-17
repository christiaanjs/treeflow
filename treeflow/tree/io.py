import ete3
import numpy as np
import dendropy
from dendropy.dataio import nexusprocessing
from treeflow import DEFAULT_FLOAT_DTYPE_NP
from treeflow.tree.rooted.numpy_rooted_tree import NumpyRootedTree
from treeflow.tree.taxon_set import DictTaxonSet

_EPSILON = 1e-6


def remove_zero_edges(
    tree: NumpyRootedTree, epsilon: float = _EPSILON
) -> NumpyRootedTree:
    heights = np.array(tree.heights)
    child_indices = tree.topology.child_indices
    for node_index in tree.topology.postorder_node_indices:
        child_heights = heights[child_indices[node_index]]
        heights[node_index] = np.max(
            (heights[node_index], np.max(child_heights) + epsilon)
        )
    return NumpyRootedTree(heights=heights, topology=tree.topology)


_remove_zero_edges_func = remove_zero_edges


def parse_newick(
    newick_file: str,
    remove_zero_edges: bool = True,
    epsilon: float = _EPSILON,
    subnewick_format=0,
) -> NumpyRootedTree:
    """
    Read a rooted TreeFlow Numpy tree from a Newick file

    Parameters
    ----------
    newick_file : str
        File to read tree from
    remove_zero_edges : bool
        Whether to expand zero-length edges (default: True)
    epsilon : float
        Value to expand zero-length edges to (default: 1e-6)
    subnewick_format : int
        Format passed to `ete3.Tree` (default: 0)
        (see documentation at http://etetoolkit.org/docs/latest/reference/reference_tree.html)

    Returns
    -------
    NumpyRootedTree
        Parsed TreeFlow tree composed of Numpy arrays
    
    """
    t = ete3.Tree(newick_file, format=subnewick_format)
    ordered_nodes = sorted(t.traverse("postorder"), key=lambda n: not n.is_leaf())

    indices = {n: i for i, n in enumerate(ordered_nodes)}
    parent_indices = np.array([indices[n.up] for n in ordered_nodes[:-1]])

    root_distances = np.array(
        [t.get_distance(n) for n in ordered_nodes], dtype=DEFAULT_FLOAT_DTYPE_NP
    )  # TODO: Optimise
    root_height = max(root_distances)
    heights = root_height - root_distances

    taxon_count = (len(ordered_nodes) + 1) // 2
    taxon_set = DictTaxonSet([x.name for x in ordered_nodes[:taxon_count]])
    tree = NumpyRootedTree(
        heights=heights,
        parent_indices=parent_indices,
        taxon_set=taxon_set,
    )
    if remove_zero_edges:
        tree = _remove_zero_edges_func(tree, epsilon=epsilon)

    return tree


def tensor_to_dendro(
    topology, taxon_namespace, taxon_names, branch_lengths, branch_metadata={}
):
    taxon_count = len(taxon_names)
    leaves = [
        dendropy.Node(taxon=taxon_namespace.get_taxon(name)) for name in taxon_names
    ]
    nodes = leaves + [dendropy.Node() for _ in range(taxon_count - 1)]
    for i, node in enumerate(nodes[:-1]):
        node.edge_length = branch_lengths[i]
        for key, value in branch_metadata.items():
            node.annotations[key] = value[i]
        parent = nodes[topology.parent_indices[i]]
        parent.add_child(node)
    return dendropy.Tree(
        taxon_namespace=taxon_namespace, seed_node=nodes[-1], is_rooted=True
    )




class CustomNewickWriter(dendropy.dataio.newickwriter.NewickWriter):
    def _write_node_body(self, node, out):
        out.write(self._render_node_tag(node))
        if not self.suppress_annotations:
            node_annotation_comments = (
                nexusprocessing.format_item_annotations_as_comments(
                    node,  # Place node annotations before colon
                    nhx=self.annotations_as_nhx,
                    real_value_format_specifier=self.real_value_format_specifier,
                )
            )
            out.write(node_annotation_comments)
        if node.edge and node.edge.length != None and not self.suppress_edge_lengths:
            out.write(":{}".format(self.edge_label_compose_fn(node.edge)))
        if not self.suppress_annotations:
            edge_annotation_comments = (
                nexusprocessing.format_item_annotations_as_comments(
                    node.edge,
                    nhx=self.annotations_as_nhx,
                    real_value_format_specifier=self.real_value_format_specifier,
                )
            )
            out.write(edge_annotation_comments)
        out.write(self._compose_comment_string(node))
        out.write(self._compose_comment_string(node.edge))


class CustomNexusWriter(dendropy.dataio.nexuswriter.NexusWriter):
    def __init__(self, **kwargs):
        super(CustomNexusWriter, self).__init__(**kwargs)

        kwargs_to_preserve = [
            "unquoted_underscores",
            "preserve_spaces",
            "annotations_as_nhx",
            "suppress_annotations",
            "suppress_item_comments",
        ]

        newick_kwargs = dict(
            unquoted_underscores=self.unquoted_underscores,
            preserve_spaces=self.preserve_spaces,
            annotations_as_nhx=self.annotations_as_nhx,
            suppress_annotations=self.suppress_annotations,
            suppress_item_comments=self.suppress_item_comments,
        )
        self._newick_writer = CustomNewickWriter(**newick_kwargs)


def fit_successful(variational_fit):
    return np.isfinite(variational_fit["loss"]).all()


NUMERICAL_ISSUE_N = 2


def write_tensor_trees(topology_file, branch_lengths, output_file, branch_metadata={}):
    taxon_namespace = dendropy.Tree.get(
        path=topology_file, schema="newick", preserve_underscores=True
    ).taxon_namespace
    tree = parse_newick(topology_file)
    trees = dendropy.TreeList(
        [
            tensor_to_dendro(
                tree.topology,
                taxon_namespace,
                tree.taxon_set,
                branch_lengths[i],
                branch_metadata={
                    key: value[i] for key, value in branch_metadata.items()
                },
            )
            for i in range(branch_lengths.shape[0])
        ],
        taxon_namespace=taxon_namespace,
    )

    writer = CustomNexusWriter(unquoted_underscores=True)
    with open(output_file, "w") as f:
        writer.write_tree_list(trees, f)


__all__ = ["parse_newick", "remove_zero_edges", "write_tensor_trees"]
