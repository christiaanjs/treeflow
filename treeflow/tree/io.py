import typing as tp
import ete3
from ete3.parser.newick import NewickError
import numpy as np
import dendropy
from dendropy.dataio import nexusprocessing
import tensorflow as tf
from treeflow import DEFAULT_FLOAT_DTYPE_NP
from treeflow.tree.topology.numpy_tree_topology import NumpyTreeTopology
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


class TreeParseError(ValueError):
    pass


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
    try:
        t = ete3.Tree(newick_file, format=subnewick_format)
    except NewickError as ex:
        raise TreeParseError(f"Error parsing tree: {ex}")
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
    topology: NumpyTreeTopology,
    taxon_namespace: dendropy.TaxonNamespace,
    taxon_names: tp.Sequence[str],
    branch_lengths: tp.Sequence[float],
    branch_metadata: tp.Optional[tp.Dict[str, tp.Sequence[float]]] = None,
):
    taxon_count = len(taxon_names)
    leaves = [
        dendropy.Node(taxon=taxon_namespace.get_taxon(name)) for name in taxon_names
    ]
    nodes = leaves + [dendropy.Node() for _ in range(taxon_count - 1)]
    for i, node in enumerate(nodes[:-1]):
        node.edge_length = branch_lengths[i]
        if branch_metadata is not None:
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


def write_tensor_trees(
    topology_file: str,
    branch_lengths: tf.Tensor,
    output_file: str,
    branch_metadata: tp.Optional[tp.Mapping[str, tf.Tensor]] = None,
):
    """
    Write a collection of Tensor tree branch lengths, and possibly branch metadata,
    to a Nexus file

    Parameters
    ----------
    topology_file : str
        Newick file to read tree topology from
    branch_lengths : Tensor
        Tensor of branch lengths with dimensions (num_samples, num_branches)
    output_file : str
        File to write trees to in Nexus format
    branch_metadata : Mapping[str, Tensor] (optional)
        Mapping from keys to Tensors with dimensions (num_samples, num_branches)
        containing branch metadata
    """
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
                branch_metadata=(
                    None
                    if branch_metadata is None
                    else {key: value[i] for key, value in branch_metadata.items()}
                ),
            )
            for i in range(branch_lengths.shape[0])
        ],
        taxon_namespace=taxon_namespace,
    )

    writer = CustomNexusWriter(unquoted_underscores=True)
    with open(output_file, "w") as f:
        writer.write_tree_list(trees, f)


__all__ = ["parse_newick", "remove_zero_edges", "write_tensor_trees"]
