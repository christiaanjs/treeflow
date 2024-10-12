import yaml
import tensorflow as tf
from tensorflow_probability.python.distributions import (
    Normal,
    LogNormal,
    JointDistributionNamed,
)
from treeflow import (
    DEFAULT_FLOAT_DTYPE_TF,
    parse_newick,
    convert_tree_to_tensor,
    phylo_model_to_joint_distribution,
    PhyloModel,
    Alignment,
)
from treeflow.distributions.tree import ConstantCoalescent
from treeflow.model.approximation.dependencies import get_named_dependencies


def ct(x):
    return tf.constant(x, dtype=DEFAULT_FLOAT_DTYPE_TF)


def check_deps(actual, expected):
    assert set(actual.keys()) == set(expected.keys())
    for key in expected.keys():
        assert set(actual[key]) == set(expected[key])


def test_get_named_dependencies(hello_tensor_tree):
    initial_tree = hello_tensor_tree
    model = JointDistributionNamed(
        {
            "a": LogNormal(ct(0.0), ct(1.0)),
            "b": lambda a: ConstantCoalescent(
                initial_tree.taxon_count, a, initial_tree.sampling_times
            ),
            "c": lambda a, b: Normal(a + tf.reduce_sum(b.branch_lengths, axis=-1), 1.0),
            "d": Normal(ct(0.0), ct(1.0)),
            "e": lambda c, d: Normal(c + d, ct(1.0)),
        }
    )
    dependencies = get_named_dependencies(model)
    expected = {"a": [], "b": ["a"], "c": ["a", "b"], "d": [], "e": ["c", "d"]}

    check_deps(dependencies, expected)


def test_get_named_dependencies_from_phylo_model(
    actual_model_file, wnv_newick_file, wnv_fasta_file
):
    with open(actual_model_file) as f:
        model_dict = yaml.safe_load(f)

    tree = convert_tree_to_tensor(parse_newick(wnv_newick_file))
    alignment = Alignment(wnv_fasta_file)
    dist = phylo_model_to_joint_distribution(PhyloModel(model_dict), tree, alignment)

    names = dist._flat_resolve_names()
    print(names)  # TODO: Fix weird order from coroutine?
    dependencies = get_named_dependencies(dist)
    expected = {
        "pop_size": [],
        "tree": ["pop_size"],
        "kappa": [],
        "frequencies": [],
        "branch_rate_loc": [],
        "branch_rate_scale": [],
        "branch_rates": ["branch_rate_loc", "branch_rate_scale"],
        "site_gamma_shape": [],
        "alignment": [
            "kappa",
            "frequencies",
            "branch_rates",
            "site_gamma_shape",
            "tree",
        ],
    }
    check_deps(dependencies, expected)
