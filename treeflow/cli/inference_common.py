import typing as tp
import tensorflow as tf
import tensorflow.keras.optimizers as keras_optimizers
from treeflow.vi.optimizers.robust_optimizer import RobustOptimizer
from treeflow.model.phylo_model import DEFAULT_TREE_VAR_NAME, PhyloModel
from treeflow.tree.io import write_tensor_trees

ADAM_KEY = "adam"
ROBUST_ADAM_KEY = "robust_adam"
optimizer_builders = {
    ADAM_KEY: keras_optimizers.Adam,
    ROBUST_ADAM_KEY: lambda *args, **kwargs: RobustOptimizer(
        keras_optimizers.Adam(*args, **kwargs)
    ),
}

EXAMPLE_PHYLO_MODEL_DICT = dict(
    tree=dict(coalescent=dict(pop_size=dict(exponential=dict(rate=0.1)))),
    clock=dict(strict=dict(clock_rate=dict(exponential=dict(rate=1000.0)))),
    substitution="jc",
)


def parse_init_value(init_value_string: str) -> tp.Union[float, tp.List[float]]:
    split = [float(x) for x in init_value_string.split("|")]
    if len(split) == 1:
        return split[0]
    else:
        return split


def parse_init_values(init_values_string: str) -> tp.Dict[str, tf.Tensor]:
    str_dict = dict(item.split("=") for item in init_values_string.split(","))
    return {key: parse_init_value(value) for key, value in str_dict.items()}


def get_tree_vars(model: PhyloModel) -> tp.Set[str]:
    tree_vars = {DEFAULT_TREE_VAR_NAME}
    if model.relaxed_clock():
        tree_vars.add("branch_rates")
    return tree_vars


def write_trees(
    tree_var_samples: tp.Dict[str, tf.Tensor], topology_file, output_file
) -> None:
    branch_lengths = tree_var_samples.pop(DEFAULT_TREE_VAR_NAME).branch_lengths
    write_tensor_trees(
        topology_file, branch_lengths, output_file, branch_metadata=tree_var_samples
    )
