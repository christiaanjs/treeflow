import typing as tp
import tensorflow as tf
from treeflow.vi.optimizers.robust_optimizer import RobustOptimizer
import tensorflow.keras.optimizers as keras_optimizers

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
