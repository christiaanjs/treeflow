import typing as tp
from functools import partial
import tensorflow as tf
from tensorflow_probability.python.distributions import (
    Independent,
    Normal,
    Distribution,
    JointDistribution,
    JointDistributionSequential,
)
from treeflow import DEFAULT_FLOAT_DTYPE_TF
from treeflow.tree.rooted.tensorflow_rooted_tree import TensorflowRootedTree
from treeflow.model.approximation.mean_field import single_variable_base_distribution


def find_dependencies(model: JointDistribution) -> tp.List[tp.List[int]]:
    samples = []
    deps = []

    gen = model._model_coroutine()

    try:
        dist = next(gen)
        deps.append([])  # First variable cannot have dependencies
        if isinstance(dist, JointDistribution.Root):
            dist = dist.distribution
        sample = dist.sample()

        while True:
            if isinstance(sample, TensorflowRootedTree):
                float_sample = sample.node_heights
            else:
                float_sample = sample

            samples.append(float_sample)

            with tf.GradientTape() as t:
                t.watch(samples)

                dist = gen.send(sample)

                if isinstance(dist, JointDistribution.Root):
                    dist = dist.distribution
                    sample = dist.sample()
                    deps.append([])
                else:
                    sample = dist.sample()
                    log_prob = dist.log_prob(sample)
                    grads = t.gradient(log_prob, samples)
                    var_deps = [i for i, grad in enumerate(grads) if grad is not None]
                    deps.append(var_deps)

    except StopIteration:
        pass

    return deps


def get_named_dependencies(dist: JointDistribution) -> tp.Dict[str, tp.List[str]]:
    index_deps = find_dependencies(dist)
    names = dist._flat_resolve_names()
    return {
        names[i]: [names[j] for j in var_deps] for i, var_deps in enumerate(index_deps)
    }


def get_inverse_dependencies(
    dependencies: tp.List[tp.List[int]],
) -> tp.List[tp.List[int]]:
    reverse_deps = [[] for _ in range(len(dependencies))]
    for i, var_deps in enumerate(dependencies):
        for dep_var in var_deps:
            reverse_deps[dep_var].append(i)
    return reverse_deps


def get_distribution_with_dependencies(
    *args,
    flat_event_size: int,
    dependencies: tp.Sequence[int],
    weights: tp.Optional[tf.Tensor],
    dtype=DEFAULT_FLOAT_DTYPE_TF,
) -> Distribution:
    if len(dependencies) == 0:
        assert weights is None
        return single_variable_base_distribution(flat_event_size, dtype=dtype)
    else:
        possible_dependencies = args[::-1]
        dep_values = [
            possible_dependencies[dep_index] for dep_index in dependencies
        ]  # Sequential dist provides args in reverse order
        # weights should be shape (flat_event_size, concat_dep_size)
        loc = tf.reduce_sum(weights[..., 1:] * tf.concat(dep_values, axis=-1), axis=-1)
        scale = weights[..., -1]
        return Independent(Normal(loc, scale), reinterpreted_batch_ndims=1)


def joint_distribution_from_dependency_graph(
    flat_event_size: tp.Sequence[int],
    dependencies: tp.Sequence[tp.Sequence[int]],
    weights: tp.Sequence[tp.Optional[tf.Tensor]],
):
    return JointDistributionSequential(
        [
            partial(
                get_distribution_with_dependencies,
                flat_event_size=var_size,
                dependencies=var_dependencies,
                weights=var_weights,
            )
            for var_size, var_dependencies, var_weights in zip(
                flat_event_size, dependencies, weights
            )
        ]
    )
