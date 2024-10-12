import typing as tp
import tensorflow as tf
from tensorflow_probability.python.distributions import JointDistribution
from treeflow.tree.rooted.tensorflow_rooted_tree import TensorflowRootedTree


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
