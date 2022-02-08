import typing as tp
from tensorflow.keras.optimizers import Optimizer
from tensorflow_probability.python.distributions import Distribution
from tensorflow_probability.python.vi import fit_surrogate_posterior
from tensorflow_probability.python.optimizer.convergence_criteria import (
    ConvergenceCriterion,
)
from tensorflow_probability.python.math import MinimizeTraceableQuantities
from treeflow.tree.topology.tensorflow_tree_topology import TensorflowTreeTopology
from treeflow.model.approximation import get_fixed_topology_mean_field_approximation
from treeflow.vi.util import default_vi_trace_fn, VIResults


def fit_fixed_topology_variational_approximation(
    model: Distribution,
    topologies: tp.Dict[str, TensorflowTreeTopology],
    optimizer: Optimizer,
    num_steps: int,
    trace_fn: tp.Callable[[MinimizeTraceableQuantities], object] = default_vi_trace_fn,
    convergence_criterion: tp.Optional[ConvergenceCriterion] = None,
    init_loc: tp.Optional[object] = None,
    **vi_kwargs,
) -> tp.Tuple[Distribution, object]:
    approximation = get_fixed_topology_mean_field_approximation(
        model, init_loc=init_loc, topology_pins=topologies
    )
    opt_res = fit_surrogate_posterior(
        model.unnormalized_log_prob,
        approximation,
        optimizer,
        num_steps,
        convergence_criterion,
        trace_fn=trace_fn,
        **vi_kwargs,
    )
    return (approximation, opt_res)
