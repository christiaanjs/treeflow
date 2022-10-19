import typing as tp
from numpy import integer, single
import tensorflow as tf
import treeflow
from treeflow.tree.rooted.tensorflow_rooted_tree import TensorflowRootedTree
from treeflow.distributions.tree.coalescent.constant_coalescent import (
    ConstantCoalescent,
)
from treeflow.distributions.tree.birthdeath.birth_death_contemporary_sampling import (
    BirthDeathContemporarySampling,
)
from treeflow.distributions.tree.birthdeath.yule import Yule
from tensorflow_probability.python.distributions import (
    JointDistributionCoroutine,
    Distribution,
    LogNormal,
    Gamma,
    Normal,
    Sample,
    Weibull,
    Dirichlet,
    Beta,
    Exponential,
)
from treeflow.evolution.substitution.base_substitution_model import (
    EigendecompositionSubstitutionModel,
)
from treeflow.evolution.substitution.nucleotide import JC, HKY, GTR
from treeflow.evolution.substitution.nucleotide.gtr import GTR_RATE_ORDER
from treeflow.evolution.seqio import Alignment, WeightedAlignment
from treeflow.distributions.discrete import FiniteDiscreteDistribution
from treeflow.distributions.discretized import DiscretizedDistribution
from treeflow.distributions.discrete_parameter_mixture import DiscreteParameterMixture
from treeflow.evolution.substitution.probabilities import (
    get_transition_probabilities_tree,
)
from treeflow.distributions.leaf_ctmc import LeafCTMC
from treeflow.distributions.sample_weighted import SampleWeighted

# TODO: Better type hints
def parse_model(
    model: tp.Union[tp.Dict[str, object], str]
) -> tp.Tuple[str, tp.Optional[tp.Dict[str, tp.Union[object, tp.Dict[str, object]]]]]:
    if isinstance(model, dict):
        return next(iter(model.items()))
    else:
        return model, None


prior_distribution_classes = dict(
    lognormal=LogNormal,
    gamma=Gamma,
    normal=Normal,
    dirichlet=Dirichlet,
    beta=Beta,
    exponential=Exponential,
)

RELAXED_CLOCK_MODELS = {"relaxed_lognormal"}


class PhyloModel:
    """Class to represent the configuration of a basic phylogenetic model"""

    def __init__(self, model_dict: tp.Dict[str, tp.Union[str, tp.Dict[str, object]]]):
        self.tree_model, self.tree_params = parse_model(model_dict["tree"])
        self.clock_model, self.clock_params = parse_model(model_dict["clock"])
        self.subst_model, self.subst_params = parse_model(model_dict["substitution"])
        self.site_model, self.site_params = parse_model(model_dict.get("site", "none"))

    def all_params(
        self,
    ) -> tp.Dict[str, object]:
        return {
            key: value
            for comp_params in [
                self.tree_params,
                self.clock_params,
                self.subst_params,
                self.site_params,
            ]
            if comp_params is not None
            for key, value in comp_params.items()
        }

    def free_params(self) -> tp.Dict[str, tp.Dict[str, object]]:
        res = {
            key: value
            for key, value in self.all_params().items()
            if isinstance(value, dict)
        }
        return res

    def relaxed_clock(self) -> bool:
        return self.clock_model in RELAXED_CLOCK_MODELS


def is_scalar_or_vector_of_type(value: object, numeric_type: tp.Type):
    return isinstance(value, numeric_type) or (
        isinstance(value, tp.Iterable) and isinstance(next(iter(value)), numeric_type)
    )


def constant(value: object) -> tp.Union[tf.Tensor, object]:
    if is_scalar_or_vector_of_type(value, float):
        return tf.constant(value, dtype=treeflow.DEFAULT_FLOAT_DTYPE_TF)
    elif is_scalar_or_vector_of_type(value, integer):
        return tf.constant(value)
    else:
        return value


def get_prior(
    var_name: str, dist_name: str, params: tp.Dict[str, tp.Union[tf.Tensor, object]]
) -> Distribution:
    dist_class = prior_distribution_classes.get(dist_name, None)
    if dist_class is None:
        raise ValueError(f"Unknown prior distribution for {var_name}: {dist_name}")
    else:
        return dist_class(**params, name=var_name)


def get_params(
    params: tp.Optional[tp.Dict[str, tp.Union[object, tp.Dict[str, object]]]]
) -> tp.Generator[
    JointDistributionCoroutine.Root,
    tf.Tensor,
    tp.Tuple[tp.Dict[str, tp.Union[tf.Tensor, object]], bool],
]:
    """
    Get parameters for part of the model.
    Builds prior distributions or converts literals to Tensor constants.

    Returns
    -------
    dict
        Dictionary mapping parameter names to values
    has_root
        Whether one of the parameters is a root node
    """
    out_params: tp.Dict[str, tp.Union[tf.Tensor, object]] = {}
    has_root = False
    if params is not None:
        for name, value in params.items():
            if isinstance(value, dict):
                prior, prior_params = next(iter(value.items()))
                assert isinstance(prior_params, dict)
                prior_params = {
                    key: constant(value) for key, value in prior_params.items()
                }
                out_params[name] = yield JointDistributionCoroutine.Root(
                    get_prior(name, prior, prior_params)
                )
                has_root = True
            else:
                out_params[name] = constant(value)

    return out_params, has_root


DEFAULT_TREE_VAR_NAME = "tree"


def wrap_in_root_if_needed(
    dist: Distribution, has_root_param: bool
) -> tp.Union[Distribution, JointDistributionCoroutine.Root]:
    if has_root_param:
        return dist
    else:
        return JointDistributionCoroutine.Root(dist)


def get_tree_model(  # TODO: Support unrooted trees
    tree_model: str,
    tree_model_params: tp.Dict[str, object],
    has_root_param: bool,
    initial_tree: TensorflowRootedTree,
    var_name: str = DEFAULT_TREE_VAR_NAME,
) -> tp.Generator[
    tp.Union[Distribution, JointDistributionCoroutine.Root],
    TensorflowRootedTree,
    TensorflowRootedTree,
]:
    if tree_model == "fixed":
        tree = initial_tree
    elif tree_model == "coalescent":
        tree = yield wrap_in_root_if_needed(
            ConstantCoalescent(
                initial_tree.taxon_count,
                sampling_times=initial_tree.sampling_times,
                **tree_model_params,
                name=var_name,
            ),
            has_root_param,
        )
    elif tree_model == "birth_death":
        tree = yield wrap_in_root_if_needed(
            BirthDeathContemporarySampling(
                initial_tree.taxon_count,
                name=var_name,
                **tree_model_params,
            ),
            has_root_param,
        )
    elif tree_model == "yule":
        tree = yield wrap_in_root_if_needed(
            Yule(initial_tree.taxon_count, name=var_name, **tree_model_params),
            has_root_param,
        )
    else:
        raise ValueError(f"Unknown tree model for {var_name}: {tree_model}")
    return tree


JC_KEY = "jc"
GTR_KEY = "gtr"
GTR_REL_KEY = "gtr_rel"
subst_model_classes = {JC_KEY: JC, "hky": HKY, GTR_KEY: GTR, GTR_REL_KEY: GTR}


def get_subst_model(
    subst_model: str,
) -> EigendecompositionSubstitutionModel:  # TODO: Support non-eigen substitution models
    subst_model_class = subst_model_classes.get(subst_model, None)
    if subst_model_class is None:
        raise ValueError(f"Unknown substitution model {subst_model}")
    else:
        return subst_model_class()


def get_subst_model_params(
    subst_model: str,
    params: tp.Optional[tp.Dict[str, tp.Union[object, tp.Dict[str, object]]]],
    float_dtype: tf.DType = treeflow.DEFAULT_FLOAT_DTYPE_TF,
    include_jc_frequencies: bool = True,
) -> tp.Generator[
    Distribution, tf.Tensor, tp.Tuple[tp.Dict[str, tp.Union[tf.Tensor, object]], bool]
]:
    processed_params, has_root = yield from get_params(params)
    if include_jc_frequencies and subst_model == JC_KEY:
        processed_params["frequencies"] = JC().frequencies(dtype=float_dtype)
    elif subst_model == GTR_KEY:
        rates = processed_params.pop("gtr_rates")
        processed_params["rates"] = rates
    elif subst_model == GTR_REL_KEY:
        rate_ct = tf.ones_like(processed_params["rate_ac"])
        processed_params["rates"] = tf.stack(
            [
                (rate_ct if pair == "ct" else processed_params.pop(f"rate_{pair}"))
                for pair in GTR_RATE_ORDER
            ],
            axis=-1,
        )
    return processed_params, has_root


def get_strict_clock_rates(clock_rate: tf.Tensor):
    return tf.expand_dims(clock_rate, -1)


def get_relaxed_lognormal_clock_rate_distribution(  # TODO: Think about rate parameterisation
    branch_rate_loc: tf.Tensor,
    branch_rate_scale: tf.Tensor,
    has_root_param: bool,
    initial_tree: TensorflowRootedTree,
) -> Distribution:
    return wrap_in_root_if_needed(
        Sample(
            LogNormal(loc=branch_rate_loc, scale=branch_rate_scale),
            sample_shape=2 * initial_tree.taxon_count - 2,
            name="branch_rates",
        ),
        has_root_param,
    )


def get_clock_model_rates(
    clock_model: str,
    clock_model_params: tp.Dict[str, object],
    has_root_param: bool,
    initial_tree: TensorflowRootedTree,
) -> tp.Generator[Distribution, tf.Tensor, tf.Tensor]:
    if clock_model == "strict":
        return get_strict_clock_rates(**clock_model_params)
    elif clock_model == "relaxed_lognormal":
        rates = yield get_relaxed_lognormal_clock_rate_distribution(
            initial_tree=initial_tree,
            has_root_param=has_root_param,
            **clock_model_params,
        )
        return rates
    else:
        raise ValueError(f"Unknown clock model {clock_model}")


def get_discrete_gamma_site_rate_distribution(
    category_count: tf.Tensor, site_gamma_shape: tf.Tensor
) -> DiscretizedDistribution:
    return DiscretizedDistribution(
        category_count=category_count,
        distribution=Gamma(concentration=site_gamma_shape, rate=site_gamma_shape),
    )


def get_discrete_weibull_site_rate_distribution(
    category_count: tf.Tensor,
    site_weibull_concentration: tf.Tensor,
    site_weibull_scale: tp.Optional[tf.Tensor] = None,
) -> DiscretizedDistribution:
    if site_weibull_scale is None:
        site_weibull_scale = tf.constant(1.0, dtype=site_weibull_concentration.dtype)
    return DiscretizedDistribution(
        category_count=category_count,
        distribution=Weibull(
            concentration=site_weibull_concentration, scale=site_weibull_scale
        ),
    )


def get_site_rate_distribution(
    site_model, site_model_params
) -> FiniteDiscreteDistribution:
    if site_model == "discrete_gamma":
        return get_discrete_gamma_site_rate_distribution(**site_model_params)
    elif site_model == "discrete_weibull":
        return get_discrete_weibull_site_rate_distribution(**site_model_params)
    else:
        raise ValueError(f"Unknown site model {site_model}")


DEFAULT_ALIGNMENT_VAR_NAME = "alignment"


def get_sequence_distribution(  # TODO: Consider case where sequence is root?
    alignment: Alignment,
    tree: TensorflowRootedTree,
    subst_model: EigendecompositionSubstitutionModel,
    subst_model_params: tp.Dict[str, tf.Tensor],
    site_model: str,
    site_model_params: tp.Dict[str, object],
    clock_model_rates: tf.Tensor,
    pattern_counts: tp.Optional[tf.Tensor] = None,
) -> Distribution:
    unrooted_tree = tree.get_unrooted_tree()
    scaled_tree = unrooted_tree.with_branch_lengths(
        unrooted_tree.branch_lengths * clock_model_rates
    )
    if site_model == "none":
        assert len(site_model_params) == 0
        transition_probs_tree = get_transition_probabilities_tree(
            scaled_tree,
            subst_model,
            **subst_model_params,
        )
        single_site_distribution = LeafCTMC(
            transition_probs_tree, subst_model_params["frequencies"]
        )
    else:
        site_rate_distribution = get_site_rate_distribution(
            site_model, site_model_params
        )
        transition_probs_tree = get_transition_probabilities_tree(
            scaled_tree,
            subst_model,
            rate_categories=site_rate_distribution.normalised_support,
            **subst_model_params,
        )
        single_site_distribution = DiscreteParameterMixture(
            site_rate_distribution,
            LeafCTMC(
                transition_probs_tree,
                tf.expand_dims(subst_model_params["frequencies"], -2),
            ),
        )
    if pattern_counts is None:
        return Sample(
            single_site_distribution,
            sample_shape=alignment.site_count,
            name=DEFAULT_ALIGNMENT_VAR_NAME,
        )
    else:
        return SampleWeighted(
            single_site_distribution,
            sample_shape=alignment.site_count,
            name=DEFAULT_ALIGNMENT_VAR_NAME,
            weights=pattern_counts,
        )


def phylo_model_to_joint_distribution(
    model: PhyloModel,
    initial_tree: TensorflowRootedTree,
    initial_alignment: Alignment,
    pattern_counts: tp.Optional[tf.Tensor] = None,
) -> JointDistributionCoroutine:
    def model_fn() -> tp.Generator[Distribution, tf.Tensor, None]:
        tree_model_params, tree_has_root_param = yield from get_params(
            model.tree_params
        )
        tree = yield from get_tree_model(
            model.tree_model,
            tree_model_params=tree_model_params,
            has_root_param=tree_has_root_param,
            initial_tree=initial_tree,
        )

        subst_model_params, _ = yield from get_subst_model_params(
            model.subst_model, model.subst_params
        )
        subst_model = get_subst_model(model.subst_model)

        clock_model_params, clock_has_root_param = yield from get_params(
            model.clock_params
        )
        rates = yield from get_clock_model_rates(
            model.clock_model, clock_model_params, clock_has_root_param, initial_tree
        )

        site_model_params, _ = yield from get_params(model.site_params)
        alignment = yield get_sequence_distribution(
            initial_alignment,
            tree,
            subst_model,
            subst_model_params,
            model.site_model,
            site_model_params,
            rates,
            pattern_counts=pattern_counts,
        )

    return JointDistributionCoroutine(model_fn)
