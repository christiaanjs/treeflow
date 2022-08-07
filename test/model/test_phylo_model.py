import pytest
from tensorflow_probability.python.distributions import Sample
from treeflow.model.phylo_model import phylo_model_to_joint_distribution, PhyloModel

model_dicts_and_keys = [
    (
        dict(
            tree=dict(
                birth_death=dict(
                    birth_diff_rate=dict(lognormal=dict(loc=0.05, scale=0.1)),
                    relative_death_rate=dict(lognormal=dict(loc=0.1, scale=0.2)),
                    sample_probability=dict(
                        beta=dict(concentration1=1.0, concentration0=5.0)
                    ),
                ),
            ),
            clock=dict(strict=dict(clock_rate=1e-3)),
            substitution=dict(
                gtr=dict(
                    frequencies=dict(
                        dirichlet=dict(
                            concentration=[4.0, 4.0, 4.0, 4.0],
                        ),
                    ),
                    gtr_rates=dict(
                        dirichlet=dict(
                            concentration=[2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                        )
                    ),
                )
            ),
            site="none",
        ),
        {
            "tree",
            "birth_diff_rate",
            "relative_death_rate",
            "sample_probability",
            "frequencies",
            "gtr_rates",
            "alignment",
        },
    ),
    (
        dict(
            tree=dict(
                coalescent=dict(pop_size=dict(lognormal=dict(loc=0.05, scale=0.1))),
            ),
            clock=dict(
                relaxed_lognormal=dict(
                    branch_rate_loc=dict(lognormal=dict(loc=0.1, scale=0.5)),
                    branch_rate_scale=dict(gamma=dict(concentration=2.0, rate=1.5)),
                )
            ),
            substitution=dict(
                hky=dict(
                    frequencies=dict(
                        dirichlet=dict(
                            concentration=[4.0, 4.0, 4.0, 4.0],
                        ),
                    ),
                    kappa=dict(lognormal=dict(loc=1, scale=0.8)),
                )
            ),
            site=dict(
                discrete_weibull=dict(
                    site_weibull_concentration=dict(
                        gamma=dict(concentration=2.0, rate=3.0)
                    ),
                    site_weibull_scale=1.0,
                    category_count=4,
                )
            ),
        ),
        {
            "tree",
            "pop_size",
            "branch_rates",
            "branch_rate_loc",
            "branch_rate_scale",
            "frequencies",
            "kappa",
            "site_weibull_concentration",
            "alignment",
        },
    ),
]


@pytest.fixture(params=model_dicts_and_keys)
def model_dict_and_keys(request):
    return request.param


@pytest.mark.parametrize("sample_shape", [(), (1,), (2, 3)])
def test_phylo_model_to_joint_distribution_sample(
    model_dict_and_keys, hello_tensor_tree, hello_alignment, sample_shape
):
    model_dict, expected_keys = model_dict_and_keys
    model = PhyloModel(model_dict)
    dist = phylo_model_to_joint_distribution(model, hello_tensor_tree, hello_alignment)
    sample = dist.sample(sample_shape)
    assert set(sample._asdict().keys()) == expected_keys


def test_get_site_rate_distribution_gamma_site_model_rates(
    hello_tensor_tree, hello_alignment
):
    site_gamma_shape = 0.3
    model_dict = dict(
        tree=dict(
            coalescent=1.0,
        ),
        clock=dict(strict=dict(clock_rate=1e-3)),
        substitution="jc",
        site=dict(
            discrete_gamma=dict(
                site_gamma_shape=site_gamma_shape,
                category_count=4,
            )
        ),
    )
    model = PhyloModel(model_dict)
    dist = phylo_model_to_joint_distribution(model, hello_tensor_tree, hello_alignment)
    sequence_dist: Sample = list(dist._get_single_sample_distributions())[-1]
    transition_probs_tree = (
        sequence_dist.distribution._components_distribution.transition_probs_tree
    )
    assert False
