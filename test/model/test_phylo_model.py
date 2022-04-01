import pytest
from treeflow.model.phylo_model import phylo_model_to_joint_distribution, PhyloModel


@pytest.fixture
def model_dict():
    return dict(
        tree=dict(
            coalescent=dict(pop_size=dict(lognormal=dict(loc=0.05, scale=0.1))),
        ),
        clock=dict(
            relaxed_lognormal=dict(
                rate_loc=dict(lognormal=dict(loc=0.1, scale=0.5)),
                rate_scale=dict(gamma=dict(concentration=2.0, rate=1.5)),
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
                concentration=dict(gamma=dict(concentration=2.0, rate=3.0)),
                scale=1.0,
                category_count=4,
            )
        ),
    )


@pytest.mark.parametrize("sample_shape", [(), (1,), (2, 3)])
def test_phylo_model_to_joint_distribution_sample(
    model_dict, hello_tensor_tree, hello_alignment, sample_shape
):
    model = PhyloModel(model_dict)
    dist = phylo_model_to_joint_distribution(model, hello_tensor_tree, hello_alignment)
    sample = dist.sample(sample_shape)
