# Model definition format

For an example model definition file, see [`examples/h3n2-model.yaml`](https://github.com/christiaanjs/treeflow/blob/master/examples/h3n2-model.yaml).

TreeFlow's command line interfaces use a YAML model definition format. Each model definition file has up to four sections:

```yaml
tree: ...
clock: ...
site: # optional
  ...
substitution: ...
```

The `tree`, `clock` and `substitution` sections are required; the `site` section is
optional (if omitted, a single site rate is used).

Each section takes a YAML mapping from the name of the chosen model component to its
parameters. Each parameter can either be a **fixed value**, or a **prior distribution**
if the parameter is to be estimated. Prior distributions are specified as a mapping from a
distribution name to its parameters.

For example, a fixed `kappa` and an estimated set of base `frequencies`:

```yaml
substitution:
  hky:
    kappa: 2.0
    frequencies:
      dirichlet:
        concentration: [2.0, 2.0, 2.0, 2.0]
```

The reference implementation is
[`treeflow.model.phylo_model.phylo_model_to_joint_distribution`](https://github.com/christiaanjs/treeflow/blob/master/treeflow/model/phylo_model.py).

## Tree models (`tree`)

| Name          | Description                                         | Parameters                                                                               |
| ------------- | --------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `fixed`       | Uses the supplied input tree as-is (no tree prior). | _(none)_                                                                                 |
| `coalescent`  | Kingman's constant-size coalescent prior.           | `pop_size` (effective population size)                                                   |
| `birth_death` | Birth–death process with contemporaneous sampling.  | `birth_diff_rate`, `relative_death_rate`, `sample_probability` (optional, default `1.0`) |
| `yule`        | Yule (pure-birth) process.                          | `birth_rate`                                                                             |

## Clock models (`clock`)

| Name                | Description                                                  | Parameters                             |
| ------------------- | ------------------------------------------------------------ | -------------------------------------- |
| `strict`            | Single molecular clock rate shared by all branches.          | `clock_rate`                           |
| `relaxed_lognormal` | Uncorrelated log-normal relaxed clock (one rate per branch). | `branch_rate_loc`, `branch_rate_scale` |

## Site rate models (`site`, optional)

| Name               | Description                                             | Parameters                                                                                     |
| ------------------ | ------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `none` (default)   | A single site rate; no rate heterogeneity across sites. | _(none)_                                                                                       |
| `discrete_gamma`   | Discretised Gamma distribution of site rates.           | `category_count`, `site_gamma_shape`                                                           |
| `discrete_weibull` | Discretised Weibull distribution of site rates.         | `category_count`, `site_weibull_concentration`, `site_weibull_scale` (optional, default `1.0`) |

## Substitution models (`substitution`)

| Name      | Description                                                                                                    | Parameters                                                                   |
| --------- | -------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| `jc`      | Jukes–Cantor. Base frequencies are fixed to uniform.                                                           | _(none)_                                                                     |
| `hky`     | HKY85.                                                                                                         | `kappa`, `frequencies`                                                       |
| `gtr`     | General Time Reversible, with a six-element vector of relative rates.                                          | `gtr_rates` (length-6 vector, order `ac, ag, at, cg, ct, gt`), `frequencies` |
| `gtr_rel` | GTR parameterised by five independent relative rates, with `ct` fixed to 1 (used for comparison with BEAST 2). | `rate_ac`, `rate_ag`, `rate_at`, `rate_cg`, `rate_gt`, `frequencies`         |

## Prior distributions

Any parameter can be given a prior by replacing its value with a mapping from a
distribution name to that distribution's parameters (which match the corresponding
TensorFlow Probability distribution). For example `kappa: {lognormal: {loc: 1.0, scale: 1.25}}`.

- `normal`
- `lognormal`
- `gamma`
- `exponential`
- `beta`
- `dirichlet`

## Beyond the YAML format

The YAML format is intended to cover common, standard phylogenetic models as a convenient
input for the command line interfaces; it is not intended to express the full range of
models that can be built with TreeFlow's Python API in combination with the probabilistic
modelling tools provided by TensorFlow Probability.

A demonstration of a richer model implemented with the Python API can be found in
[`examples/carnivores.ipynb`](https://github.com/christiaanjs/treeflow/blob/master/examples/carnivores.ipynb).
This implements an HKY model with a **per-branch ("relaxed") transition/transversion ratio**,
where each branch is given its own independent `kappa`, by broadcasting a per-branch parameter
tensor through the standard substitution machinery. It draws `kappa` with shape `(..., n_branches)` (e.g. via
`tfp.distributions.Sample(tfp.distributions.LogNormal(0.0, 2.0), tree.branch_lengths.shape)`)
and feeds it through `treeflow.evolution.substitution.probabilities.get_transition_probabilities_tree`.
See the [tutorials](tutorials.rst) for more on the Python API.
