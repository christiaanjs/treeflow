# Model definition format

For an example model definition file, see [`examples/h3n2-model.yaml`](https://github.com/christiaanjs/treeflow/blob/master/examples/h3n2-model.yaml).

TreeFlow's command line interfaces use a YAML model definition format. Each model definition file has four sections:

```yaml
tree:
    ...
clock:
    ...
site:
    ...
substitution:
    ...
```

Each section takes a YAML mapping from the name for the selection of that model component to its parameters.

Each parameter can be a fixed value, or a prior distribution if the parameter is to be estimated.

Prior distributions are specified as mappings from a distribution name to parameters.

For example:

```yaml
substitution:
    hky:
        kappa:
            2.0
        frequencies:
            dirichlet:
                concentration: [2.0, 2.0, 2.0, 2.0]

```

Documentation on all the options is still to come, see the source code at [`treeflow.model.phylo_model.phylo_model_to_joint_distribution`](https://github.com/christiaanjs/treeflow/blob/master/treeflow/model/phylo_model.py) for reference for now.

## Prior distributions

For parameters, see the corresponding TensorFlow Probability distribution.

* `normal`
* `lognormal`
* `gamma`
* `exponential`
* `beta`
* `dirichlet`

## Discrete-trait substitution model

For fixed-tree discrete-trait analyses (e.g. Bayesian phylogeography and
migration-rate estimation), TreeFlow provides a K-state time-reversible
substitution model following [Lemey et al. (2009)](https://doi.org/10.1371/journal.pcbi.1000520).
Use the `discrete_trait` substitution block together with the
[`treeflow_dta_hmc`](cli.treeflow_dta_hmc) CLI.

The block requires:

* `n_states`: the integer number of discrete states K.
* `frequencies`: a `K`-vector of equilibrium state frequencies (typically a
  Dirichlet prior on the `K`-simplex).
* `rates`: the `K*(K-1)/2` symmetric exchangeability rates in row-major
  upper-triangular order: `(0,1), (0,2), ..., (0,K-1), (1,2), ...`. For
  reversible models the rate matrix is normalised so that its trace equals
  one, so only the *relative* rates are identifiable; the overall migration
  rate scale is absorbed into `clock.strict.clock_rate`.

Example for K = 5 states:

```yaml
tree: fixed
clock:
  strict:
    clock_rate:
      lognormal:
        loc: 0.0
        scale: 1.0
substitution:
  discrete_trait:
    n_states: 5
    frequencies:
      dirichlet:
        concentration: [1.0, 1.0, 1.0, 1.0, 1.0]
    rates:
      dirichlet:
        concentration: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
site: none
```

The tip data format for this model is a two-column CSV of `(taxon, trait)`
labels rather than a sequence alignment; see
[`treeflow_dta_hmc`](cli.treeflow_dta_hmc) for the CLI that consumes it.