# TreeFlow model definition format

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