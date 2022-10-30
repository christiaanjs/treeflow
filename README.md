# TreeFlow

TreeFlow is a library for phylogenetic modelling and inference based on [TensorFlow Probability](https://www.tensorflow.org/probability) (TFP).

## Installation

1. (Optional) Build and install [`bito`](https://github.com/phylovi/bito) for accelerated computations - not used in CLI
2. Clone this repository and navigate to the cloned directory
3. `pip install -r requirements.txt`
4. `pip install -e .`

## Usage

### Python API

See the TFP documentation for examples of probabilistic programming and inference.
See [`examples/carnivores.ipynb`](examples/carnivores.ipynb) for a TreeFlow example.

### CLI

TreeFlow has a command line interface for variational inference on a given tree topology with standard phylogenetic models. See [`docs/model-definition.md`](docs/model-definition.md) for documentation of the YAML model definition file format.

```
treeflow_vi --help

Usage: treeflow_vi [OPTIONS]

Options:
  -i, --input PATH                Alignment file (FASTA format)  [required]
  -t, --topology PATH             Topology file  [required]
  -m, --model-file PATH           YAML model definition file
  -n, --num-steps INTEGER         Number of VI iterations  [default: 40000;
                                  required]
  -o, --optimizer [adam|robust_adam]
                                  [required]
  --init-values TEXT              Initial values in the format 'scalar_paramet
                                  er=value1,vector_parameter=value2a|value2b'
  -s, --seed INTEGER
  --trace-output PATH             Path to save pickled optimization trace
  --samples-output PATH           Path to save parameter samples in CSV format
  --tree-samples-output PATH
  --n-output-samples INTEGER      Number of samples to use for outputs
                                  [default: 200; required]
  -r, --learning-rate FLOAT       [default: 0.001; required]
  -c, --convergence-criterion [nonfinite]
  --elbo-samples INTEGER RANGE    Number of samples to use in displayed
                                  estimate of evidence lower bound  [default:
                                  100; x>=1; required]
  --progress-bar / --no-progress-bar
  --subnewick-format INTEGER      Subnewick format (see `ete3.Tree`)
                                  [default: 0; required]
  --help                          Show this message and exit.
```