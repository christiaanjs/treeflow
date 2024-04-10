# TreeFlow

TreeFlow is a library for phylogenetic modelling and inference based on [TensorFlow Probability](https://www.tensorflow.org/probability) (TFP).

It also includes [command line interfaces](https://treeflow.readthedocs.io/en/latest/cli.html) for fixed-topology phylogenetic inference.

## Documentation

[Online manual: tutorials, API documentation, CLI description](https://treeflow.readthedocs.io/en/latest/)

## Installation and getting started

See [installation instructions](https://treeflow.readthedocs.io/en/latest/installation.html)
* (Optional) Build and install [`bito`](https://github.com/phylovi/bito) for accelerated computations - not used in CLI

## Citation

If you want to cite or read about TreeFlow, please see the paper: 

Christiaan Swanepoel, Mathieu Fourment, Xiang Ji, Hassan Nasif, Marc A Suchard, Frederick A Matsen IV, Alexei Drummond. ["TreeFlow: probabilistic programming and automatic differentiation for phylogenetics"](https://arxiv.org/abs/2211.05220). arXiv preprint arXiv:2211.05220 (2022).

## Unit tests

1. `pip install -r dev/requirements.txt`
2. `pytest`

Note tests for acceleration and the benchmark CLI will fail if the extra dependencies for those components are not installed (and `bito` cannot yet be installed with `pip`)