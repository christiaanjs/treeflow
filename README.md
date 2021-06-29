# treeflow

Phylogenetics in Tensorflow

## Development installation

1. Build and install [`libsbn`](https://github.com/phylovi/libsbn)
    * Currently depends on branch [`264-ratio-gradient-jacobian`]([https://github.com/phylovi/libsbn/tree/264-ratio-gradient-jacobian])
2. `conda activate libsbn`
3. `pip install -e .` from the repository root

## Test

1. `pip install numdifftools`
2. `pytest`
