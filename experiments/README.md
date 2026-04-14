# Experiments

Jupyter notebooks that validate and explore treeflow functionality.

## Installation

Notebooks in this directory require extra dependencies beyond the core package.
Install them with the `experiments` extra:

```bash
pip install -e ".[experiments]"
```

## Notebooks

| Notebook | Description |
|---|---|
| `validate_hmc_birth_death.ipynb` | Validates `fit_fixed_topology_hmc` (NUTS) against direct simulation from `BirthDeathContemporarySampling` using KS tests on all node-height marginals. |

## Running a notebook

```bash
# Interactively
jupyter notebook experiments/<notebook>.ipynb

# Non-interactively (saves output to a new file)
jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=600 \
    --output experiments/<notebook>_executed.ipynb \
    experiments/<notebook>.ipynb
```
