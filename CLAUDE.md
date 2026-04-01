# CLAUDE.md

## Repository Overview

Treeflow is a phylogenetics library built on TensorFlow and TensorFlow Probability. It provides differentiable tree operations, evolutionary models, and variational inference for Bayesian phylogenetics.

## Environment Setup (Claude Code Web Session)

Run the following commands at the start of a session before running tests:

```bash
# Upgrade setuptools first to fix legacy package build issues (ete3, silence_tensorflow)
pip install --upgrade setuptools --ignore-installed
pip install -e ".[test]"

# Ensure the pip-installed pytest takes precedence over any uv-tool-installed pytest
if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
  echo 'export PATH="/usr/local/bin:$PATH"' >> "$CLAUDE_ENV_FILE"
fi
```

This installs the package in editable mode with test dependencies (`pandas`, `pytest`, `allpairspy`).

## Running Tests

```bash
pytest
```

The default `pytest.ini` configuration excludes `cli` and `bito` markers. To run specific subsets:

```bash
pytest test/tree/        # tree data structure tests
pytest test/distributions/  # distribution tests
pytest test/evolution/   # evolutionary model tests
pytest -m cli            # CLI end-to-end tests (requires data files)
pytest -m bito           # tests requiring the optional bito package
```

Always use `/usr/local/bin/pytest` (the pip-installed version) if there is ambiguity on PATH.

## Project Structure

```
treeflow/           # Main package
  tree/             # Tree data structures and traversal
  distributions/    # TFP-compatible phylogenetic distributions
  evolution/        # Substitution models and likelihoods
  model/            # Probabilistic model building
  vi/               # Variational inference
  cli/              # Command-line interface (click-based)
  bijectors/        # TFP bijectors for tree topology/branch lengths
  acceleration/     # Optional bito-accelerated likelihoods
test/               # Test suite (mirrors package structure)
  fixtures/         # Shared pytest fixtures (not collected as tests)
  helpers/          # Test helper utilities (not collected as tests)
conftest.py         # Root-level pytest fixtures and plugin registration
pytest.ini          # Pytest configuration
```

## Key Dependencies

- **TensorFlow** >= 2.11 + **TensorFlow Probability** >= 0.19 — core computation
- **ete3** — tree parsing and manipulation
- **dendropy** — additional tree I/O
- **attrs** — data classes for tree nodes
- **click** + **tqdm** — CLI tooling
- **silence_tensorflow** — suppresses TF startup noise

## Notes

- Python 3.9–3.12 supported
- `setuptools` must be upgraded before installation to avoid build failures with `ete3` and `silence_tensorflow` (legacy `setup.py` packages)
- Tests marked `cli` exercise the full CLI pipeline and may require test data files
- Tests marked `bito` require the optional `bito` C++ extension (not installed by default)
