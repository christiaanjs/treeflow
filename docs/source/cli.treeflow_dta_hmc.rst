Discrete-trait HMC CLI
=======================

``treeflow_dta_hmc`` runs fixed-topology Hamiltonian Monte Carlo (or NUTS) on a
discrete-trait evolutionary model (e.g. Bayesian phylogeography of
:cite:`lemey2009bayesian`-style migration-rate estimation) given a fixed
time-tree and a two-column CSV of tip trait labels.

Inputs
------

* ``-t / --topology`` — Newick tree file, branch lengths in time units.
* ``-r / --traits`` — CSV with columns ``taxon,trait`` (column names are
  configurable via ``--taxon-column`` and ``--trait-column``). Unknown states
  can be marked with ``?``, ``-``, ``NA``, ``N/A``, or an empty string and
  are treated as missing (flat partials).
* ``-m / --model-file`` — YAML model file declaring priors on the
  exchangeability rates and equilibrium frequencies. See
  :doc:`model-definition` for the ``discrete_trait`` block format.

Minimal YAML model
------------------

.. code-block:: yaml

    tree: fixed
    clock:
      strict:
        clock_rate: 0.5           # mean migration-rate scalar
    substitution:
      discrete_trait:
        n_states: 5               # K states
        frequencies:
          dirichlet:
            concentration: [1.0, 1.0, 1.0, 1.0, 1.0]
        rates:                    # K*(K-1)/2 = 10 exchangeability rates
          dirichlet:
            concentration: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    site: none

Minimal traits CSV
------------------

.. code-block:: text

    taxon,trait
    seq1,NY
    seq2,HK
    seq3,NZ
    seq4,NY

Example invocation
------------------

.. code-block:: bash

    treeflow_dta_hmc \
      -t tree.nwk \
      -r traits.csv \
      -m model.yaml \
      -n 1000 \
      --num-burnin-steps 500 \
      --kernel nuts \
      --samples-output posterior.csv

Outputs
-------

* ``--samples-output`` — CSV of posterior samples for ``frequencies``,
  ``rates``, and any other free parameters declared in the YAML model file.

Full CLI reference
------------------

.. click:: treeflow.cli.dta_hmc:treeflow_dta_hmc
   :prog: treeflow_dta_hmc
   :nested: full
