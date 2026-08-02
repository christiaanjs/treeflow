# TreeFlow examples

## Running the notebooks

The notebooks run interactively as usual. To execute one non-interactively with
its variational-inference progress bar streamed live to the terminal (plain
`jupyter nbconvert --execute` hides it until each cell finishes), use the runner
script:

```bash
python run_example.py carnivores            # writes carnivores.executed.ipynb
python run_example.py rates-and-dates       # writes rates-and-dates.executed.ipynb
python run_example.py all --inplace         # run both, overwriting in place
python run_example.py carnivores --timeout 14400
```

For a quick smoke run of the carnivores example, shrink the variational-inference
workload with `--num-steps`, `--n-runs` and `--n-samples` (these set the
`TREEFLOW_EXAMPLE_NUM_STEPS` / `_N_RUNS` / `_N_SAMPLES` environment variables the
notebook reads; unset, it uses its full defaults, so browser runs are unchanged):

```bash
python run_example.py carnivores --num-steps 200 --n-runs 2 --n-samples 40
```

It drives the notebook with an `nbclient` client that forwards cell output as it
arrives, so the (text) `tqdm` bars are visible while the run is in progress. This
mirrors `experiments/run_benchmark.py`.

## H3N2

[`h3n2-vi.sh`](h3n2-vi.sh) uses TreeFlow's variational inference command line interface to estimate dates and model parameters on an alignment of 980 influenza genomes, taken from:

> Vaughan, Timothy G., et al. "Efficient Bayesian inference under the structured coalescent." *Bioinformatics* 30.16 (2014): 2272-2279.

It uses the model specified in [`h3n2-model.yaml`](h3n2-model.yaml).

[`h3n2-vi-multi-run.sh`](h3n2-vi-multi-run.sh) runs the same analysis 4 times (60,000
iterations each) with different seeds, to compare the fitted approximations/ELBOs across
runs. Wall-clock timing for each run is appended to
`demo-out/h3n2-multi-run-timing.csv`.

## Rates and dates
[`rates-and-dates.ipynb`](rates-and-dates.ipynb) is a Jupyter notebook that demonstrates TreeFlow's variational inference and model comparison API. We also provide a YAML version of the model definition in [`rates-and-dates-model.yaml`](rates-and-dates-model.yaml).

The data and model are based on [the BEAST documentation](https://beast.community/rates_and_dates). The original sequences are taken from:

> Bryant, Juliet E., Edward C. Holmes, and Alan D. T. Barrett. "Out of Africa: a molecular perspective on the introduction of yellow fever virus into the Americas." *PLoS Pathogens* 3.5 (2007): e75.

## Carnivores

[`carnivores.ipynb`](carnivores.ipynb) is a Jupyter notebook that shows how TreeFlow's probabilistic modelling API can be used for rapid model development. It investigates variation in the transition-transversion ratio over lineages.

The dataset is an alignment of mitochondrial DNA sequences from carnivores, [accessed from the BEAST examples](https://github.com/beast-dev/beast-mcmc/blob/v1.10.4/examples/Benchmarks/benchmark2.xml), taken from:

> Suchard, Marc A., and Andrew Rambaut. "Many-core algorithms for statistical phylogenetics." *Bioinformatics* 25.11 (2009): 1370-1376.

