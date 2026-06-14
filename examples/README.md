# TreeFlow examples

## H3N2

[`h3n2-vi.sh`](h3n2-vi.sh) uses TreeFlow's variational inference command line interface to estimate dates and model parameters on an alignment of 980 influenza genomes, taken from:

> Vaughan, Timothy G., et al. "Efficient Bayesian inference under the structured coalescent." *Bioinformatics* 30.16 (2014): 2272-2279.

It uses the model specified in [`h3n2-model.yaml`](h3n2-model.yaml).

## Rates and dates
[`rates-and-dates.ipynb`](rates-and-dates.ipynb) is a Jupyter notebook that demonstrates TreeFlow's variational inference and model comparison API. We also provide a YAML version of the model definition in [`rates-and-dates-model.yaml`](rates-and-dates.yaml).

The data and model are based on [the BEAST documentation](https://beast.community/rates_and_dates). The original sequences are taken from:

> Bryant, Juliet E., Edward C. Holmes, and Alan D. T. Barrett. "Out of Africa: a molecular perspective on the introduction of yellow fever virus into the Americas." *PLoS Pathogens* 3.5 (2007): e75.

## Carnivores

[`carnivores.ipynb`](carnivores.ipynb) is a Jupyter notebook that shows how TreeFlow's probabilistic modelling API can be used for rapid model development. It investigates variation in the transition-tranversion ratio over lineages.

The dataset is an alignment of mitochondrial DNA sequences from carnivores, [accessed from the BEAST examples](https://github.com/beast-dev/beast-mcmc/blob/v1.10.4/examples/Benchmarks/benchmark2.xml), taken from:

> Suchard, Marc A., and Andrew Rambaut. "Many-core algorithms for statistical phylogenetics." *Bioinformatics* 25.11 (2009): 1370-1376.

## Tree-traversal execution backends

[`tree_traversal_backends_benchmark.ipynb`](tree_traversal_backends_benchmark.ipynb)
benchmarks ways of running a *generic*, differentiable tree traversal — a
`tf.while_loop` under graph mode vs. XLA (`jit_compile`), an unrolled
fixed-topology graph, the hand-written native C++ ops, and an optional JAX
`lax.scan` prototype (which runs only if JAX is installed). It compares them on
simple per-node operations (the node-height ratio transform and the Felsenstein
partial likelihood, both of which have a native op) and on a neural-network
message-passing block whose hidden width is swept. The goal is to decide when a
native kernel is worth hand-writing versus when a compiled generic combinator
(which can host arbitrary differentiable per-node functions, including NN layers)
is the better tool.

