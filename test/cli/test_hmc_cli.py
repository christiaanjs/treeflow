import pytest
from treeflow.cli.hmc import treeflow_hmc, KERNEL_CHOICES
from click.testing import CliRunner

pytestmark = pytest.mark.cli

# CLI plumbing coverage is dataset-independent and runtime is dominated by TF
# graph tracing, so we exercise the option surface on the tiny ``hello`` dataset
# only. NUTS is the expensive kernel here -- it auto-tunes trajectory length, so
# its cost is highly nonlinear in the sample count and explodes on the relaxed
# clock ``model.yaml`` (minutes per run). We therefore (a) keep a small
# ``num_results``, (b) run NUTS only on the cheap default (strict-clock) model,
# and (c) leave the large ``wnv`` dataset out of the HMC matrix (the VI suite
# carries the large-alignment smoke). Between them these cases still cover both
# kernels, the default and model-file paths, and init values on/off.
_NUM_RESULTS = 4

# (newick, fasta, model_file, kernel, include_init_values)
_HELLO = ("hello.nwk", "hello.fasta")
_CASES = [
    (*_HELLO, None, "hmc", True),
    (*_HELLO, "model.yaml", "hmc", False),
    (*_HELLO, None, "nuts", False),
]
assert set(KERNEL_CHOICES) <= {c[3] for c in _CASES}, "a kernel is uncovered"

_INIT_VALUES = {
    None: "clock_rate=0.01",
    "model.yaml": "pop_size=10",
}


def _case_id(case):
    _, _, model_filename, kernel, include_init_values = case
    return "-".join([
        "hello",
        model_filename.removesuffix(".yaml") if model_filename else "no-model",
        kernel,
        "init" if include_init_values else "no-init",
    ])


@pytest.mark.parametrize(
    "newick_filename,fasta_filename,model_filename,kernel,include_init_values",
    _CASES,
    ids=[_case_id(c) for c in _CASES],
)
def test_hmc(
    test_data_dir,
    samples_output_path,
    tree_samples_output_path,
    newick_filename,
    fasta_filename,
    model_filename,
    kernel,
    include_init_values,
):
    import pandas as pd
    import dendropy

    newick_file = str(test_data_dir / newick_filename)
    fasta_file = str(test_data_dir / fasta_filename)
    model_file = str(test_data_dir / model_filename) if model_filename is not None else None

    runner = CliRunner()
    num_results = _NUM_RESULTS
    args = [
        "-i", fasta_file,
        "-t", newick_file,
        "-n", str(num_results),
        "--num-burnin-steps", str(3),
        "--num-adaptation-steps", str(3),
        "--num-leapfrog-steps", str(3),
        "--kernel", kernel,
        "--samples-output", str(samples_output_path),
        "--tree-samples-output", str(tree_samples_output_path),
    ]
    if model_file is not None:
        args += ["-m", model_file]
    if include_init_values:
        args += ["--init-values", _INIT_VALUES[model_filename]]

    res = runner.invoke(treeflow_hmc, args, catch_exceptions=False)
    assert res.exit_code == 0, res.stdout
    print(res.stdout)

    samples = pd.read_csv(samples_output_path)
    assert samples.shape[0] == num_results

    trees = dendropy.TreeList.get(path=tree_samples_output_path, schema="nexus")
    assert len(trees) == num_results
