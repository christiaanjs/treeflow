"""End-to-end CLI test for treeflow_dta_hmc."""
import textwrap

import pandas as pd
import pytest
from click.testing import CliRunner

from treeflow.cli.dta_hmc import treeflow_dta_hmc

pytestmark = pytest.mark.cli


@pytest.fixture
def tiny_dta_inputs(tmp_path):
    tree_path = tmp_path / "tree.nwk"
    tree_path.write_text("((mars:0.1,saturn:0.1):0.2,jupiter:0.3);\n")

    traits_path = tmp_path / "traits.csv"
    traits_path.write_text(
        textwrap.dedent(
            """\
            taxon,trait
            mars,A
            saturn,B
            jupiter,C
            """
        )
    )

    model_path = tmp_path / "model.yaml"
    model_path.write_text(
        textwrap.dedent(
            """\
            tree: fixed
            clock:
              strict:
                clock_rate: 0.5
            substitution:
              discrete_trait:
                n_states: 3
                frequencies:
                  dirichlet:
                    concentration: [1.0, 1.0, 1.0]
                rates:
                  dirichlet:
                    concentration: [1.0, 1.0, 1.0]
            site: none
            """
        )
    )

    return tree_path, traits_path, model_path


def test_dta_hmc_cli_end_to_end(tiny_dta_inputs, tmp_path):
    tree_path, traits_path, model_path = tiny_dta_inputs
    samples_output = tmp_path / "samples.csv"

    runner = CliRunner()
    num_results = 10
    res = runner.invoke(
        treeflow_dta_hmc,
        [
            "-r", str(traits_path),
            "-t", str(tree_path),
            "-m", str(model_path),
            "-n", str(num_results),
            "--num-burnin-steps", "3",
            "--num-adaptation-steps", "3",
            "--num-leapfrog-steps", "3",
            "--kernel", "nuts",
            "--samples-output", str(samples_output),
            "--seed", "42",
        ],
        catch_exceptions=False,
    )
    assert res.exit_code == 0, res.stdout
    assert samples_output.exists()
    samples = pd.read_csv(samples_output)
    assert samples.shape[0] == num_results


def test_dta_hmc_cli_mismatched_n_states(tiny_dta_inputs, tmp_path):
    """Model declares 4 states but traits only observe 3: CLI must refuse."""
    tree_path, traits_path, _ = tiny_dta_inputs
    bad_model_path = tmp_path / "model_bad.yaml"
    bad_model_path.write_text(
        textwrap.dedent(
            """\
            tree: fixed
            clock:
              strict:
                clock_rate: 0.5
            substitution:
              discrete_trait:
                n_states: 4
                frequencies:
                  dirichlet:
                    concentration: [1.0, 1.0, 1.0, 1.0]
                rates:
                  dirichlet:
                    concentration: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            site: none
            """
        )
    )

    runner = CliRunner()
    res = runner.invoke(
        treeflow_dta_hmc,
        [
            "-r", str(traits_path),
            "-t", str(tree_path),
            "-m", str(bad_model_path),
            "-n", "2",
            "--num-burnin-steps", "1",
        ],
        catch_exceptions=False,
    )
    assert res.exit_code != 0
    assert "n_states" in res.output
