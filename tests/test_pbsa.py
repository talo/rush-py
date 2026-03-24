import sys
from pathlib import Path

from rush.client import RunOpts, set_opts
from rush.pbsa import PBSAResult, fetch_outputs, pbsa, save_outputs


def test_pbsa():
    set_opts(workspace_dir=Path.cwd() / "test-runs")
    data_dir = Path.cwd() / "tests" / "data"
    res = pbsa(
        data_dir / "ethane_t.json",
        solute_dielectric=1.0,
        solvent_dielectric=78.54,
        solvent_radius=0.14,
        ion_concentration=0.0,
        temperature=298.0,
        spacing=0.04,
        sasa_gamma=2.26778,
        sasa_beta=3.84928,
        sasa_n_samples=1000,
        convergence=0.00001,
        box_size_factor=2.0,
        run_opts=RunOpts(
            name="Rush-Py Test PBSA 01",
            tags=["rush-py", "test", "ethane"],
        ),
        collect=True,
    )
    print(res, file=sys.stderr)
    fetched = fetch_outputs(res)
    assert isinstance(fetched, PBSAResult)
    print(fetched, file=sys.stderr)
    print(save_outputs(res), file=sys.stderr)


if __name__ == "__main__":
    test_pbsa()
