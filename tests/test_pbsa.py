from pathlib import Path

import pytest

from rush import pbsa
from rush import RunOpts
from tests._module_test_utils import assert_run_collects_and_caches


def test_pbsa(test_data_dir: Path):
    run = pbsa.solvation_energy(
        test_data_dir / "ethane_t.json",
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
    )
    assert_run_collects_and_caches(run, pbsa.ResultRef)

    result = run.fetch()
    assert isinstance(result, pbsa.Result)
    assert result.solvation_energy == pytest.approx(
        result.polar_solvation_energy + result.nonpolar_solvation_energy
    )

    saved = run.save()
    assert isinstance(saved, pbsa.ResultPaths)
    assert saved.output.suffix == ".json"
    assert saved.output.exists()
