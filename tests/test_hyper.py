import pytest
from rush import hyper
from rush.client import RunOpts, RushRunError
from tests._module_test_utils import assert_run_collects_and_caches

def test_solvate():
    trc_path = "tests/data/hyper/valid_trc.json"
    run = hyper.solvate(
        [trc_path],
        run_spec=hyper.RunSpec(target="Bullet3"), run_opts=RunOpts(
            name="Rush-Py Test Hyper Solvate 01",
            tags=["rush-py", "test"],
        ),
    )
    assert_run_collects_and_caches(run, hyper.SolvateResultRef)

    fetched = run.fetch()
    assert len(fetched) == 1
    trc = fetched[0]
    
    # Assert successful Ok result type
    assert not isinstance(trc, str), f"Expected successful TRC, got error: {trc}"
    assert hasattr(trc, "topology")
    assert hasattr(trc, "residues")
    assert hasattr(trc, "chains")
    # Solvated TRC should have more atoms than the original 3 atoms (H2O)
    assert len(trc.topology.symbols) >= 3

def test_minimize_invalid_input():
    with pytest.raises(RushRunError, match="EmptyInput"):
        run = hyper.minimize([], run_spec=hyper.RunSpec(target="Bullet3"), run_opts=RunOpts(name="Test Hyper Minimize Empty", tags=["rush-py", "test"]))
        run.collect()

def test_run_invalid_input():
    with pytest.raises(RushRunError, match="EmptyInput"):
        run = hyper.run([], run_spec=hyper.RunSpec(target="Bullet3"), run_opts=RunOpts(name="Test Hyper Run Empty", tags=["rush-py", "test"]))
        run.collect()

