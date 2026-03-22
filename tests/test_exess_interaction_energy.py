import sys
from pathlib import Path

from rush import Topology, exess
from rush.client import RunOpts, set_opts
from rush.exess import interaction_energy
from rush.mol import FragmentRef


def test_exess_interaction_energy():
    set_opts(workspace_dir=Path.cwd() / "test-runs")
    data_dir = Path.cwd() / "tests" / "data"
    topology = Topology.from_json(data_dir / "tyk2_ejm_31_t.json")
    lig_idx = 93
    frag_idcs = topology.get_fragments_near_fragment(lig_idx, 6.0) + [lig_idx]
    run = interaction_energy(
        data_dir / "tyk2_ejm_31_t.json",
        lig_idx,
        basis="PCSeg-0",
        frag_keywords=exess.FragKeywords(
            level="Trimer",
            dimer_cutoff=5.0,
            trimer_cutoff=1.0,
            cutoff_type="Centroid",
            distance_metric="Min",
            included_fragments=frag_idcs,
        ),
        run_opts=RunOpts(
            name="Rush-Py Test EXESS Energy 02: Interaction Energy w/ Frag Keywords",
            tags=["rush-py", "test", "tyk2+ejm-31", "deploy"],
        ),
    )
    result = run.collect()
    print(result, file=sys.stderr)
    fetched = result.fetch()
    assert fetched.calc.qmmbe is not None
    assert fetched.calc.qmmbe.reference_fragment == FragmentRef(lig_idx)
    assert fetched.calc.qmmbe.nmers[0][0].fragments == [FragmentRef(lig_idx)]


if __name__ == "__main__":
    test_exess_interaction_energy()
