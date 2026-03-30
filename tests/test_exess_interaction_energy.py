from pathlib import Path

from rush import FragmentRef, Topology, exess
from rush import RunOpts
from tests._module_test_utils import assert_run_collects_and_caches


def test_exess_interaction_energy(test_data_dir: Path):
    topology = Topology.from_json(test_data_dir / "tyk2_ejm_31_t.json")
    lig_idx = 93
    frag_idcs = topology.get_fragments_near_fragment(lig_idx, 6.0) + [lig_idx]
    run = exess.interaction_energy(
        test_data_dir / "tyk2_ejm_31_t.json",
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
    assert_run_collects_and_caches(run, exess.ResultRef)

    result = run.fetch()
    assert isinstance(result, exess.Result)
    assert result.calc.qmmbe.reference_fragment == FragmentRef(lig_idx)
    assert result.calc.qmmbe.nmers[0][0].fragments == [FragmentRef(lig_idx)]

    saved = run.save()
    assert isinstance(saved, exess.ResultPaths)
    assert saved.calc.exists()
