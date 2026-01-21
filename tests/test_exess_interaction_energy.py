import sys
from pathlib import Path

from rush import Topology, exess
from rush.client import RunOpts, set_opts


def test_exess_interaction_energy():
    set_opts(workspace_dir=Path.cwd() / "test-runs")
    data_dir = Path.cwd() / "tests" / "data"
    topology = Topology.from_json(data_dir / "tyk2_ejm_31_t.json")
    lig_idx = 93
    frag_idcs = topology.get_fragments_near_fragment(lig_idx, 6.0) + [lig_idx]
    res = exess.interaction_energy(
        data_dir / "tyk2_ejm_31_t.json",
        lig_idx,
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
            tags=["rush-py", "test", "tyk2+ejm-31"],
        ),
        collect=True,
    )
    print(res, file=sys.stderr)
    exess.save_energy_outputs(res)


if __name__ == "__main__":
    test_exess_interaction_energy()
