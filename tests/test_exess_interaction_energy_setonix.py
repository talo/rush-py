import sys
from pathlib import Path

import pytest

from rush import Topology, exess
from rush.client import RunOpts, RunSpec
from rush.exess import interaction_energy


@pytest.mark.timeout(1800)
def test_exess_interaction_energy_setonix(test_data_dir: Path):
    topology = Topology.from_json(test_data_dir / "tyk2_ejm_31_t.json")
    lig_idx = 93
    frag_idcs = topology.get_fragments_near_fragment(lig_idx, 6.0) + [lig_idx]
    run = interaction_energy(
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
            name="Rush-Py Test EXESS Interaction Energy: Setonix",
            tags=["rush-py", "test", "tyk2+ejm-31", "setonix"],
        ),
        run_spec=RunSpec(target="Setonix"),
    )
    result = run.collect()
    print(result, file=sys.stderr)
    result.save()
