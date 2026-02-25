import json
import sys
from pathlib import Path
from pprint import pp

from rush import exess
from rush.client import RunOpts, download_object, set_opts


def test_exess_energy_chelpg_tyk2_ejm_31():
    set_opts(workspace_dir=Path.cwd() / "test-runs")
    data_dir = Path(__file__).parent / "data"
    res = exess.energy(
        data_dir / "tyk2_ejm_31_t.json",
        basis="PCSeg-0",
        frag_keywords=None,  # Important, to disable fragmentation
        export_keywords=exess.ExportKeywords(export_chelpg_charges=True),
        convert_hdf5_to_json=True,
        run_opts=RunOpts(
            name="Rush-Py Test EXESS Energy 03.1: ChelpG via Energy",
            tags=["rush-py", "test", "tyk2+ejm-31"],
        ),
        collect=True,
    )
    print(res, file=sys.stderr)
    charges = json.loads(download_object(res[1]["Json"]["path"]))["chelpg_charges"]
    pp(charges, width=130, compact=True, stream=sys.stderr)


def test_exess_energy_chelpg_benzene():
    set_opts(workspace_dir=Path.cwd() / "test-runs")
    data_dir = Path(__file__).parent / "data"
    res = exess.energy(
        data_dir / "benzene_t.json",
        method="RestrictedRIMP2",
        basis="def2-TZVP",
        aux_basis="def2-TZVP-RIFIT",
        scf_keywords=exess.SCFKeywords(fock_build_type="RI"),
        frag_keywords=None,  # Important, to disable fragmentation
        export_keywords=exess.ExportKeywords(export_chelpg_charges=True),
        convert_hdf5_to_json=True,
        run_opts=RunOpts(
            name="Rush-Py Test EXESS Energy 03.2: ChelpG via Energy",
            tags=["rush-py", "test", "benzene"],
        ),
        collect=True,
    )
    print(res, file=sys.stderr)
    charges = json.loads(download_object(res[1]["Json"]["path"]))["chelpg_charges"]
    pp(charges, width=130, compact=True, stream=sys.stderr)


if __name__ == "__main__":
    test_exess_energy_chelpg_tyk2_ejm_31()
    test_exess_energy_chelpg_benzene()
