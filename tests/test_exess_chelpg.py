import json
import sys
import tempfile
from pathlib import Path
from pprint import pp

from rush import exess, from_json
from rush.client import RunOpts, fetch_object, set_opts
from rush.exess import exess_energy


def test_exess_energy_chelpg_1hsg_MK1():
    set_opts(workspace_dir=Path.cwd() / "test-runs")
    data_dir = Path(__file__).parent / "data"
    with (data_dir / "1hsg_MK1_trc.json").open() as f:
        trc = from_json(json.load(f)[0])

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
        json.dump(trc.topology.to_json(), tf)
        topology_path = tf.name

    res = exess_energy(
        topology_path,
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
    charges = json.loads(fetch_object(res[1]["Json"]["path"]))["chelpg_charges"]
    pp(charges, width=130, compact=True, stream=sys.stderr)


def test_exess_energy_chelpg_benzene():
    set_opts(workspace_dir=Path.cwd() / "test-runs")
    data_dir = Path(__file__).parent / "data"
    res = exess_energy(
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
    charges = json.loads(fetch_object(res[1]["Json"]["path"]))["chelpg_charges"]
    pp(charges, width=130, compact=True, stream=sys.stderr)


if __name__ == "__main__":
    test_exess_energy_chelpg_1hsg_MK1()
    test_exess_energy_chelpg_benzene()
