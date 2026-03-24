import sys
from pathlib import Path
from pprint import pp

from rush import exess, from_json
from rush.client import RunOpts
from rush.exess import energy


def test_exess_energy_chelpg_1hsg_MK1(test_data_dir: Path):
    trc = from_json(test_data_dir / "1hsg_MK1_trc.json")[0]

    result = energy(
        trc,
        basis="PCSeg-0",
        frag_keywords=None,  # Important, to disable fragmentation
        export_keywords=exess.ExportKeywords(export_chelpg_charges=True),
        convert_hdf5_to_json=True,
        run_opts=RunOpts(
            name="Rush-Py Test EXESS Energy 03.1: ChelpG via Energy",
            tags=["rush-py", "test", "tyk2+ejm-31"],
        ),
    ).collect()
    print(result, file=sys.stderr)
    assert result.exports is not None
    fetched = result.fetch()
    assert isinstance(fetched.exports, dict)
    charges = fetched.exports["chelpg_charges"]
    pp(charges, width=130, compact=True, stream=sys.stderr)


def test_exess_energy_chelpg_benzene(test_data_dir: Path):
    result = energy(
        test_data_dir / "benzene_t.json",
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
    ).collect()
    print(result, file=sys.stderr)
    assert result.exports is not None
    fetched = result.fetch()
    assert isinstance(fetched.exports, dict)
    charges = fetched.exports["chelpg_charges"]
    pp(charges, width=130, compact=True, stream=sys.stderr)
