"""
Example: EXESS Interaction Energy

This script demonstrates how to:
1. Compute fragment-based interaction energy between a ligand and its environment
2. Prepare a complex from PDB and run an end-to-end interaction energy calculation

Tutorial: docs/tutorials/exess-interaction-energy.md

Prerequisites:
    - Set RUSH_TOKEN and RUSH_PROJECT environment variables
    - Input files from tests/data/ (tyk2_ejm_31_t.json)
"""

import json
from pathlib import Path

from rush import exess
from rush.client import RunOpts


# ===== Example 1: Fragment-based interaction energy =====
print("=" * 60)
print("Example 1: Fragment-based interaction energy")
print("=" * 60)

DATA_DIR = Path(__file__).parent.parent.parent / "tests" / "data"

out = exess.interaction_energy(
    DATA_DIR / "tyk2_ejm_31_t.json",
    93,  # This is the index of the fragment that contains the ligand
    frag_keywords=exess.FragKeywords(
        level="Trimer",
        dimer_cutoff=10.0,
        trimer_cutoff=5.0,
        cutoff_type="Centroid",
        distance_metric="Min",
    ),
    run_opts=RunOpts(
        name="Tutorial: Interaction Energy Basic",
    ),
    collect=True,
)

out_file, _ = exess.save_energy_outputs(out)
with open(out_file) as f:
    out_data = json.load(f)
print(f"Interaction energy: {out_data['qmmbe']['expanded_hf_energy']}")


# ===== Example 2: End-to-end from PDB =====
print()
print("=" * 60)
print("Example 2: End-to-end interaction energy from PDB")
print("=" * 60)

from rush.prepare_complex import prepare_complex

# Step 1: Prepare the system
trc = prepare_complex(
    Path(__file__).parent.parent.parent / "demos/data/1hsg.pdb",
    ligand_names=["MK1", "HOH"],
    run_opts=RunOpts(name="Tutorial: Interaction Energy E2E - Prepare Complex"),
    collect=True,
)

# Print the charged amino acids
print("Charged amino acids:")
for i, (res_name, formal_charge) in enumerate(
    zip(trc.residues.seqs, trc.topology.fragment_formal_charges)
):
    if int(formal_charge) != 0:
        print(f"{i:>4} {res_name}: {int(formal_charge):+}")

# Step 2: Find ligand fragment index + nearby pocket
lig_idx = trc.residues.seqs.index("MK1")
frag_idcs = trc.topology.get_fragments_near_fragment(lig_idx, 5.0) + [lig_idx]

# Step 3: Write topology and run interaction energy
with open("1hsg_t.json", "w") as f:
    f.write(json.dumps(trc.topology.to_json(), indent=2))

out = exess.interaction_energy(
    "1hsg_t.json",
    lig_idx,
    frag_keywords=exess.FragKeywords(
        level="Dimer",
        included_fragments=frag_idcs,
    ),
    run_opts=RunOpts(name="Tutorial: Interaction Energy E2E - EXESS"),
    collect=True,
)

out_file, _ = exess.save_energy_outputs(out)
with open(out_file) as f:
    out_data = json.load(f)
print(f"Interaction energy: {out_data['qmmbe']['expanded_hf_energy']}")
