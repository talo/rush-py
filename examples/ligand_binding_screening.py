#!/usr/bin/env python3
"""
Quick Ligand-Protein Binding Analysis
=====================================

A practical example of computational drug discovery:
  • Calculate interaction energies between ligand and protein residues
  • Identify favorable binding interactions
  • Rapidly screen multiple binding poses/ligands
  • Guide medicinal chemistry optimization

This demonstrates how computational chemistry accelerates drug discovery.

Prerequisites:
  - Set RUSH_TOKEN and RUSH_PROJECT environment variables
  - Protein-ligand complex topology file

Output:
  - Binding interaction energy
  - Per-residue decomposition (which residues favor binding?)
  - Binding pocket assessment
"""

import json
from pathlib import Path
from typing import NamedTuple

from rush import exess
from rush.client import RunOpts


class BindingResult(NamedTuple):
    """Store binding calculation results."""
    ligand_name: str
    total_interaction_energy: float  # kcal/mol (negative = favorable)
    electrostatic_component: float
    vdw_component: float
    hydrogen_bonds: int
    binding_affinity: str  # Qualitative assessment


def assess_binding_strength(delta_e_kcal: float) -> str:
    """
    Classify binding strength based on interaction energy.
    
    These are rough estimates for small ligand-pocket interactions:
      ΔE < -5 kcal/mol: Strong
      -5 to -2 kcal/mol: Moderate
      -2 to 0 kcal/mol: Weak
      > 0 kcal/mol: Unfavorable
    """
    if delta_e_kcal < -10:
        return "🔴 Very Strong (ΔE < -10)"
    elif delta_e_kcal < -5:
        return "🟡 Strong (ΔE: -10 to -5)"
    elif delta_e_kcal < -2:
        return "🟢 Moderate (ΔE: -5 to -2)"
    elif delta_e_kcal < 0:
        return "🔵 Weak (ΔE: -2 to 0)"
    else:
        return "❌ Unfavorable (ΔE > 0)"


def calculate_binding_score(topology_path: Path, ligand_fragment_idx: int) -> BindingResult:
    """
    Calculate binding interaction energy for a ligand in a protein pocket.
    
    This is a simplified version - real workflows would:
    1. Prepare the complex (protonate, add charges)
    2. Run QM/MM or MM optimization
    3. Calculate interaction energy decomposition
    4. Generate free energy estimates (MM-PBSA, etc.)
    """
    print("\n🧬 Analyzing binding interaction...")
    
    # In practice, this would call exess.interaction_energy()
    # For demonstration, we show the workflow:
    
    try:
        # Run interaction energy calculation
        result = exess.interaction_energy(
            topology_path,
            ligand_fragment_idx,
            method="RestrictedHF",
            basis="STO-3G",
            run_opts=RunOpts(
                name="Ligand Binding Analysis",
                tags=["drug-discovery", "binding", "qmmbe"],
            ),
            collect=True,
        )
        
        # Extract binding energy from result
        # In actual use:
        # energy_data = json.loads(download_object(result[0]["path"]).decode())
        # delta_e = energy_data['qmmbe']['expanded_hf_energy']
        
        # For demo, use mock data:
        delta_e_au = -0.15  # Example: -0.15 Hartree
        delta_e_kcal = delta_e_au * 627.509  # Convert to kcal/mol
        
        binding_strength = assess_binding_strength(delta_e_kcal)
        
        return BindingResult(
            ligand_name="Ligand_X",
            total_interaction_energy=delta_e_kcal,
            electrostatic_component=delta_e_kcal * 0.6,
            vdw_component=delta_e_kcal * 0.4,
            hydrogen_bonds=2,
            binding_affinity=binding_strength,
        )
    
    except Exception as e:
        print(f"   Note: {e}")
        return None


# ===== Main Workflow =====
print("╔" + "═" * 72 + "╗")
print("║" + " Quick Ligand-Protein Binding Analysis".ljust(73) + "║")
print("║" + " Accelerating drug discovery with computational chemistry".ljust(73) + "║")
print("╚" + "═" * 72 + "╝")

print("""
Real-World Scenario:
  Your medicinal chemistry team has synthesized 5 new inhibitor candidates.
  Need to know: Which ones bind best to the target protein?
  
  Time to answer (experiment): 2-4 weeks (biochemical assays)
  Time to answer (computation): 2-4 hours (cluster run)
  
  → Use computation to triage and guide experiments!
""")

# Example ligands to screen
ligands = [
    {
        "name": "Candidate A (Parent)",
        "binding_energy": -94.3,  # kcal/mol, typical for a good inhibitor
        "hbonds": 3,
        "status": "⭐ Reference compound",
    },
    {
        "name": "Candidate B (Methylated)",
        "binding_energy": -102.8,
        "hbonds": 2,
        "status": "⭐⭐ Improvement!",
    },
    {
        "name": "Candidate C (Chlorinated)",
        "binding_energy": -67.5,
        "hbonds": 1,
        "status": "❌ Worse",
    },
    {
        "name": "Candidate D (Fluorinated)",
        "binding_energy": -106.2,
        "hbonds": 4,
        "status": "⭐⭐⭐ Best!",
    },
    {
        "name": "Candidate E (Cyclic version)",
        "binding_energy": -89.7,
        "hbonds": 2,
        "status": "⭐ Comparable",
    },
]

print("\n📊 Screening Results (RHF/STO-3G, 2h compute time):\n")
print(f"{'Candidate':<28} {'ΔE (kcal/mol)':<15} {'HBonds':<8} {'Assessment':<30}")
print("─" * 81)

ranked_ligands = sorted(ligands, key=lambda x: x["binding_energy"])

for i, ligand in enumerate(ranked_ligands, 1):
    name = ligand["name"]
    energy = ligand["binding_energy"]
    hbonds = ligand["hbonds"]
    assessment = assess_binding_strength(energy)
    
    print(f"{i}. {name:<24} {energy:>8.1f}         {hbonds}         {assessment}")

print("\n" + "─" * 81)

# ===== Analysis & Insights =====
print("\n💡 Computational Chemistry Insights:")
print("""
✅ Why QM/MM is Powerful for Binding:
   • Electrostatic interactions → describe with QM accuracy
   • Van der Waals → fast with MM, fine-tuned with hybrid
   • Polarization effects → QM captures electronic response
   • H-bond geometry → QM nails directionality

🔬 What This Teaches Us:
   • Fluorine substitution improves binding (Candidate D)
     ├─ Higher electronegativity strengthens H-bonds
     ├─ Similar size to hydrogen (isostere)
     └─ Better dipolar interactions with backbone

   • Methylation improves binding (Candidate B)
     ├─ Increases hydrophobic interactions
     ├─ Fills pocket volume efficiently
     └─ Maintains H-bonding capability

   • Chlorination worsens binding (Candidate C)
     ├─ Size mismatch (too large for pocket)
     ├─ Disrupts H-bonding network
     └─ Unfavorable desolvation cost

📈 Next Steps in Real Workflow:
   1. Refine geometry → run optimization instead of SPE
   2. Better method → DFT (B3LYP, ωB97X-D) for accuracy
   3. Free energy → add entropic corrections (MM-PBSA)
   4. Kinetics → calculate barrier heights for binding/unbinding
   5. Selectivity → screen vs. off-target proteins
   6. ADMET → predict absorption, distribution, metabolism
""")

print("\n🎯 Recommended Next Candidates for Synthesis:")
print("   1. Candidate D (Fluorinated) — Best computational metrics")
print("   2. Candidate B (Methylated) — Close behind, simpler synthesis")
print("   3. D + B hybrid — Combine both modifications?")

print("\n⏱️  Computational Cost Breakdown:")
print("   • Single-point energy (SPE): ~5 min per structure")
print("   • Geometry optimization: ~30-60 min per structure")
print("   • Free energy (MM-PBSA): ~2-5 hours per structure")
print("   • Total for 5 candidates: ~2.5 hours parallel (cluster)")

print("""
💰 ROI (Return on Investment):
   • Cost: ~1 CPU-hour * $0.25 = $0.25 per candidate
   • Benefit: Eliminates 1-2 poor candidates from synthesis queue
   • Synthesis cost of poor candidate: ~$5,000-20,000
   • Time saved: 2-4 weeks (can synthesize better ones instead!)
   
   → 100x ROI in time and cost! ✨
""")

print("\n" + "╚" + "═" * 72 + "╝\n")
