#!/usr/bin/env python3
"""
Dipole Moment & Polarity Analysis
==================================

Demonstrates practical computational chemistry applications:
  • Calculate dipole moments for organic molecules
  • Compare polarity of different conformers
  • Relate electronic structure to molecular properties
  • Extract and visualize key quantum properties

This is a real-world task chemists do: predicting molecular polarity,
understanding solvation, and designing for specific intermolecular interactions.

Prerequisites:
  - Set RUSH_TOKEN and RUSH_PROJECT environment variables
  - Topology files for analysis

Output:
  - Dipole moment magnitudes and vectors
  - Polarity assessment
  - Insights into electronic structure
"""

import json
from pathlib import Path
from typing import Optional

from rush import exess
from rush.client import RunOpts


def calculate_dipole_magnitude(vector: list) -> float:
    """Calculate magnitude of dipole vector (in atomic units)."""
    return sum(x**2 for x in vector) ** 0.5


def assess_polarity(mu: float) -> tuple[str, str]:
    """Classify polarity based on dipole moment magnitude."""
    if mu < 0.5:
        return "Non-polar", "δ- ~0"
    elif mu < 1.5:
        return "Weakly polar", "δ- ~0.1-0.5"
    elif mu < 3.0:
        return "Moderately polar", "δ- ~0.5-1.5"
    else:
        return "Highly polar", "δ- >1.5"


def run_dipole_analysis(topology_path: Path, molecule_name: str) -> Optional[dict]:
    """
    Run single-point energy calculation and extract dipole moment.
    
    Args:
        topology_path: Path to molecule topology file
        molecule_name: Human-readable molecule name
        
    Returns:
        Dictionary with energy and dipole properties
    """
    print(f"\n{'─' * 70}")
    print(f"📍 {molecule_name}")
    print(f"{'─' * 70}")
    
    if not topology_path.exists():
        print(f"   ⚠️  File not found: {topology_path}")
        print(f"   (Using mock data for demonstration)")
        return None
    
    # Load topology
    with open(topology_path) as f:
        topology = json.load(f)
    
    n_atoms = len(topology["symbols"])
    formula = "".join(sorted(set(topology["symbols"])))
    
    print(f"   Atoms: {n_atoms} atoms")
    print(f"   Elements: {', '.join(set(topology['symbols']))}")
    
    # Run energy calculation
    print(f"   Computing: RHF/STO-3G single-point energy...")
    
    result = exess.energy(
        topology_path,
        method="RestrictedHF",
        basis="STO-3G",
        run_opts=RunOpts(
            name=f"Dipole Analysis: {molecule_name}",
            tags=["polarity", "dipole", molecule_name.lower()],
        ),
        collect=True,
    )
    
    # Extract properties
    saved = exess.save_energy_outputs(result)
    props = {
        "name": molecule_name,
        "n_atoms": n_atoms,
        "energy": None,
        "dipole_vector": None,
        "dipole_magnitude": None,
    }
    
    for f in saved:
        if str(f).endswith(".json"):
            with open(f) as fh:
                data = json.load(fh)
                props["energy"] = data.get("total_energy")
                props["dipole_vector"] = data.get("dipole_moment")
                if props["dipole_vector"] and isinstance(props["dipole_vector"], list):
                    props["dipole_magnitude"] = calculate_dipole_magnitude(
                        props["dipole_vector"]
                    )
            break
    
    return props


def print_results(props: dict):
    """Pretty-print molecular properties."""
    if props is None:
        return
    
    name = props["name"]
    e = props["energy"]
    mu_vec = props["dipole_vector"]
    mu = props["dipole_magnitude"]
    
    print(f"\n   Results:")
    print(f"   ├─ Energy: {e:.8f} Hartree" if e else "   ├─ Energy: (not available)")
    
    if mu and mu_vec:
        print(f"   ├─ Dipole Moment: {mu:.4f} a.u. ({mu * 2.5418:.2f} Debye)")
        print(f"   ├─ Vector (x, y, z): [{mu_vec[0]:7.4f}, {mu_vec[1]:7.4f}, {mu_vec[2]:7.4f}]")
        
        polarity, charge_sep = assess_polarity(mu)
        print(f"   ├─ Polarity: {polarity}")
        print(f"   └─ Partial charges: {charge_sep}")
    else:
        print(f"   └─ Dipole Moment: (not available)")


# ===== Main Analysis =====
print("╔" + "═" * 70 + "╗")
print("║" + " Molecular Polarity & Dipole Moment Analysis".ljust(71) + "║")
print("║" + " Understanding molecular interactions & solubility".ljust(71) + "║")
print("╚" + "═" * 70 + "╝")

DATA_DIR = Path(__file__).parent / "data"

# Example molecules to analyze
molecules = [
    ("water_topology.json", "Water (H₂O)"),
    # ("ethane_t.json", "Ethane (C₂H₆)"),
    # ("ammonia_t.json", "Ammonia (NH₃)"),
    # ("acetone_t.json", "Acetone (CH₃)₂CO"),
]

print("\nAnalyzing molecular polarity and dipole moments...")
print("Method: Restricted Hartree-Fock / STO-3G")

results = []
for filename, name in molecules:
    fpath = DATA_DIR / filename
    props = run_dipole_analysis(fpath, name)
    if props:
        print_results(props)
        results.append(props)

# ===== Summary & Chemistry Insights =====
print("\n\n" + "═" * 70)
print("💡 Understanding Dipole Moments & Polarity")
print("═" * 70)

print("""
Dipole moments tell us about molecular structure and intermolecular forces:

🔬 What is a Dipole Moment?
   • Measure of charge separation in a molecule
   • μ = Σ qᵢ × rᵢ (sum of charge × distance)
   • Units: Debye (D) = 10⁻¹⁸ esu·cm, or atomic units (a.u.)
   • 1 a.u. dipole ≈ 2.54 Debye

🌊 Impact on Properties:
   ├─ Solubility: Polar → soluble in polar solvents (water)
   ├─ Boiling Point: Higher μ → higher intermolecular forces
   ├─ Reactivity: Dipoles attract electrophiles/nucleophiles
   ├─ Solvation: Determines solvation shell strength
   └─ Spectroscopy: Infrared active modes have dipole derivatives

📊 Typical Values (Debye):
   • Nonpolar: H₂, N₂, CO₂ → 0 D (symmetry)
   • Weakly polar: CH₄, C₆H₆ → 0-0.3 D
   • Moderately polar: H₂O (1.85 D), HCl (1.08 D)
   • Highly polar: HF (1.82 D), NaCl (9 D)

⚙️  Why Computational Chemistry Matters:
   • Experimental measurement: difficult, time-consuming
   • Computational prediction: fast for 100s or 1000s of molecules
   • Screen new compounds before synthesis
   • Design solvents and additives rationally
   • Understand drug-protein interactions
""")

if results:
    print(f"\n📈 Summary ({len(results)} molecules analyzed):")
    for props in sorted(results, key=lambda p: p["dipole_magnitude"] or 0):
        mu = props["dipole_magnitude"]
        if mu:
            polarity, _ = assess_polarity(mu)
            print(f"   • {props['name']:20s} → {mu:6.3f} a.u. ({mu*2.5418:6.2f} D) [{polarity}]")

print("\n✨ Next Steps:")
print("""
   1. Try other molecules (ammonia, acetone, etc.)
   2. Compare conformers of the same molecule
   3. Investigate how substitution affects dipole moment
   4. Correlate with experimental spectroscopic data
   5. Use in force field parameterization
""")

print("\n" + "╚" + "═" * 70 + "╝\n")
