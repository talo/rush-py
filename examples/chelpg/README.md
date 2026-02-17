# CHELPG Example: Aspirin Charge Distribution

This example demonstrates how to use Rush to calculate and visualize CHELPG partial charges for a drug molecule (aspirin).

## What You'll Learn

- Load a molecule from a PDB file
- Calculate CHELPG charges using Rush
- Extract charges from HDF5 results
- Visualize charge distribution with a bar chart and interactive 3D plot

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install rush-py matplotlib py3Dmol
   ```

2. **Run the example:**
   ```bash
   cd examples/chelpg
   python 01_chelpg_aspirin.py
   ```

3. **View results:**
   - `chelpg_charges.png` — Bar chart of charges by atom
   - `chelpg_aspirin.html` — Interactive visualization (3D structure + chart)
   - Terminal output — Summary statistics

## What the Script Does

1. Loads `aspirin.pdb` using `from_pdb()`
2. Converts to topology JSON format
3. Submits CHELPG calculation to Rush
4. Extracts charges from HDF5 results
5. Generates:
   - Bar chart with RdBu coloring (red=positive, blue=negative)
   - Interactive 3D view of the molecule with charge-colored atoms
   - Combined HTML visualization

## Output

```
✓ Topology saved to aspirin_topology.json
✓ CHELPG calculation complete!
✓ Extracted 21 atomic charges
✓ Bar chart saved: chelpg_charges.png
✓ Combined visualization saved: chelpg_aspirin.html
✓ CHELPG Charges (Aspirin):
----------------------------------------
  Atom  0 (O): -0.47332 e
  Atom  1 (O): -0.63418 e
  ...
----------------------------------------
  Total charge: -0.00000 e
  Min charge:   -0.63418 e
  Max charge:   +0.93806 e
✓ All done! Open 'chelpg_aspirin.html' in a browser to view results.
```

## Interpretation

- **Red atoms (positive charges):** Electron-poor regions, good H-bond donors
- **Blue atoms (negative charges):** Electron-rich regions, good H-bond acceptors
- **Gray atoms (near-zero):** Neutral, hydrophobic character

For aspirin:
- **Oxygens:** Highly negative (-0.47 to -0.63e) — H-bond acceptors
- **Carboxylic acid carbon:** Positive (+0.41e) — acidic proton source
- **Aromatic ring:** Near-neutral — hydrophobic interactions

## Next Steps

- Try with your own molecules by replacing `aspirin.pdb`
- Customize visualization colors by editing `charge_to_hex()`
- Export charges to CSV for downstream analysis
- Link to the [CHELPG Tutorial](../../docs/tutorials/01-chelpg.md) for more details
