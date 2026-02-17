"""
Example: CHELPG Charge Analysis for Aspirin

This script demonstrates how to:
1. Load a molecule from a PDB file
2. Calculate CHELPG partial charges using Rush
3. Extract and visualize the charge distribution

Run this script in a directory containing aspirin.pdb:
    python 01_chelpg_aspirin.py

Output files:
    - aspirin_topology.json: Converted topology
    - chelpg_charges.png: Bar chart of charges by atom
    - chelpg_aspirin.html: Interactive 3D visualization (left) + chart (right)
"""

from pathlib import Path
from rush import exess
from rush.client import RunError, download_object
from rush.convert.pdb import from_pdb
import json
import h5py
import tarfile
import zstandard as zstd
from io import BytesIO
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import py3Dmol
import base64


# ===== 1. Load PDB and convert to topology =====
print("Loading aspirin.pdb...")
pdb_content = Path("data/aspirin.pdb").read_text()
trc = from_pdb(pdb_content)

# Convert to topology JSON format
topology_path = Path("aspirin_topology.json")
topology_json = trc.topology.to_json()
if "schema_version" not in topology_json:
    topology_json["schema_version"] = "0.2.0"
topology_path.write_text(json.dumps(topology_json, indent=2))
print(f"✓ Topology saved to {topology_path}")


# ===== 2. Run CHELPG calculation =====
print("\nRunning CHELPG calculation...")
result = exess.chelpg(topology_path=topology_path, collect=True)

if isinstance(result, RunError):
    print(f"Run failed: {result.message}")
else:
    json_output, charges_ref = result
    print("✓ CHELPG calculation complete!")
    
    # ===== 3. Extract charges from HDF5 =====
    if isinstance(charges_ref, dict) and "Hdf5" in charges_ref:
        hdf5_obj = charges_ref["Hdf5"]
        qm_output = download_object(hdf5_obj["path"])
        decompressed = zstd.ZstdDecompressor().decompress(qm_output, max_output_size=int(1e9))
        
        with tarfile.open(fileobj=BytesIO(decompressed)) as tar:
            hdf5_f = tar.extractfile(tar.getnames()[1])
            with h5py.File(hdf5_f, "r") as f:
                frag_indices = sorted([int(x) for x in f["monomers"].keys()])
                charges = [
                    float(x)
                    for frag_idx in frag_indices
                    for x in f[f"monomers/{frag_idx}/chelpg_charges"]
                ]
        
        print(f"✓ Extracted {len(charges)} atomic charges")
        symbols = [trc.topology.symbols[i] for i in range(len(charges))]
        
        # ===== 4. Generate bar chart =====
        print("\nGenerating bar chart...")
        labels = [f"{s}{i}" for i, s in enumerate(symbols)]
        q_range = max(charges) - min(charges)
        norm_charges = [(q - min(charges)) / q_range for q in charges] if q_range > 0 else [0.5] * len(charges)
        colors = cm.RdBu([1 - c for c in norm_charges])  # Red=positive, Blue=negative
        
        fig, ax = plt.subplots(figsize=(max(10, len(charges)*0.5), 6))
        ax.bar(range(len(charges)), charges, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_xticks(range(len(charges)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.axhline(0, color='grey', linewidth=0.8, linestyle='--', alpha=0.7)
        ax.set_ylabel("Partial Charge (e)", fontsize=11, fontweight='bold')
        ax.set_title("CHELPG Charges – Aspirin", fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig("chelpg_charges.png", dpi=150, bbox_inches='tight')
        print("✓ Bar chart saved: chelpg_charges.png")
        
        # Convert to base64 for embedding
        with open("chelpg_charges.png", "rb") as img_file:
            chart_base64 = base64.b64encode(img_file.read()).decode()
        plt.close()
        
        # ===== 5. Generate 3D visualization =====
        print("Generating 3D visualization...")
        view = py3Dmol.view(width=700, height=600)
        view.addModel(pdb_content, "pdb")
        
        def charge_to_hex(q, q_min, q_max):
            """Map charge value to RdBu hex color (red=positive, blue=negative)"""
            if q_max == q_min:
                t = 0.5
            else:
                t = (q - q_min) / (q_max - q_min)
            rgba = cm.RdBu(1 - t)
            return '#{:02x}{:02x}{:02x}'.format(
                int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255)
            )
        
        q_min, q_max = min(charges), max(charges)
        for i, q in enumerate(charges):
            color = charge_to_hex(q, q_min, q_max)
            # Note: 3Dmol uses 1-based serial numbering
            view.setStyle({'serial': i + 1}, {
                'sphere': {'radius': 0.4, 'color': color},
                'stick': {'color': color}
            })
        
        view.zoomTo()
        html = view._make_html()
        
        # ===== 6. Create combined HTML visualization =====
        combined_html = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ margin: 0; font-family: Arial; }}
        .container {{ display: flex; height: 100vh; }}
        .viewer {{ flex: 1; }}
        .chart {{ flex: 1; display: flex; align-items: center; justify-content: center; background: #f5f5f5; padding: 20px; }}
        .chart img {{ max-width: 100%; max-height: 100%; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="viewer">
            {html}
        </div>
        <div class="chart">
            <img src="data:image/png;base64,{chart_base64}" alt="CHELPG Charges">
        </div>
    </div>
</body>
</html>
"""
        with open("chelpg_aspirin.html", "w") as f:
            f.write(combined_html)
        print("✓ Combined visualization saved: chelpg_aspirin.html")
        
        # ===== 7. Print summary =====
        print("\n✓ CHELPG Charges (Aspirin):")
        print("-" * 40)
        for i, (sym, q) in enumerate(zip(symbols, charges)):
            print(f"  Atom {i:2d} ({sym}): {q:8.5f} e")
        print("-" * 40)
        print(f"  Total charge: {sum(charges):8.5f} e")
        print(f"  Min charge:   {min(charges):8.5f} e")
        print(f"  Max charge:   {max(charges):8.5f} e")
        print("\n✓ All done! Open 'chelpg_aspirin.html' in a browser to view results.")
    else:
        print(f"Unexpected charges format: {charges_ref}")
