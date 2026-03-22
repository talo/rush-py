"""
Example: CHELPG Charge Analysis for Aspirin

This script demonstrates how to:
1. Load a molecule from a PDB file
2. Calculate CHELPG partial charges using Rush
3. Extract and visualize the charge distribution

Tutorial: https://exess.qdx.co/docs/tutorials/01-exess-chelpg.html

Run this script from its directory:
    python 01_exess_chelpg.py

Output files (saved to chelpg-outputs/):
    - aspirin_topology.json: Converted topology
    - chelpg_charges.png: Bar chart of charges by atom
    - chelpg_aspirin.html: Single-page viz — 3D structure (left), 2D charge map (center), bar chart (right)
"""

import base64
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import py3Dmol
from rdkit import Chem
from rdkit.Chem import rdDepictor

from rush import TRC, exess, from_pdb
from rush.exess import energy

matplotlib.use("Agg")  # Use non-GUI backend

# Shared charge-to-color mapping (RdBu: red=negative/electron-rich, blue=positive/electron-poor)
Q_ABSMAX = 0.5  # fixed scale range


def charge_to_rgb(q):
    """Map charge to (r,g,b) floats. Negative=red (electron-rich), Positive=blue (electron-poor)."""
    t = max(-1.0, min(1.0, q / Q_ABSMAX))
    if t < 0:
        r, g, b = 1.0, 1.0 + t * 0.6, 1.0 + t * 0.6
    else:
        r, g, b = 1.0 - t * 0.6, 1.0 - t * 0.6, 1.0
    return (r, g, b)


def charge_to_hex(q):
    """Map charge to hex color string."""
    r, g, b = charge_to_rgb(q)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def generate_aspirin_2d_svg(
    pdb_content, all_charges, all_symbols, width=480, height=400
):
    """Generate a charge-colored 2D SVG of aspirin using RDKit for coordinates.

    Args:
        pdb_content: PDB file text (used for atom ordering consistency)
        all_charges: List of charges for ALL atoms (heavy + H) from CHELPG
        all_symbols: List of element symbols for ALL atoms
        width, height: SVG dimensions
    """
    # Load molecule from PDB (heavy atoms only) for 2D layout
    mol = Chem.MolFromPDBBlock(pdb_content, removeHs=True, sanitize=True)
    rdDepictor.Compute2DCoords(mol)
    conf = mol.GetConformer()
    n_heavy = mol.GetNumAtoms()

    # Aggregate charges: sum each H's charge onto its parent heavy atom
    heavy_charges = list(all_charges[:n_heavy])  # start with heavy atom charges
    mol_full = Chem.MolFromPDBBlock(pdb_content, removeHs=False, sanitize=True)
    for i in range(n_heavy, len(all_charges)):
        if all_symbols[i] == "H":
            for neighbor in mol_full.GetAtomWithIdx(i).GetNeighbors():
                parent_idx = neighbor.GetIdx()
                if parent_idx < n_heavy:
                    heavy_charges[parent_idx] += all_charges[i]
                break  # H atoms have exactly one neighbor

    # Extract 2D coordinates
    coords = [
        (conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y) for i in range(n_heavy)
    ]

    # SVG coordinate transform
    xs, ys = [c[0] for c in coords], [c[1] for c in coords]
    margin = 55
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    scale = min(
        (width - 2 * margin) / (x_max - x_min or 1),
        (height - 80 - 2 * margin) / (y_max - y_min or 1),
    )  # reserve 80px for legend

    # Center the molecule in the available SVG area
    mol_w = (x_max - x_min) * scale
    mol_h = (y_max - y_min) * scale
    avail_w = width - 2 * margin
    avail_h = height - 80 - 2 * margin  # match legend reservation
    offset_x = margin + (avail_w - mol_w) / 2
    offset_y = margin + (avail_h - mol_h) / 2

    def to_svg(x, y):
        return offset_x + (x - x_min) * scale, offset_y + (y_max - y) * scale

    svg_coords = [to_svg(x, y) for x, y in coords]

    lines = []
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
    )

    # Draw bonds
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        x1, y1 = svg_coords[i]
        x2, y2 = svg_coords[j]
        bt = str(bond.GetBondType())

        if bt == "DOUBLE":
            dx, dy = x2 - x1, y2 - y1
            length = math.sqrt(dx * dx + dy * dy) or 1
            ox, oy = -dy / length * 3, dx / length * 3
            lines.append(
                f'<line x1="{x1 + ox:.1f}" y1="{y1 + oy:.1f}" x2="{x2 + ox:.1f}" y2="{y2 + oy:.1f}" stroke="#52525b" stroke-width="2"/>'
            )
            lines.append(
                f'<line x1="{x1 - ox:.1f}" y1="{y1 - oy:.1f}" x2="{x2 - ox:.1f}" y2="{y2 - oy:.1f}" stroke="#52525b" stroke-width="2"/>'
            )
        else:
            lines.append(
                f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#52525b" stroke-width="2"/>'
            )

    # Aromatic ring circle
    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        if len(ring) == 6 and all(
            mol.GetBondBetweenAtoms(ring[k], ring[(k + 1) % 6]).GetIsAromatic()
            for k in range(6)
        ):
            cx = sum(svg_coords[r][0] for r in ring) / 6
            cy = sum(svg_coords[r][1] for r in ring) / 6
            r = 0.55 * math.sqrt(
                (svg_coords[ring[0]][0] - cx) ** 2 + (svg_coords[ring[0]][1] - cy) ** 2
            )
            lines.append(
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" '
                f'fill="none" stroke="#52525b" stroke-width="1.5" stroke-dasharray="4,3"/>'
            )

    # Draw atoms — ALL atoms get a colored circle; heteroatoms also get a label
    for i in range(n_heavy):
        atom = mol.GetAtomWithIdx(i)
        sym = atom.GetSymbol()
        x, y = svg_coords[i]
        q = heavy_charges[i]
        fill = charge_to_hex(q)
        # Circle radius scales with charge magnitude
        base_r = 14
        mag_r = base_r + abs(q / Q_ABSMAX) * 6  # 14–20px

        # Charge-colored halo (semi-transparent)
        lines.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{mag_r:.1f}" '
            f'fill="{fill}" opacity="0.35"/>'
        )

        if sym != "C":
            # Solid background + label for heteroatoms
            lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="13" fill="#18181b"/>')
            lines.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="13" fill="{fill}" opacity="0.25"/>'
            )
            lines.append(
                f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="middle" '
                f'dominant-baseline="central" fill="{fill}" '
                f'font-family="Arial, sans-serif" font-size="14" font-weight="bold">{sym}</text>'
            )
            # H labels on heteroatoms
            n_h = atom.GetTotalNumHs()
            if n_h > 0:
                h_text = f"H{'' if n_h == 1 else n_h}"
                lines.append(
                    f'<text x="{x + 14:.1f}" y="{y:.1f}" text-anchor="start" '
                    f'dominant-baseline="central" fill="{fill}" '
                    f'font-family="Arial, sans-serif" font-size="11">{h_text}</text>'
                )
        else:
            # Small dot for carbon vertices
            lines.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{fill}" opacity="0.8"/>'
            )

        # Charge value label (small, below atom)
        lines.append(
            f'<text x="{x:.1f}" y="{y + mag_r + 10:.1f}" text-anchor="middle" '
            f'fill="#a1a1aa" font-family="Arial, sans-serif" font-size="9">'
            f"{q:+.3f}</text>"
        )

    # ===== Color legend / colorbar =====
    legend_y = height - 45
    bar_x, bar_w, bar_h = 60, width - 120, 14
    # Gradient definition
    lines.append("<defs>")
    lines.append('<linearGradient id="chg-grad" x1="0" x2="1" y1="0" y2="0">')
    for pct in range(0, 101, 5):
        q_val = -Q_ABSMAX + (pct / 100) * 2 * Q_ABSMAX
        r, g, b = charge_to_rgb(q_val)
        lines.append(
            f'  <stop offset="{pct}%" stop-color="rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})"/>'
        )
    lines.append("</linearGradient>")
    lines.append("</defs>")
    # Bar
    lines.append(
        f'<rect x="{bar_x}" y="{legend_y}" width="{bar_w}" height="{bar_h}" '
        f'rx="3" fill="url(#chg-grad)" opacity="0.8"/>'
    )
    lines.append(
        f'<rect x="{bar_x}" y="{legend_y}" width="{bar_w}" height="{bar_h}" '
        f'rx="3" fill="none" stroke="#3f3f46" stroke-width="1"/>'
    )
    # Labels
    label_y = legend_y + bar_h + 14
    mid_x = bar_x + bar_w / 2
    lines.append(
        f'<text x="{bar_x:.0f}" y="{label_y}" text-anchor="middle" '
        f'fill="#ef4444" font-family="Arial, sans-serif" font-size="10" font-weight="bold">'
        f"−{Q_ABSMAX}</text>"
    )
    lines.append(
        f'<text x="{mid_x:.0f}" y="{label_y}" text-anchor="middle" '
        f'fill="#a1a1aa" font-family="Arial, sans-serif" font-size="10">0</text>'
    )
    lines.append(
        f'<text x="{bar_x + bar_w:.0f}" y="{label_y}" text-anchor="middle" '
        f'fill="#60a5fa" font-family="Arial, sans-serif" font-size="10" font-weight="bold">'
        f"+{Q_ABSMAX}</text>"
    )
    # Title
    lines.append(
        f'<text x="{mid_x:.0f}" y="{legend_y - 6}" text-anchor="middle" '
        f'fill="#71717a" font-family="Arial, sans-serif" font-size="10">'
        f"CHELPG Charge (e) — red: electron-rich · blue: electron-poor</text>"
    )

    lines.append("</svg>")
    return "\n".join(lines)


# ===== 0. Setup paths =====
data_dir = Path(__file__).parent / "data"
output_dir = Path(__file__).parent / "chelpg-outputs"
output_dir.mkdir(exist_ok=True)
print(f"Output directory: {output_dir}")

# ===== 1. Load PDB and convert to topology =====
pdb_path = data_dir / "aspirin.pdb"
mol_name = pdb_path.stem.replace("_", " ").replace("-", " ").title()
print(f"Loading {pdb_path.name}...")
pdb_content = pdb_path.read_text(encoding="utf-8")
trc = from_pdb(pdb_content)
assert isinstance(trc, TRC)  # Confirm we got just one structure

# Derive molecular formula from topology (Hill order: C first, H second, then alphabetical)
elem_counts = Counter(
    str(trc.topology.symbols[i]) for i in range(len(trc.topology.symbols))
)
subscript = str.maketrans(
    "0123456789", "\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089"
)
formula_parts = []
for elem in ["C", "H"]:  # C and H first (Hill order)
    if elem in elem_counts:
        n = elem_counts.pop(elem)
        formula_parts.append(f"{elem}{n}" if n > 1 else elem)
for elem in sorted(elem_counts):
    n = elem_counts[elem]
    formula_parts.append(f"{elem}{n}" if n > 1 else elem)
mol_formula = "".join(formula_parts)
mol_formula_sub = mol_formula.translate(subscript)  # for terminal output
# HTML subscript version
mol_formula_html = ""
for ch in mol_formula:
    if ch.isdigit():
        mol_formula_html += f"<sub>{ch}</sub>"
    else:
        mol_formula_html += ch

print(f"  Molecule: {mol_name} ({mol_formula_sub})")

# Convert to topology JSON format
topology_path = output_dir / f"{pdb_path.stem}_topology.json"
topology_path.write_text(json.dumps(trc.topology.to_json(), indent=2), encoding="utf-8")
print(f"✓ Topology saved to {topology_path}")


# ===== 2. Run CHELPG calculation =====
print("\nRunning CHELPG calculation...")
result = energy(
    topology_path=topology_path,
    frag_keywords=None,  # disable fragmentation for CHELPG
    export_keywords=exess.ExportKeywords(export_chelpg_charges=True),
    convert_hdf5_to_json=True,
).collect()

exports = result.fetch().exports
assert isinstance(exports, dict)
charges = exports["chelpg_charges"]

print("✓ CHELPG calculation complete!")
print(f"✓ Extracted {len(charges)} atomic charges")
symbols = [trc.topology.symbols[i] for i in range(len(charges))]

# ===== 3. Generate bar chart (tall, narrow — fits right column) =====
print("\nGenerating bar chart...")
labels = [f"{s}{i}" for i, s in enumerate(symbols)]

bar_colors = [charge_to_rgb(q) for q in charges]

fig, ax = plt.subplots(figsize=(3.5, 6.5))
fig.patch.set_facecolor("#000000")
ax.set_facecolor("#000000")
ax.barh(
    range(len(charges)),
    charges,
    color=bar_colors,
    edgecolor="#27272a",
    linewidth=0.5,
)
ax.set_yticks(range(len(charges)))
ax.set_yticklabels(labels, fontsize=8, color="#a1a1aa", fontfamily="monospace")
ax.invert_yaxis()
ax.axvline(0, color="#52525b", linewidth=0.8, linestyle="--", alpha=0.7)
ax.set_xlabel("Partial Charge (e)", fontsize=10, fontweight="bold", color="#a1a1aa")
ax.tick_params(axis="x", colors="#a1a1aa")
ax.grid(axis="x", alpha=0.15, color="#52525b")
for spine in ax.spines.values():
    spine.set_color("#27272a")
plt.tight_layout()
chart_path = output_dir / "chelpg_charges.png"
plt.savefig(chart_path, dpi=150, bbox_inches="tight", facecolor="#000000")
print(f"✓ Bar chart saved: {chart_path}")

# Convert to base64 for embedding
with open(chart_path, "rb") as img_file:
    chart_base64 = base64.b64encode(img_file.read()).decode()
plt.close()

# ===== 4. Generate 3D visualization =====
print("Generating 3D visualization...")
view = py3Dmol.view(width=700, height=600)
view.addModel(pdb_content, "pdb")

for i, q in enumerate(charges):
    color = charge_to_hex(q)
    # Note: 3Dmol uses 1-based serial numbering
    view.setStyle(
        {"serial": i + 1},
        {"sphere": {"radius": 0.4, "color": color}, "stick": {"color": color}},
    )

view.setBackgroundColor("black")
view.zoomTo()
html_3d = view._make_html()

# ===== 5. Generate 2D structure diagram =====
print("Generating 2D structure diagram...")
structure_svg = generate_aspirin_2d_svg(pdb_content, charges, symbols)
print("✓ 2D structure SVG generated")

# ===== 6. Create combined HTML visualization =====
combined_html = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; background: #0a0a0c; color: #e4e4e7; height: 100vh; overflow: hidden; display: flex; flex-direction: column; }}

        .title-bar {{ text-align: center; padding: 14px 20px 10px; flex-shrink: 0; }}
        .title-bar h1 {{ font-size: 15px; color: #d4d4d8; font-weight: 600; letter-spacing: 0.3px; }}
        .title-bar .formula {{ font-size: 11px; color: #52525b; margin-top: 3px; letter-spacing: 0.2px; }}

        .panels {{ display: flex; flex: 1; gap: 8px; padding: 0 8px 8px; min-height: 0; }}

        .panel {{
            display: flex; flex-direction: column; min-height: 0;
            background: #000000; border: 1px solid #1e1e22; border-radius: 10px; overflow: hidden;
        }}
        .panel-side {{ flex: 1; }}
        .panel-center {{ flex: 1.3; }}

        .panel-label {{
            font-size: 10px; color: #5a5a65; text-transform: uppercase; letter-spacing: 1.2px;
            padding: 10px 14px 6px; flex-shrink: 0; font-weight: 600;
        }}

        .viewer-3d {{ flex: 1; position: relative; border-radius: 0 0 10px 10px; overflow: hidden; background: #000; }}
        .viewer-3d div {{ width: 100% !important; height: 100% !important; position: relative; }}
        .viewer-3d iframe {{ width: 100% !important; height: 100% !important; position: absolute; top: 0; left: 0; border: none; }}

        .viewer-2d {{ flex: 1; display: flex; align-items: center; justify-content: center; padding: 16px; width: 100%; }}
        .viewer-2d svg {{ display: block; margin: 0 auto; max-width: 100%; max-height: 100%; }}

        .chart {{ flex: 1; display: flex; align-items: center; justify-content: center; padding: 16px; min-height: 0; }}
        .chart img {{ max-height: 100%; max-width: 100%; object-fit: contain; }}
    </style>
</head>
<body>
    <div class="title-bar">
        <h1>CHELPG Charge Analysis &mdash; {mol_name}</h1>
        <div class="formula">{mol_formula_html} &middot; Total charge: {sum(charges):+.4f} e</div>
    </div>
    <div class="panels">
        <div class="panel panel-side">
            <div class="panel-label">2D Charge Map</div>
            <div class="viewer-2d">{structure_svg}</div>
        </div>
        <div class="panel panel-center">
            <div class="panel-label">3D Structure</div>
            <div class="viewer-3d">{html_3d}</div>
        </div>
        <div class="panel panel-side">
            <div class="panel-label">Charge Distribution</div>
            <div class="chart"><img src="data:image/png;base64,{chart_base64}" alt="CHELPG Charges"></div>
        </div>
    </div>
</body>
</html>
"""
html_path = output_dir / f"chelpg_{pdb_path.stem}.html"
with open(html_path, "w", encoding="utf-8") as f:
    f.write(combined_html)
print(f"✓ Combined visualization saved: {html_path}")

# ===== 7. Print summary =====
print(f"\n✓ CHELPG Charges ({mol_name}):")
print("-" * 40)
for i, (sym, q) in enumerate(zip(symbols, charges)):
    print(f"  Atom {i:2d} ({sym}): {q:8.5f} e")
print("-" * 40)
print(f"  Total charge: {sum(charges):8.5f} e")
print(f"  Min charge:   {min(charges):8.5f} e")
print(f"  Max charge:   {max(charges):8.5f} e")
print(f"\n✓ All done! Open '{html_path}' in a browser to view results.")
