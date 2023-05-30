#!/usr/bin/env python3

import json
import sys

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

from .data import QDXQCMol

DEBUG = False


def standard_bond_length(sym):
    # https://www.ibchem.com/IB16/Section04-energetics/data/Hbde_data.htm
    bonds = {
        "H": 0.74,
        "C": 1.09,
        "N": 1.01,
        "O": 0.96,
        "F": 0.92,
        "P": 1.44,
        "S": 1.34,
        "CL": 1.27,
        "SE": 1.46,
    }
    return bonds.get(sym.upper(), 1.0)


def scale_bond(A, B, d):
    R = np.subtract(B, A)
    r = np.linalg.norm(R)
    return np.add(A, R * d / r)


def nuclear_charge(frag):
    Ztot = 0
    for atom in frag.GetAtoms():
        Ztot += atom.GetAtomicNum()
    return Ztot


def dump_to_file(frags, fmt="mol"):
    for i in range(len(frags)):
        frag = frags[i]
        fname = "frag{}.{}".format(i, fmt)
        with open(fname, "w") as file:
            if fmt == "mol":
                file.write(Chem.MolToMolBlock(frag))
            else:
                file.write(Chem.MolToXYZBlock(frag))


def extract_fragments(data: QDXQCMol):
    """
    Takes in JSON formatted data and extracts the fragments into Chem.RWMol objects.
    These fragments have the capping hydrogen atoms added with standardized bonds.
    """
    symbols = data.symbols
    fragments = data.fragments or []
    geometry = data.geometry
    connectivity = data.connectivity or []

    coords = np.array(geometry)
    coords = coords.reshape(len(symbols), 3)

    frags = []
    atom2frag = {}
    atom2atom = {}

    # Create XYZ block to parse as a RWMol
    for ifrag in range(len(fragments)):
        xyz = "{}\n\n".format(len(fragments[ifrag]))
        iatom = 0

        for atom in fragments[ifrag]:
            line = "{}   {}   {}   {}\n".format(symbols[atom], *coords[atom])
            xyz += line
            atom2frag[atom] = ifrag
            atom2atom[atom] = iatom
            iatom += 1

        mol = Chem.RWMol(Chem.MolFromXYZBlock(xyz))
        frags.append(mol)

    # Assign bond to their respective fragments, adding caps for inter-fragment bonds
    for bond in connectivity:
        atom1, atom2, order = bond
        f1 = atom2frag[atom1]
        f2 = atom2frag[atom2]
        a1 = atom2atom[atom1]
        a2 = atom2atom[atom2]

        if f1 == f2:
            # intra-fragment bond
            if order == 1:
                frags[f1].AddBond(a1, a2, Chem.BondType.SINGLE)
            elif order == 2:
                frags[f1].AddBond(a1, a2, Chem.BondType.DOUBLE)
            elif order == 3:
                frags[f1].AddBond(a1, a2, Chem.BondType.TRIPLE)
            else:
                sys.stderr.write("WARNING: unknown bond order encountered: {}".format(order))

        else:
            # inter-fragment bond => add H caps
            conf = frags[f1].GetConformer(-1)
            idx = frags[f1].AddAtom(Chem.Atom(1))
            blen = standard_bond_length(symbols[atom1])
            pos = scale_bond(coords[atom1], coords[atom2], blen)
            conf.SetAtomPosition(idx, pos)

            conf = frags[f2].GetConformer(-1)
            idx = frags[f2].AddAtom(Chem.Atom(1))
            blen = standard_bond_length(symbols[atom2])
            pos = scale_bond(coords[atom2], coords[atom1], blen)
            conf.SetAtomPosition(idx, pos)

    # if DEBUG: dump_to_file(frags,"xyz")

    return frags


def calc_formal_charge(frag):
    """
    Takes in a RWMol object and looks for charged functional groups, adding
    formal charges as required.
    """
    frag.UpdatePropertyCache(strict=False)

    # N+ on LYS, ARG and HIS
    for indices in frag.GetSubstructMatches(Chem.MolFromSmarts("N")):
        atom = frag.GetAtomWithIdx(indices[0])
        if atom.GetExplicitValence() == 4 and atom.GetFormalCharge() == 0:
            atom.SetFormalCharge(1)
            if DEBUG:
                print(
                    "N+   {}:  {}   {}   {}".format(
                        indices[0], atom.GetAtomicNum(), atom.GetExplicitValence(), atom.GetFormalCharge()
                    )
                )

    # S- on CYS
    for indices in frag.GetSubstructMatches(Chem.MolFromSmarts("S")):
        atom = frag.GetAtomWithIdx(indices[0])
        if atom.GetExplicitValence() == 1 and atom.GetFormalCharge() == 0:
            atom.SetFormalCharge(-1)
            if DEBUG:
                print(
                    "S-   {}:  {}   {}   {}".format(
                        indices[0], atom.GetAtomicNum(), atom.GetExplicitValence(), atom.GetFormalCharge()
                    )
                )

    # COO- on ASP and GLU
    for indices in frag.GetSubstructMatches(Chem.MolFromSmarts("C(=O)[O]")):
        atom = frag.GetAtomWithIdx(indices[2])
        if atom.GetExplicitValence() == 1 and atom.GetFormalCharge() == 0:
            atom.SetFormalCharge(-1)
            if DEBUG:
                print(
                    "COO- {}:  {}   {}   {}".format(
                        indices[2], atom.GetAtomicNum(), atom.GetExplicitValence(), atom.GetFormalCharge()
                    )
                )

    # Protonated imidazole ring on HIS
    for indices in frag.GetSubstructMatches(Chem.MolFromSmiles("C1=CNCN1")):
        c = frag.GetAtomWithIdx(indices[3])
        if c.GetExplicitValence() == 4:
            continue
        n1 = frag.GetAtomWithIdx(indices[2])
        n2 = frag.GetAtomWithIdx(indices[4])

        if n1.GetExplicitValence() == 3 and n2.GetExplicitValence() == 3:
            if n1.GetFormalCharge() == 0 and n2.GetFormalCharge() == 0:
                n1.SetFormalCharge(1)
            if DEBUG:
                print(
                    "HIS+ {}:  {}   {}   {}".format(
                        indices[2], n1.GetAtomicNum(), n1.GetExplicitValence(), n1.GetFormalCharge()
                    )
                )

    return Chem.rdmolops.GetFormalCharge(frag)


def run(data: QDXQCMol):
    frags = extract_fragments(data)
    # if DEBUG: dump_to_file(frags)

    charges = []

    radical = 0
    ifrag = 0
    for f in frags:
        q = calc_formal_charge(f)
        n = nuclear_charge(f)
        if (n - q) % 2 != 0:
            radical += 1

        if DEBUG:
            print(Chem.MolToSmiles(f))
            print("Formal charge frag {}: {}".format(ifrag, q))
            print("Total nuclear charge:  {}".format(n))
            print("Number of electrons:   {}".format(n - q))
            if (n - q) % 2 != 0:
                print("*** Non singlet system ***")
            print("")

        ifrag += 1
        charges.append(q)

    if radical > 0:
        sys.stderr.write("WARNING: {} radical systems found\n".format(radical))

    if not DEBUG:
        data.fragment_charges = charges
        return data


def main():
    if sys.argv[1] == "-d":
        global DEBUG
        DEBUG = True
        fname = sys.argv[2]
    else:
        fname = sys.argv[1]

    with open(fname, "r") as f:
        data = json.load(f)
        result = run(data)
        print(json.dumps(result, indent=4))


if __name__ == "__main__":
    main()
