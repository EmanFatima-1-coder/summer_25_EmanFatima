from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd
from collections import Counter

def extract_features(file_path, label):
    supplier = Chem.SDMolSupplier(file_path)
    data = []

    for mol in supplier:
        if mol is None:
            continue

        # Molecular descriptors
        molwt = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        h_donors = Descriptors.NumHDonors(mol)
        h_acceptors = Descriptors.NumHAcceptors(mol)
        tpsa = Descriptors.TPSA(mol)
        rot_bonds = Descriptors.NumRotatableBonds(mol)

        # Atom encodings: count atom types and properties
        atom_types = Counter()
        aromatic_count = 0
        sp2_count = 0

        for atom in mol.GetAtoms():
            atom_types[atom.GetSymbol()] += 1
            if atom.GetIsAromatic():
                aromatic_count += 1
            if str(atom.GetHybridization()) == "SP2":
                sp2_count += 1

        # Build row dictionary
        row = {
            "MolWt": molwt,
            "LogP": logp,
            "NumHDonors": h_donors,
            "NumHAcceptors": h_acceptors,
            "TPSA": tpsa,
            "RotatableBonds": rot_bonds,
            "NumAromaticAtoms": aromatic_count,
            "NumSP2Atoms": sp2_count,
            "NumC": atom_types.get("C", 0),
            "NumN": atom_types.get("N", 0),
            "NumO": atom_types.get("O", 0),
            "NumS": atom_types.get("S", 0),
            "NumF": atom_types.get("F", 0),
            "NumCl": atom_types.get("Cl", 0),
            "NumBr": atom_types.get("Br", 0),
            "Label": label
        }

        data.append(row)

    return pd.DataFrame(data)

# File paths
drug_file = "drug_compounds.sdf"
non_drug_file = "non_drug.sdf"

# Extract features
print("Extracting drug-like features...")
df_drug = extract_features(drug_file, 1)

print("Extracting non-drug features...")
df_non_drug = extract_features(non_drug_file, 0)

# Combine into one dataset
df_combined = pd.concat([df_drug, df_non_drug], ignore_index=True)
df_combined.to_csv("unified_ligand_features.csv", index=False)

print("\nUnified dataset saved as 'unified_ligand_features.csv'")
print("Shape:", df_combined.shape)
print(df_combined.head())
