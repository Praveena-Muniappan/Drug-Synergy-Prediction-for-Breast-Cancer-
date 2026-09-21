from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
from tdc.multi_pred import DrugSyn
import logging

# Disable RDKit warnings for your review
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

data = DrugSyn(name='OncoPolyPharmacology')
df = data.get_data(format='df')

def generate_3d_conformer(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            mol = Chem.AddHs(mol)
            # Use ETKDGv3 which is more robust
            res = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
            return mol if res == 0 else None
    except:
        return None
    return None

# Only process first 50 rows for the review demo
subset = df.head(50).copy()

print("Generating 3D conformers for subset...")
subset['Drug1_3D'] = subset['Drug1'].apply(generate_3d_conformer)
subset['Drug2_3D'] = subset['Drug2'].apply(generate_3d_conformer)

# Save to a smaller file
subset.to_pickle('synergy_subset_3d.pkl')
print("Success! Subset saved to 'synergy_subset_3d.pkl'.")