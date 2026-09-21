"""
=============================================================
 Phase 2 — Step 1: 3D Molecular Feature Extraction
=============================================================
Generates a 633-dim 3D descriptor vector for every unique drug
in the OncoPolyPharmacology dataset using RDKit's:
  AUTOCORR3D  (80 features)
  RDF         (210 features)
  MORSE       (224 features)
  WHIM        (114 features)
  Shape (NPR1, NPR2, Asphericity, Eccentricity, Spherocity) (5)

Output:
  data/drug_3d_features.pkl
  data/drug_3d_features_report.csv
=============================================================
"""

import os, pickle, warnings
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors3D
from rdkit.Chem.rdMolDescriptors import CalcAUTOCORR3D, CalcRDF, CalcMORSE, CalcWHIM

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATASET  = "/home/sanjay/Documents/Final Year/Code/data/oncopolypharmacology.pkl"
OUT_PKL  = os.path.join(DATA_DIR, "drug_3d_features.pkl")
OUT_CSV  = os.path.join(DATA_DIR, "drug_3d_features_report.csv")
os.makedirs(DATA_DIR, exist_ok=True)

print("=" * 60)
print(" Phase 2 — Step 1: 3D Molecular Feature Extraction")
print("=" * 60)

df = pd.read_pickle(DATASET)

drug1_map = df[["Drug1_ID", "Drug1"]].drop_duplicates(subset="Drug1_ID")
drug2_map = (df[["Drug2_ID", "Drug2"]].drop_duplicates(subset="Drug2_ID")
             .rename(columns={"Drug2_ID": "Drug1_ID", "Drug2": "Drug1"}))
all_drugs = (pd.concat([drug1_map, drug2_map], ignore_index=True)
             .drop_duplicates(subset="Drug1_ID").reset_index(drop=True))

print(f"\nUnique drugs found: {len(all_drugs)}")
print(f"Drugs: {sorted(all_drugs['Drug1_ID'].tolist())}\n")


def extract_3d_descriptors(smiles, drug_id):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    res = AllChem.EmbedMolecule(mol, params)
    if res != 0:
        return None
    AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
    try:
        autocorr = np.array(CalcAUTOCORR3D(mol))   # 80
        rdf      = np.array(CalcRDF(mol))            # 210
        morse    = np.array(CalcMORSE(mol))          # 224
        whim     = np.array(CalcWHIM(mol))           # 114
        shape    = np.array([
            Descriptors3D.NPR1(mol),
            Descriptors3D.NPR2(mol),
            Descriptors3D.Asphericity(mol),
            Descriptors3D.Eccentricity(mol),
            Descriptors3D.SpherocityIndex(mol),
        ])                                           # 5
        feats = np.concatenate([autocorr, rdf, morse, whim, shape])
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        return feats
    except Exception as e:
        print(f"  Descriptor error for {drug_id}: {e}")
        return None


feature_store = {}
report_rows   = []

print("Extracting 3D descriptors...\n")
for _, row in all_drugs.iterrows():
    drug_id = row["Drug1_ID"]
    smiles  = row["Drug1"]
    print(f"  [{drug_id}]...", end=" ", flush=True)
    feats = extract_3d_descriptors(smiles, drug_id)
    if feats is not None:
        feature_store[drug_id] = feats
        report_rows.append({"drug_id": drug_id, "n_features": len(feats),
                            "feat_mean": round(float(feats.mean()), 4),
                            "feat_std": round(float(feats.std()), 4), "status": "OK"})
        print(f"OK  (dim={len(feats)}, mean={feats.mean():.3f})")
    else:
        report_rows.append({"drug_id": drug_id, "status": "FAILED"})
        print("FAILED")

with open(OUT_PKL, "wb") as f:
    pickle.dump(feature_store, f)
pd.DataFrame(report_rows).to_csv(OUT_CSV, index=False)

print("\n" + "=" * 60)
print(f"  Successfully encoded : {len(feature_store)} / {len(all_drugs)} drugs")
print(f"  Feature vector dim   : {next(iter(feature_store.values())).shape[0]}")
print(f"  Output PKL           : {OUT_PKL}")
print(f"  Output report CSV    : {OUT_CSV}")
print("=" * 60)
