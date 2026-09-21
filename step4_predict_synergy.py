"""
=============================================================
 Phase 2 — Step 4: Predict Synergy for Any Drug Pair
=============================================================
Given:
  - SMILES of Drug A
  - SMILES of Drug B
  - A Cell Line ID (or "all" to predict across all 39 cell lines)

Outputs the predicted synergy score Y.

Usage:
  python step4_predict_synergy.py
  (interactive prompts for SMILES and cell line)

Or import and call predict() directly.
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

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR       = os.path.join(BASE_DIR, "data")
DRUG_FEAT_PKL  = os.path.join(DATA_DIR, "drug_3d_features.pkl")
CELL_EMBED_PKL = os.path.join(DATA_DIR, "cell_line_embeddings.pkl")
MODEL_PKL      = os.path.join(DATA_DIR, "synergy_model_3d.pkl")


def extract_3d_descriptors_from_smiles(smiles):
    """Return 633-dim 3D feature vector for a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot parse SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    res = AllChem.EmbedMolecule(mol, params)
    if res != 0:
        raise ValueError(f"3D embedding failed for: {smiles}")
    AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)

    autocorr = np.array(CalcAUTOCORR3D(mol))
    rdf      = np.array(CalcRDF(mol))
    morse    = np.array(CalcMORSE(mol))
    whim     = np.array(CalcWHIM(mol))
    shape    = np.array([
        Descriptors3D.NPR1(mol), Descriptors3D.NPR2(mol),
        Descriptors3D.Asphericity(mol), Descriptors3D.Eccentricity(mol),
        Descriptors3D.SpherocityIndex(mol),
    ])
    feats = np.concatenate([autocorr, rdf, morse, whim, shape])
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


def load_resources():
    """Load model, drug features, and cell line embeddings."""
    with open(DRUG_FEAT_PKL,  "rb") as f: drug_features    = pickle.load(f)
    with open(CELL_EMBED_PKL, "rb") as f: cell_embeddings  = pickle.load(f)
    with open(MODEL_PKL,      "rb") as f: bundle           = pickle.load(f)
    return drug_features, cell_embeddings, bundle


def predict(smiles_a, smiles_b, cell_line_id="all",
            drug_id_a=None, drug_id_b=None,
            drug_features=None, cell_embeddings=None, bundle=None):
    """
    Predict synergy score for a drug pair on one or all cell lines.

    Parameters
    ----------
    smiles_a, smiles_b : str   SMILES strings for the two drugs
    cell_line_id       : str   Cell line name or 'all'
    drug_id_a/b        : str   Optional: known drug ID (skips 3D extraction)

    Returns
    -------
    pd.DataFrame with columns: cell_line, predicted_synergy, interpretation
    """
    if drug_features is None or cell_embeddings is None or bundle is None:
        drug_features, cell_embeddings, bundle = load_resources()

    model  = bundle["model"]
    scaler = bundle["scaler"]

    # ── Get 3D features ──
    # Use pre-computed if available (faster); else compute fresh
    if drug_id_a and drug_id_a in drug_features:
        feat_a = drug_features[drug_id_a]
    else:
        print(f"  Computing 3D descriptors for Drug A...")
        feat_a = extract_3d_descriptors_from_smiles(smiles_a)

    if drug_id_b and drug_id_b in drug_features:
        feat_b = drug_features[drug_id_b]
    else:
        print(f"  Computing 3D descriptors for Drug B...")
        feat_b = extract_3d_descriptors_from_smiles(smiles_b)

    # ── Predict ──
    if cell_line_id == "all":
        target_cls = sorted(cell_embeddings.keys())
    else:
        target_cls = [cell_line_id]

    rows = []
    for cl in target_cls:
        if cl not in cell_embeddings:
            rows.append({"cell_line": cl, "predicted_synergy": None,
                         "interpretation": "cell line not found"})
            continue
        feat = np.concatenate([feat_a, feat_b, cell_embeddings[cl]]).reshape(1, -1)
        feat_s = scaler.transform(feat)
        y_hat  = float(model.predict(feat_s)[0])

        if y_hat > 10:
            interp = "Strong Synergy"
        elif y_hat > 0:
            interp = "Mild Synergy"
        elif y_hat > -10:
            interp = "Additive / Neutral"
        else:
            interp = "Antagonistic"

        rows.append({"cell_line": cl, "predicted_synergy": round(y_hat, 3),
                     "interpretation": interp})

    return pd.DataFrame(rows)


# ── CLI mode ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print(" Phase 2 — Step 4: Synergy Predictor")
    print("=" * 60)

    # Check all files exist
    for fpath in [DRUG_FEAT_PKL, CELL_EMBED_PKL, MODEL_PKL]:
        if not os.path.exists(fpath):
            print(f"[ERROR] Missing file: {fpath}")
            print("  Please run step1, step2, and step3 first.")
            exit(1)

    drug_features, cell_embeddings, bundle = load_resources()
    print(f"\nModel loaded   : {bundle['model_name']}")
    print(f"Feature dim    : {bundle['feature_dim']}")
    print(f"Drugs in cache : {len(drug_features)}")
    print(f"Cell lines     : {sorted(cell_embeddings.keys())}")

    # ── Example 1: Known drug pair from the dataset ────────────────────────────
    print("\n" + "-" * 60)
    print(" Example 1: CARBOPLATIN + 5-FU (known synergistic pair)")
    print("-" * 60)

    # Get SMILES from dataset
    df = pd.read_pickle("/home/sanjay/Documents/Final Year/Code/data/oncopolypharmacology.pkl")
    smiles_map = dict(zip(df["Drug1_ID"], df["Drug1"]))
    smiles_map.update(dict(zip(df["Drug2_ID"], df["Drug2"])))

    results = predict(
        smiles_a=smiles_map.get("CARBOPLATIN", ""),
        smiles_b=smiles_map.get("5-FU", ""),
        cell_line_id="all",
        drug_id_a="CARBOPLATIN",
        drug_id_b="5-FU",
        drug_features=drug_features,
        cell_embeddings=cell_embeddings,
        bundle=bundle,
    )
    print(results.sort_values("predicted_synergy", ascending=False).to_string(index=False))

    # ── Example 2: Novel SMILES (not in training set) ──────────────────────────
    print("\n" + "-" * 60)
    print(" Example 2: Novel drug pair (SMILES input)")
    print("-" * 60)
    novel_smiles_a = input("\n Enter SMILES for Drug A (or press Enter to skip): ").strip()
    if novel_smiles_a:
        novel_smiles_b = input(" Enter SMILES for Drug B: ").strip()
        cl_choice      = input(" Cell line (or 'all'): ").strip() or "all"
        try:
            res2 = predict(novel_smiles_a, novel_smiles_b, cell_line_id=cl_choice,
                           drug_features=drug_features, cell_embeddings=cell_embeddings,
                           bundle=bundle)
            print(res2.sort_values("predicted_synergy", ascending=False).to_string(index=False))
        except Exception as e:
            print(f"[ERROR] {e}")
    else:
        print("Skipped.")

    print("\nDone.")
