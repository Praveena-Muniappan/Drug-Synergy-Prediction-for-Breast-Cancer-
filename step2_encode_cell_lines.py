"""
=============================================================
 Phase 2 — Step 2: Cell Line Gene Expression Encoding
=============================================================
Each row in the dataset has a Cell_Line column containing an
8785-dimensional gene expression vector.

This script:
  1. Extracts the unique gene expression vector for each of the
     39 cell lines
  2. Fits StandardScaler + PCA(n_components=50)
  3. Saves the 50-dim embedding for each cell line
  4. Saves the fitted scaler and PCA for use at inference time

Output:
  data/cell_line_embeddings.pkl   { cell_line_id -> np.array(50,) }
  data/cell_line_pca_model.pkl    { 'scaler': ..., 'pca': ... }
  data/cell_line_pca_report.csv
=============================================================
"""

import os, pickle, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(BASE_DIR, "data")
DATASET      = "/home/sanjay/Documents/Final Year/Code/data/oncopolypharmacology.pkl"
OUT_EMBED    = os.path.join(DATA_DIR, "cell_line_embeddings.pkl")
OUT_MODEL    = os.path.join(DATA_DIR, "cell_line_pca_model.pkl")
OUT_CSV      = os.path.join(DATA_DIR, "cell_line_pca_report.csv")
N_COMPONENTS = 38

os.makedirs(DATA_DIR, exist_ok=True)

print("=" * 60)
print(" Phase 2 — Step 2: Cell Line Gene Expression Encoding")
print("=" * 60)

df = pd.read_pickle(DATASET)
print(f"\nDataset shape : {df.shape}")
print(f"Cell lines    : {df['Cell_Line_ID'].nunique()}")

# ── 1. Extract one unique expression vector per cell line ─────────────────────
# The Cell_Line column holds the same array for every row with the same Cell_Line_ID
print("\nExtracting unique cell line expression vectors...")
cl_records = {}
for _, row in df.iterrows():
    cid = row["Cell_Line_ID"]
    if cid not in cl_records:
        cl_records[cid] = row["Cell_Line"]   # numpy array (8785,)

cell_line_ids = sorted(cl_records.keys())
print(f"  Unique cell lines extracted: {len(cell_line_ids)}")

# Build matrix: (n_cell_lines, 8785)
X_raw = np.vstack([cl_records[cid] for cid in cell_line_ids])
print(f"  Raw expression matrix shape: {X_raw.shape}")
print(f"  Contains NaN: {np.isnan(X_raw).any()}")

# Replace any NaN with column mean
col_means = np.nanmean(X_raw, axis=0)
inds = np.where(np.isnan(X_raw))
X_raw[inds] = np.take(col_means, inds[1])

# ── 2. Scale + PCA ────────────────────────────────────────────────────────────
print(f"\nFitting StandardScaler + PCA(n_components={N_COMPONENTS})...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

pca = PCA(n_components=N_COMPONENTS, random_state=42)
X_pca = pca.fit_transform(X_scaled)

var_explained = pca.explained_variance_ratio_.sum() * 100
print(f"  Explained variance with {N_COMPONENTS} components: {var_explained:.1f}%")

# ── 3. Save embeddings ────────────────────────────────────────────────────────
cell_line_embeddings = {cid: X_pca[i] for i, cid in enumerate(cell_line_ids)}

with open(OUT_EMBED, "wb") as f:
    pickle.dump(cell_line_embeddings, f)

with open(OUT_MODEL, "wb") as f:
    pickle.dump({"scaler": scaler, "pca": pca}, f)

# Report
report_rows = []
for i, cid in enumerate(cell_line_ids):
    emb = X_pca[i]
    report_rows.append({
        "cell_line_id": cid,
        "embed_mean": round(float(emb.mean()), 4),
        "embed_std":  round(float(emb.std()),  4),
        "embed_norm": round(float(np.linalg.norm(emb)), 4),
    })
pd.DataFrame(report_rows).to_csv(OUT_CSV, index=False)

print("\n" + "=" * 60)
print(f"  Cell lines encoded  : {len(cell_line_embeddings)}")
print(f"  Embedding dimension : {N_COMPONENTS}")
print(f"  Variance retained   : {var_explained:.1f}%")
print(f"  Embeddings PKL      : {OUT_EMBED}")
print(f"  PCA model PKL       : {OUT_MODEL}")
print(f"  Report CSV          : {OUT_CSV}")
print("=" * 60)
