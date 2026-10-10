3D Aware Drug Synergy Prediction

## Overview

This folder implements **Phase 2** of the Topology-Aware Drug Synergy Engine.

It moves beyond flat 2D identifiers to **actual 3D molecular geometry features**,
combined with **cancer cell line gene expression**, to predict the continuous
synergy score Y for any drug pair × cell line combination.

## Pipeline

```
step1_extract_3d_features.py   →  633-dim 3D descriptor per drug
step2_encode_cell_lines.py     →  50-dim PCA embedding per cell line
step3_train_model.py           →  regression model on Y
step4_predict_synergy.py       →  inference on new drug pairs
step5_dashboard.py             →  interactive 3D dashboard with predictions
```

## Feature Vector

| Component | Source | Dimension |
|-----------|--------|-----------|
| Drug 1 — 3D descriptors | RDKit (AUTOCORR3D + RDF + MORSE + WHIM + shape) | 633 |
| Drug 2 — 3D descriptors | RDKit (same set) | 633 |
| Cell line expression | PCA(50) on 8785-dim gene expression | 50 |
| **Total** | | **1316** |

## 3D Descriptor Breakdown

| Descriptor | # Features | What it captures |
|-----------|-----------|-----------------|
| AUTOCORR3D | 80 | Spatial autocorrelations of atomic properties |
| RDF | 210 | Radial distribution functions (atom-pair distances) |
| MORSE | 224 | 3D-MoRSE scattering codes |
| WHIM | 114 | Weighted holistic invariant (shape + size + symmetry) |
| Shape (PMI/NPR) | 5 | Globularity, asphericity, eccentricity |

## Quick Start

```bash
# Run the full pipeline in one command:
bash run_all.sh

# Or step by step:
python step1_extract_3d_features.py
python step2_encode_cell_lines.py
python step3_train_model.py
python step4_predict_synergy.py
python step5_dashboard.py
```

## Output Files (in data/)

| File | Description |
|------|-------------|
| `drug_3d_features.pkl` | 3D feature vectors for all 37 drugs |
| `cell_line_embeddings.pkl` | PCA embeddings for all 39 cell lines |
| `cell_line_pca_model.pkl` | Fitted scaler + PCA (for inference) |
| `synergy_model_3d.pkl` | Trained regression model bundle |
| `training_results.csv` | RMSE / Pearson r / R² for all models |
| `drug_3d_features_report.csv` | Per-drug extraction report |

## Validation Strategy

- **Split A (Random):** 80/20 random holdout — baseline comparison
- **Split B (Cold-start):** 20% of drug PAIRS held out entirely —
  validates generalisation to *novel, unseen* drug combinations
#output results
<img width="923" height="385" alt="WhatsApp Image 2026-10-10 at 21 40 30" src="https://github.com/user-attachments/assets/5e2873bf-a42e-4700-b79a-dc8dabca1060" />
<img width="923" height="385" alt="2" src="https://github.com/user-attachments/assets/43871870-c293-40e8-a68a-5e385c851dc6" />
<img width="923" height="385" alt="WhatsApp Image 2026-10-10 at 21 40 29" src="https://github.com/user-attachments/assets/6c6135b9-67f0-4961-a185-7c05b35e9c10" />
<img width="923" height="385" alt="3" src="https://github.com/user-attachments/assets/b9a13345-9602-4984-a9fc-5d39eb36f5f6" />
<img width="923" height="385" alt="4" src="https://github.com/user-attachments/assets/35473d4a-8774-4b04-8352-94b3ac169740" />
<img width="1600" height="845" alt="5" src="https://github.com/user-attachments/assets/626d38ff-b892-4689-bb08-abf0574773aa" />
<img width="1550" height="717" alt="image" src="https://github.com/user-attachments/assets/ca3b123b-42a8-4eec-ac5e-21c37f3cd6f2" />
