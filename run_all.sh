#!/bin/bash
# =============================================================
#  Phase 2 — Full Pipeline Runner
#  Run from the phase2_3d_synergy/ directory
# =============================================================
set -e

PYTHON="/home/sanjay/Documents/Final Year/Code/venv_project/bin/python"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo "============================================================="
echo " Phase 2 — 3D Drug Synergy Prediction Pipeline"
echo "============================================================="
echo ""

echo "[Step 1/5] Extracting 3D molecular descriptors..."
"$PYTHON" "$DIR/step1_extract_3d_features.py"
echo ""

echo "[Step 2/5] Encoding cell line gene expression (PCA)..."
"$PYTHON" "$DIR/step2_encode_cell_lines.py"
echo ""

echo "[Step 3/5] Training regression models..."
"$PYTHON" "$DIR/step3_train_model.py"
echo ""

echo "[Step 4/5] Testing predictor on known pairs..."
"$PYTHON" -c "
import sys, os
sys.path.insert(0, '$DIR')
from step4_predict_synergy import predict, load_resources
import pandas as pd
df = load_resources()
drug_features, cell_embeddings, bundle = df
import pickle
raw = pd.read_pickle('/home/sanjay/Documents/Final Year/Code/data/oncopolypharmacology.pkl')
smap = dict(zip(raw['Drug1_ID'], raw['Drug1']))
smap.update(dict(zip(raw['Drug2_ID'], raw['Drug2'])))
res = predict(smap['CARBOPLATIN'], smap['5-FU'], 'all', 'CARBOPLATIN', '5-FU',
              drug_features, cell_embeddings, bundle)
print(res.sort_values('predicted_synergy', ascending=False).head(5).to_string(index=False))
"
echo ""

echo "[Step 5/5] Generating interactive dashboard..."
"$PYTHON" "$DIR/step5_dashboard.py"
echo ""

echo "============================================================="
echo " Pipeline complete! Dashboard opened in browser."
echo "============================================================="
