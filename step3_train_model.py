"""
=============================================================
 Phase 2 — Step 3: 3D-Aware Synergy Regression Model
=============================================================
Builds the full feature matrix:
  [Drug1_3D (633) | Drug2_3D (633) | CellLine_PCA (50)] = 1316 features
Trains regressors to predict the continuous synergy score Y.

Two evaluation modes:
  A) Random split      — standard 80/20 holdout
  B) Cold-start split  — hold out entire drug pairs unseen during training
                         This validates generalisation to NOVEL drug combos.

Output:
  data/synergy_model_3d.pkl       — best trained model
  data/training_results.csv       — metrics for all models and splits
=============================================================
"""

import os, pickle, warnings, time
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

warnings.filterwarnings("ignore")

# ── Optional GPU-accelerated libs (graceful fallback) ─────────────────────────
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[warn] xgboost not found — skipping XGBRegressor")

try:
    from lightgbm import LGBMRegressor
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("[warn] lightgbm not found — skipping LGBMRegressor")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(BASE_DIR, "data")
DATASET       = "/home/sanjay/Documents/Final Year/Code/data/oncopolypharmacology.pkl"
DRUG_FEAT_PKL = os.path.join(DATA_DIR, "drug_3d_features.pkl")
CELL_EMBED_PKL= os.path.join(DATA_DIR, "cell_line_embeddings.pkl")
OUT_MODEL     = os.path.join(DATA_DIR, "synergy_model_3d.pkl")
OUT_RESULTS   = os.path.join(DATA_DIR, "training_results.csv")

print("=" * 65)
print(" Phase 2 — Step 3: 3D-Aware Synergy Regression Model Training")
print("=" * 65)

# ── 1. Load pre-computed features ─────────────────────────────────────────────
print("\nLoading pre-computed features...")
with open(DRUG_FEAT_PKL, "rb") as f:
    drug_features = pickle.load(f)     # { drug_id -> np.array(633,) }
with open(CELL_EMBED_PKL, "rb") as f:
    cell_embeddings = pickle.load(f)   # { cell_line_id -> np.array(50,) }

print(f"  Drug features loaded    : {len(drug_features)} drugs")
print(f"  Cell line embeddings    : {len(cell_embeddings)} cell lines")

# ── 2. Build feature matrix from dataset ──────────────────────────────────────
print("\nBuilding feature matrix from dataset...")
df = pd.read_pickle(DATASET)
print(f"  Dataset rows : {len(df)}")

X_rows, Y_vals, pair_ids = [], [], []
skipped = 0

for _, row in df.iterrows():
    d1 = row["Drug1_ID"]
    d2 = row["Drug2_ID"]
    cl = row["Cell_Line_ID"]
    y  = row["Y"]

    if d1 not in drug_features or d2 not in drug_features:
        skipped += 1
        continue
    if cl not in cell_embeddings:
        skipped += 1
        continue

    feat = np.concatenate([drug_features[d1], drug_features[d2], cell_embeddings[cl]])
    X_rows.append(feat)
    Y_vals.append(y)
    pair_ids.append(f"{d1}||{d2}")

X = np.array(X_rows)
Y = np.array(Y_vals)
pairs = np.array(pair_ids)

print(f"  Rows used    : {len(X)}  (skipped {skipped} due to missing features)")
print(f"  Feature dim  : {X.shape[1]}")
print(f"  Y range      : [{Y.min():.2f}, {Y.max():.2f}]  mean={Y.mean():.2f}")

# ── 3. Define models ──────────────────────────────────────────────────────────
models = {}

if HAS_XGB:
    models["XGBoost"] = XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        tree_method="hist", device="cpu",
        random_state=42, verbosity=0
    )
if HAS_LGB:
    models["LightGBM"] = LGBMRegressor(
        n_estimators=300, num_leaves=63, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbose=-1
    )

models["Random Forest"] = RandomForestRegressor(
    n_estimators=200, max_depth=12, n_jobs=-1, random_state=42
)
models["MLP"] = MLPRegressor(
    hidden_layer_sizes=(512, 256, 128), activation="relu",
    max_iter=300, random_state=42, early_stopping=True, verbose=False
)

# ── 4. Evaluation helper ──────────────────────────────────────────────────────
def evaluate(y_true, y_pred, label=""):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    pr   = pearsonr(y_true, y_pred)[0]
    if label:
        print(f"      RMSE={rmse:.3f}  Pearson r={pr:.3f}  R²={r2:.3f}")
    return rmse, pr, r2

# ── 5. Split A — Random 80/20 ─────────────────────────────────────────────────
print("\n" + "-" * 65)
print(" Split A: Random 80/20 holdout")
print("-" * 65)

Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.2, random_state=42)

# Scale features
scaler_A = StandardScaler()
Xtr_s = scaler_A.fit_transform(Xtr)
Xte_s = scaler_A.transform(Xte)

results_A = []
trained_A = {}

for name, model in models.items():
    print(f"\n  Training [{name}]...")
    t0 = time.time()
    model.fit(Xtr_s, Ytr)
    preds = model.predict(Xte_s)
    elapsed = time.time() - t0
    rmse, pr, r2 = evaluate(Yte, preds, label=name)
    results_A.append({"split": "random", "model": name,
                       "rmse": round(rmse,3), "pearson_r": round(pr,3),
                       "r2": round(r2,3), "time_s": round(elapsed,1)})
    trained_A[name] = model
    print(f"      Time: {elapsed:.1f}s")

# ── 6. Split B — Cold-Start (held-out drug pairs) ─────────────────────────────
print("\n" + "-" * 65)
print(" Split B: Cold-start — held-out drug pairs (NOVEL COMBO TEST)")
print("-" * 65)

unique_pairs = np.unique(pairs)
np.random.seed(42)
np.random.shuffle(unique_pairs)
n_test_pairs = max(1, int(len(unique_pairs) * 0.20))
test_pairs_set  = set(unique_pairs[:n_test_pairs])
train_pairs_set = set(unique_pairs[n_test_pairs:])

train_mask = np.array([p in train_pairs_set for p in pairs])
test_mask  = np.array([p in test_pairs_set  for p in pairs])

Xtr_cs, Ytr_cs = X[train_mask], Y[train_mask]
Xte_cs, Yte_cs = X[test_mask],  Y[test_mask]

print(f"\n  Held-out drug pairs    : {n_test_pairs} / {len(unique_pairs)}")
print(f"  Train rows             : {Xtr_cs.shape[0]}")
print(f"  Test  rows (novel)     : {Xte_cs.shape[0]}")
print(f"  Test pairs (sample)    : {list(test_pairs_set)[:3]}")

scaler_B = StandardScaler()
Xtr_cs_s = scaler_B.fit_transform(Xtr_cs)
Xte_cs_s = scaler_B.transform(Xte_cs)

results_B = []
trained_B = {}

for name, model in models.items():
    # Re-initialise fresh model for cold-start
    fresh = type(model)(**model.get_params())
    print(f"\n  Training [{name}] on non-held-out pairs...")
    t0 = time.time()
    fresh.fit(Xtr_cs_s, Ytr_cs)
    preds = fresh.predict(Xte_cs_s)
    elapsed = time.time() - t0
    rmse, pr, r2 = evaluate(Yte_cs, preds, label=name)
    results_B.append({"split": "cold_start", "model": name,
                       "rmse": round(rmse,3), "pearson_r": round(pr,3),
                       "r2": round(r2,3), "time_s": round(elapsed,1)})
    trained_B[name] = fresh
    print(f"      Time: {elapsed:.1f}s")

# ── 7. Save best model (based on cold-start Pearson r) ────────────────────────
all_results = pd.DataFrame(results_A + results_B)
all_results.to_csv(OUT_RESULTS, index=False)

best_cs = max(results_B, key=lambda r: r["pearson_r"])
best_name = best_cs["model"]
best_model = trained_B[best_name]

bundle = {
    "model":       best_model,
    "scaler":      scaler_B,
    "model_name":  best_name,
    "feature_dim": X.shape[1],
    "drug_feat_dim": next(iter(drug_features.values())).shape[0],
    "cell_feat_dim": next(iter(cell_embeddings.values())).shape[0],
    "metrics_random": {r["model"]: r for r in results_A},
    "metrics_coldstart": {r["model"]: r for r in results_B},
}
with open(OUT_MODEL, "wb") as f:
    pickle.dump(bundle, f)

# ── 8. Summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print(" RESULTS SUMMARY")
print("=" * 65)
print("\n[A] Random Split Results:")
print(all_results[all_results.split=="random"][["model","rmse","pearson_r","r2"]].to_string(index=False))
print("\n[B] Cold-Start Results (NOVEL DRUG PAIRS):")
print(all_results[all_results.split=="cold_start"][["model","rmse","pearson_r","r2"]].to_string(index=False))
print(f"\nBest model (cold-start Pearson r): {best_name}  r={best_cs['pearson_r']}")
print(f"\nModel saved  -> {OUT_MODEL}")
print(f"Results CSV  -> {OUT_RESULTS}")
print("=" * 65)
