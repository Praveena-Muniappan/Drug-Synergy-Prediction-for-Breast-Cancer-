"""
=========================================================
 Breast Cancer Drug Synergy Prediction
 Final Optimized Pipeline
=========================================================
"""
import time
import pandas as pd
import joblib
import xgboost as xgb
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# --- 1. Load Data ---
df = pd.read_csv('/home/sanjay/Documents/Final Year/Code/dataset/processed_dataset.csv')
X = df.drop(columns=['label'])
y = df['label']

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 2. Model Configuration ---
# Models that follow standard Scikit-Learn fit/predict API
models_sklearn = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=42)),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "LightGBM": LGBMClassifier(device="gpu", random_state=42, verbose=-1),
    "CatBoost": CatBoostClassifier(task_type="GPU", devices="0", verbose=False, random_state=42)
}

# --- 3. Training Loop ---
results = []
trained_models = {}

print(f"\nStarting training on {len(X_train)} samples...")

# Train Scikit-Learn API models
for name, model in tqdm(models_sklearn.items(), desc="Training Models"):
    start_time = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    results.append({"Model": name, "Accuracy": accuracy_score(y_test, y_pred), "F1-Score": f1_score(y_test, y_pred), "ROC AUC": roc_auc_score(y_test, y_prob), "Time (s)": round(time.time() - start_time, 2)})
    trained_models[name] = model

# Train XGBoost with DMatrix for GPU optimization
start_time = time.time()
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
xgb_params = {'tree_method': 'hist', 'device': 'cuda', 'eval_metric': 'logloss', 'objective': 'binary:logistic'}
xgb_model = xgb.train(xgb_params, dtrain, num_boost_round=100)
y_prob = xgb_model.predict(dtest)
y_pred = (y_prob > 0.5).astype(int)
results.append({"Model": "XGBoost", "Accuracy": accuracy_score(y_test, y_pred), "F1-Score": f1_score(y_test, y_pred), "ROC AUC": roc_auc_score(y_test, y_prob), "Time (s)": round(time.time() - start_time, 2)})
trained_models["XGBoost"] = xgb_model

# --- 4. Export Results and Save Best Model ---
results_df = pd.DataFrame(results).sort_values(by="F1-Score", ascending=False)
results_df.to_csv('model_performance_summary.csv', index=False)

print("\n" + "="*50)
print("          MODEL PERFORMANCE SUMMARY SAVED")
print("="*50)
print(results_df.to_string(index=False))
print("="*50)

best_model_name = results_df.iloc[0]["Model"]
joblib.dump(trained_models[best_model_name], 'best_synergy_model.pkl')
print(f"\nSummary saved. Best model ('{best_model_name}') saved to 'best_synergy_model.pkl'.")