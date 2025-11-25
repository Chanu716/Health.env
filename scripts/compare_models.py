import os
import re
import pickle
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(ROOT, 'models')
DATA_PATH = os.path.join(ROOT, 'data', 'dengue_data_cleaned.csv')
RESULTS_DIR = os.path.join(ROOT, 'results')
RESULTS_PATH = os.path.join(RESULTS_DIR, 'model_comparison_results.csv')

FALLBACK_FEATURE_NAMES = [
    'Year', 'Week', 'Average Temperature (°C)', 'Average Rainfall (mm)',
    'Average Humidity (%)', 'Air Quality Index (AQI)', 'Mosquito Density Index',
    'Population Density (people per sq km)', 'Dengue Cases Reported', 'Latitude', 'Longitude', 'Month_Num'
]

def normalize(name: str) -> str:
    name = name.lower()
    name = name.replace('°', 'deg')
    name = name.replace('%', 'pct')
    name = re.sub(r'[^a-z0-9]', '', name)
    return name

# discover model files (.pkl, .joblib)
models_to_eval = {}
if not os.path.exists(MODEL_DIR):
    raise FileNotFoundError(f"Model directory not found: {MODEL_DIR}")
for fname in os.listdir(MODEL_DIR):
    if fname.endswith('.pkl') or fname.endswith('.joblib'):
        if fname in ('scaler.pkl', 'feature_names.pkl'):
            continue
        key = os.path.splitext(fname)[0]
        models_to_eval[key] = os.path.join(MODEL_DIR, fname)
if not models_to_eval:
    print("No model files found in models/. Place knn_dengue_model.pkl and existing models there.")
    raise SystemExit(1)

# load CSV
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

# load feature names if saved
feature_names_path = os.path.join(MODEL_DIR, 'feature_names.pkl')
if os.path.exists(feature_names_path):
    try:
        with open(feature_names_path, 'rb') as f:
            FEATURE_NAMES = pickle.load(f)
        print(f"Loaded feature order from {feature_names_path}")
    except Exception:
        FEATURE_NAMES = FALLBACK_FEATURE_NAMES
        print("Could not load feature_names.pkl — using fallback list")
else:
    FEATURE_NAMES = FALLBACK_FEATURE_NAMES

# match csv columns to FEATURE_NAMES robustly
csv_cols = list(df.columns)
norm_to_col = {normalize(c): c for c in csv_cols}
matched_cols = []
missing = []
for fname in FEATURE_NAMES:
    n = normalize(fname)
    if n in norm_to_col:
        matched_cols.append(norm_to_col[n])
    else:
        missing.append(fname)

if missing:
    # try fuzzy substring match
    for fname in list(missing):
        n = normalize(fname)
        for col in csv_cols:
            if n in normalize(col) or normalize(col) in n:
                matched_cols.append(col)
                missing.remove(fname)
                break

if missing:
    print("ERROR: Could not find required features in CSV columns:")
    for m in missing:
        print(" -", m)
    print("\nAvailable CSV columns:")
    for c in csv_cols:
        print(" -", c)
    raise RuntimeError("Feature columns mismatch. Provide matching feature_names.pkl or update FALLBACK_FEATURE_NAMES.")

# prepare features and target
X = df[matched_cols].fillna(0).values
# target: outbreak -> map text to numeric if needed
if 'Outbreak Risk Level' in df.columns:
    mapping = {'Low': 0, 'Moderate': 1, 'High': 2}
    y = df['Outbreak Risk Level'].map(mapping)
    if y.isnull().any():
        # if already numeric in different column, try common alternatives
        if 'Risk_Level' in df.columns:
            y = df['Risk_Level']
        elif 'risk_code' in df.columns:
            y = df['risk_code']
else:
    if 'Risk_Level' in df.columns:
        y = df['Risk_Level']
    elif 'risk_code' in df.columns:
        y = df['risk_code']
    else:
        raise RuntimeError("Target column not found (Outbreak Risk Level / Risk_Level / risk_code)")

# load scaler if present; prefer joblib, fallback to pickle
scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
scaler = None
if os.path.exists(scaler_path):
    try:
        scaler = joblib.load(scaler_path)
        print(f"Loaded scaler with joblib from {scaler_path}")
    except Exception as e_joblib:
        try:
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            print(f"Loaded scaler with pickle from {scaler_path}")
        except Exception as e_pickle:
            print("Scaler load failed (joblib and pickle):", e_joblib, e_pickle)
            scaler = None

if scaler is not None:
    try:
        X_scaled = scaler.transform(X)
    except Exception as e:
        print("Scaler transform failed:", e)
        print("Fitting a fresh StandardScaler for evaluation.")
        scaler = StandardScaler().fit(X)
        X_scaled = scaler.transform(X)
else:
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

results = []
for name, path in models_to_eval.items():
    try:
        m = joblib.load(path)
    except Exception:
        try:
            with open(path, 'rb') as f:
                m = pickle.load(f)
        except Exception as e:
            print(f"Skipping {name}: load failed ({e})")
            continue
    # try predict on scaled then raw
    y_pred = None
    for arr in (X_scaled, X):
        try:
            y_pred = m.predict(arr)
            break
        except Exception:
            continue
    if y_pred is None:
        print(f"Skipping {name}: predict failed on both scaled and raw arrays")
        continue
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
    results.append({'model': name, 'accuracy': acc, 'precision': prec, 'recall': rec, 'f1_score': f1})
    print(f"{name}: acc={acc:.4f}, f1={f1:.4f}")

os.makedirs(RESULTS_DIR, exist_ok=True)
pd.DataFrame(results).to_csv(RESULTS_PATH, index=False)
print("Saved comparison to", RESULTS_PATH)