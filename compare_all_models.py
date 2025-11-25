"""
Model Comparison Script for Dengue Risk Prediction
Compares KNN with XGBoost and other models to determine the best performer
"""

import os
import pickle
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Paths
MODEL_DIR = 'models'
DATA_PATH = 'data/dengue_data_cleaned.csv'
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Feature names (must match training order)
FEATURE_NAMES = [
    'Year', 'Week', 'Temperature (°C)', 'Rainfall (mm)', 
    'Humidity (%)', 'AQI', 'Mosquito Density', 
    'Population Density', 'Dengue Cases Reported',
    'Latitude', 'Longitude', 'Month_Num'
]

print("=" * 80)
print("🦟 DENGUE RISK PREDICTION - MODEL COMPARISON")
print("=" * 80)

# Load data
print("\n📊 Loading data from:", DATA_PATH)
df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Prepare features and target
# Try to match feature names flexibly
def normalize_name(name):
    """Normalize column names for matching"""
    return name.lower().replace(' ', '').replace('(', '').replace(')', '').replace('°', '').replace('%', '')

df_cols_normalized = {normalize_name(col): col for col in df.columns}
matched_features = []

for fname in FEATURE_NAMES:
    norm_fname = normalize_name(fname)
    if norm_fname in df_cols_normalized:
        matched_features.append(df_cols_normalized[norm_fname])
    else:
        # Try partial matching
        for norm_col, actual_col in df_cols_normalized.items():
            if norm_fname in norm_col or norm_col in norm_fname:
                matched_features.append(actual_col)
                break

print(f"\n✅ Matched {len(matched_features)}/{len(FEATURE_NAMES)} features")

if len(matched_features) != len(FEATURE_NAMES):
    print("⚠️ Warning: Not all features matched. Using available columns.")
    # Use all numeric columns except target
    target_cols = ['Outbreak Risk Level', 'Risk_Level', 'risk_code']
    matched_features = [col for col in df.columns if col not in target_cols and df[col].dtype in ['int64', 'float64']]

print(f"Using features: {matched_features}")

# Extract features
X = df[matched_features].fillna(0).values
print(f"Feature matrix shape: {X.shape}")

# Extract target variable
target_mapping = {'Low': 0, 'Moderate': 1, 'High': 2}
if 'Outbreak Risk Level' in df.columns:
    y = df['Outbreak Risk Level'].map(target_mapping)
elif 'Risk_Level' in df.columns:
    y = df['Risk_Level']
elif 'risk_code' in df.columns:
    y = df['risk_code']
else:
    raise ValueError("Target column not found!")

print(f"Target distribution:\n{pd.Series(y).value_counts().sort_index()}")

# Load scaler
scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
print(f"\n🔧 Loading scaler from: {scaler_path}")
try:
    scaler = joblib.load(scaler_path)
    print("✅ Scaler loaded successfully")
except:
    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print("✅ Scaler loaded successfully (pickle)")
    except Exception as e:
        print(f"❌ Failed to load scaler: {e}")
        print("Creating new scaler from data...")
        scaler = StandardScaler()
        scaler.fit(X)

# Scale features
X_scaled = scaler.transform(X)
print("✅ Features scaled")

# Split data for evaluation (same as training split)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain set: {X_train.shape}, Test set: {X_test.shape}")

# Load and evaluate all models
print("\n" + "=" * 80)
print("📈 EVALUATING MODELS")
print("=" * 80)

results = []
model_files = [f for f in os.listdir(MODEL_DIR) 
               if f.endswith('.pkl') and f not in ['scaler.pkl', 'feature_names.pkl']]

print(f"\nFound {len(model_files)} models to evaluate:")
for f in model_files:
    print(f"  - {f}")

for model_file in model_files:
    model_name = model_file.replace('_model.pkl', '').replace('.pkl', '').replace('_', ' ').title()
    model_path = os.path.join(MODEL_DIR, model_file)
    
    print(f"\n{'─' * 80}")
    print(f"🤖 Evaluating: {model_name}")
    print(f"{'─' * 80}")
    
    try:
        # Load model
        try:
            model = joblib.load(model_path)
        except:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        
        print(f"✅ Model loaded: {type(model).__name__}")
        
        # Make predictions on test set
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Store results
        results.append({
            'Model': model_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Model_Type': type(model).__name__
        })
        
        # Print metrics
        print(f"\n📊 Metrics:")
        print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        
        # Print classification report
        print(f"\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['Low', 'Moderate', 'High'],
                                   zero_division=0))
        
        # Print confusion matrix
        print(f"🔢 Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
    except Exception as e:
        print(f"❌ Error evaluating {model_name}: {e}")
        continue

# Create results DataFrame
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Accuracy', ascending=False)

# Save results
output_path = os.path.join(RESULTS_DIR, 'model_comparison_results.csv')
results_df.to_csv(output_path, index=False)
print(f"\n✅ Results saved to: {output_path}")

# Display final comparison
print("\n" + "=" * 80)
print("🏆 FINAL MODEL COMPARISON")
print("=" * 80)
print("\n" + results_df.to_string(index=False))

# Determine best model
best_model = results_df.iloc[0]
print("\n" + "=" * 80)
print("🥇 BEST MODEL")
print("=" * 80)
print(f"\nModel: {best_model['Model']}")
print(f"Type: {best_model['Model_Type']}")
print(f"Accuracy: {best_model['Accuracy']:.4f} ({best_model['Accuracy']*100:.2f}%)")
print(f"Precision: {best_model['Precision']:.4f}")
print(f"Recall: {best_model['Recall']:.4f}")
print(f"F1-Score: {best_model['F1-Score']:.4f}")

# Recommendation
xgboost_row = results_df[results_df['Model'].str.contains('Xgboost', case=False)]
knn_row = results_df[results_df['Model'].str.contains('Knn', case=False)]

if not knn_row.empty and not xgboost_row.empty:
    print("\n" + "=" * 80)
    print("💡 RECOMMENDATION")
    print("=" * 80)
    
    knn_acc = knn_row.iloc[0]['Accuracy']
    xgb_acc = xgboost_row.iloc[0]['Accuracy']
    
    if knn_acc > xgb_acc:
        diff = (knn_acc - xgb_acc) * 100
        print(f"\n✅ KNN outperforms XGBoost by {diff:.2f}%")
        print(f"   KNN Accuracy: {knn_acc:.4f} ({knn_acc*100:.2f}%)")
        print(f"   XGBoost Accuracy: {xgb_acc:.4f} ({xgb_acc*100:.2f}%)")
        print("\n🔄 RECOMMENDED ACTION: Replace XGBoost with KNN in api.py")
        print("   - Update the default model selection")
        print("   - Update model_info endpoint with new KNN metrics")
        print("   - Update README.md with new performance numbers")
    elif xgb_acc > knn_acc:
        diff = (xgb_acc - knn_acc) * 100
        print(f"\n✅ XGBoost outperforms KNN by {diff:.2f}%")
        print(f"   XGBoost Accuracy: {xgb_acc:.4f} ({xgb_acc*100:.2f}%)")
        print(f"   KNN Accuracy: {knn_acc:.4f} ({knn_acc*100:.2f}%)")
        print("\n✓ RECOMMENDED ACTION: Keep XGBoost as the primary model")
    else:
        print("\n⚖️ KNN and XGBoost have similar performance")
        print(f"   Both Accuracy: {knn_acc:.4f} ({knn_acc*100:.2f}%)")
        print("\n💭 Consider other factors:")
        print("   - Inference speed (XGBoost usually faster)")
        print("   - Model interpretability (XGBoost has feature importance)")
        print("   - Memory usage")

print("\n" + "=" * 80)
print("✅ COMPARISON COMPLETE")
print("=" * 80)
