"""
Train KNN Model and Scaler with 15 Features
This script properly trains both the scaler and KNN model
"""

import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("🤖 TRAINING KNN MODEL AND SCALER WITH 15 FEATURES")
print("=" * 80)

# Load data
print("\n📊 Loading data...")
df = pd.read_csv('data/dengue_india_weekly_with_nulls.csv')
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Define the 15 features to use (excluding target 'Outbreak Risk Level')
# We'll use all numeric features + encoded categorical features
FEATURES_15 = [
    'Year',
    'Month',  # Numeric (or we can use Month_Num instead)
    'Week',
    'Average Temperature (°C)',
    'Average Rainfall (mm)',
    'Average Humidity (%)',
    'Air Quality Index (AQI)',
    'Mosquito Density Index',
    'Population Density (people per sq km)',
    'Dengue Cases Reported',
    'Latitude',
    'Longitude',
    'Month_Num',
    # Need 2 more features - let's encode Season (4 categories = 3 dummy vars)
]

print("\n🔧 Preparing features...")

# Create dummy variables for Season (one-hot encoding)
season_dummies = pd.get_dummies(df['Season'], prefix='Season')
print(f"Season categories: {season_dummies.columns.tolist()}")

# We'll drop one dummy to avoid multicollinearity (keep 3 out of 4)
# This gives us the 15 features total
season_cols = season_dummies.columns.tolist()[:3]  # Keep first 3 season dummies

# Combine base features with season dummies
base_features = [
    'Year',
    'Week',
    'Average Temperature (°C)',
    'Average Rainfall (mm)',
    'Average Humidity (%)',
    'Air Quality Index (AQI)',
    'Mosquito Density Index',
    'Population Density (people per sq km)',
    'Dengue Cases Reported',
    'Latitude',
    'Longitude',
    'Month_Num'
]

# Create the feature matrix
X_base = df[base_features].fillna(0)
X_season = season_dummies[season_cols]
X = pd.concat([X_base, X_season], axis=1)

print(f"\n✅ Final feature set ({X.shape[1]} features):")
for i, col in enumerate(X.columns, 1):
    print(f"  {i}. {col}")

# Prepare target
print("\n🎯 Preparing target variable...")
target_mapping = {'Low': 0, 'Moderate': 1, 'High': 2}
y = df['Outbreak Risk Level'].map(target_mapping)
print(f"Target distribution:\n{y.value_counts().sort_index()}")

# Convert to numpy arrays
X_array = X.values
y_array = y.values

# Train-test split (same as original: 80-20)
print("\n📊 Splitting data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_array, y_array, test_size=0.2, random_state=42, stratify=y_array
)
print(f"Train set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Train scaler
print("\n🔧 Training StandardScaler...")
scaler = StandardScaler()
scaler.fit(X_train)
print(f"✅ Scaler trained on {scaler.n_features_in_} features")

# Transform data
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✅ Data scaled")

# Train KNN with hyperparameter tuning
print("\n🤖 Training KNN Classifier...")
print("⏳ Running GridSearchCV to find best hyperparameters...")

# Define parameter grid
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11, 15],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

# Create KNN classifier
knn_base = KNeighborsClassifier()

# Grid search with cross-validation
grid_search = GridSearchCV(
    knn_base,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_scaled, y_train)

# Get best model
knn_model = grid_search.best_estimator_
print(f"\n✅ Best KNN parameters: {grid_search.best_params_}")
print(f"✅ Best CV score: {grid_search.best_score_:.4f}")

# Evaluate on test set
print("\n📊 Evaluating on test set...")
y_pred = knn_model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print(f"\n🎯 KNN Model Performance:")
print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1-Score:  {f1:.4f}")

print(f"\n📋 Detailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Low', 'Moderate', 'High'], zero_division=0))

# Save scaler
scaler_path = 'models/scaler.pkl'
print(f"\n💾 Saving scaler to {scaler_path}...")
joblib.dump(scaler, scaler_path)
print("✅ Scaler saved")

# Save KNN model
knn_path = 'models/knn_dengue_model.pkl'
print(f"\n💾 Saving KNN model to {knn_path}...")
joblib.dump(knn_model, knn_path)
print("✅ KNN model saved")

# Save feature names for reference
feature_names = X.columns.tolist()
feature_names_path = 'models/feature_names_15.pkl'
print(f"\n💾 Saving feature names to {feature_names_path}...")
with open(feature_names_path, 'wb') as f:
    pickle.dump(feature_names, f)
print("✅ Feature names saved")

# Verify saved models
print("\n" + "=" * 80)
print("🔍 VERIFICATION")
print("=" * 80)

print("\n✅ Loading saved scaler...")
loaded_scaler = joblib.load(scaler_path)
print(f"   Scaler expects {loaded_scaler.n_features_in_} features")

print("\n✅ Loading saved KNN model...")
loaded_knn = joblib.load(knn_path)
print(f"   KNN type: {type(loaded_knn).__name__}")
print(f"   KNN parameters: {loaded_knn.get_params()}")

print("\n✅ Testing prediction with saved models...")
sample = X_test_scaled[:1]
sample_scaled = loaded_scaler.transform(X_test[:1])
prediction = loaded_knn.predict(sample_scaled)
print(f"   Sample prediction: {prediction[0]} ({['Low', 'Moderate', 'High'][prediction[0]]})")

print("\n" + "=" * 80)
print("✅ TRAINING COMPLETE!")
print("=" * 80)
print(f"\nSaved files:")
print(f"  - {scaler_path}")
print(f"  - {knn_path}")
print(f"  - {feature_names_path}")
print(f"\nKNN Model Performance: {accuracy*100:.2f}% accuracy")
print("\nNext step: Run 'python compare_all_models.py' to compare with other models")
