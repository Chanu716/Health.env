import joblib

# Check scaler
scaler = joblib.load("/models/scaler.pkl")
print("=== SCALER ===")
print("Number of features expected:", scaler.n_features_in_)
print("Feature names:", getattr(scaler, 'feature_names_in_', 'Not saved'))

# Check KNN model
knn = joblib.load("models/knn_dengue_model.pkl")
print("\n=== KNN MODEL ===")
print("KNN n_features_in_:", getattr(knn, 'n_features_in_', 'Not available'))
print("KNN feature names:", getattr(knn, 'feature_names_in_', 'Not saved'))