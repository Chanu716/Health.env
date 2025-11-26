"""
Dengue Risk Prediction API
Serves the trained XGBoost model for real-time predictions
"""

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import os
import pickle
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)  # Enable CORS for frontend access


@app.route('/', methods=['GET'])
def serve_index():
    """Render the dashboard template from `templates/index.html`."""
    try:
        return render_template('index.html')
    except Exception:
        return jsonify({'error': 'Index file not found'}), 404


@app.route('/<path:filename>', methods=['GET'])
def serve_files(filename):
    # Serve JS/CSS and other static assets from repo root if present
    root = os.path.dirname(__file__)
    possible = os.path.join(root, filename)
    if os.path.exists(possible):
        return send_from_directory(root, filename)
    # fallback to the Flask static folder
    static_path = os.path.join(root, app.static_folder)
    target = os.path.join(static_path, filename)
    if os.path.exists(target):
        return send_from_directory(static_path, filename)
    return jsonify({'error': f'File {filename} not found'}), 404

# --- Model loader: support multiple serialized models ---
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
MODEL_REGISTRY = {}
SCALER = None

# load scaler if available (prefer joblib.load for joblib-saved artifacts)
scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
try:
    if os.path.exists(scaler_path):
        try:
            SCALER = joblib.load(scaler_path)
            print("Loaded scaler (joblib) from", scaler_path)
        except Exception:
            # fallback to pickle
            with open(scaler_path, 'rb') as f:
                SCALER = pickle.load(f)
            print("Loaded scaler (pickle) from", scaler_path)
except Exception as e:
    print("Scaler load failed:", e)
    SCALER = None

# load models present in models/
for fn in os.listdir(MODEL_DIR):
    if fn.endswith('.pkl') or fn.endswith('.joblib'):
        if fn in ('scaler.pkl', 'feature_names.pkl'):
            continue
        name = os.path.splitext(fn)[0]
        model_path = os.path.join(MODEL_DIR, fn)
        try:
            # Prefer joblib.load for models saved with joblib
            try:
                MODEL_REGISTRY[name] = joblib.load(model_path)
                print(f"Loaded model (joblib): {name}")
                continue
            except Exception:
                pass

            # Fallback to pickle.load
            with open(model_path, 'rb') as f:
                MODEL_REGISTRY[name] = pickle.load(f)
            print(f"Loaded model (pickle): {name}")
        except Exception as e:
            print("Could not load model", fn, e)

# Feature names (must match training data order)
# Try to load feature names file from models/ if available (e.g. feature_names_15.pkl)
FEATURE_NAMES = None
for fn in os.listdir(MODEL_DIR):
    if fn.startswith('feature_names') and fn.endswith('.pkl'):
        try:
            with open(os.path.join(MODEL_DIR, fn), 'rb') as f:
                FEATURE_NAMES = pickle.load(f)
            print(f"Loaded feature names from {fn}: {len(FEATURE_NAMES)} features")
            break
        except Exception as e:
            print(f"Failed to load feature names from {fn}: {e}")

# If scaler is present and feature names length doesn't match, try to find a matching file
if SCALER is not None:
    scaler_n = getattr(SCALER, 'n_features_in_', None)
    if scaler_n is not None and FEATURE_NAMES is not None and len(FEATURE_NAMES) != scaler_n:
        # search for a feature_names file matching scaler_n
        for fn in os.listdir(MODEL_DIR):
            if fn.startswith('feature_names') and fn.endswith('.pkl'):
                try:
                    with open(os.path.join(MODEL_DIR, fn), 'rb') as f:
                        names = pickle.load(f)
                    if isinstance(names, (list, tuple)) and len(names) == scaler_n:
                        FEATURE_NAMES = names
                        print(f"Switched to feature names from {fn} to match scaler ({scaler_n} features)")
                        break
                except Exception:
                    continue

# Fallback default feature list (keeps backward compatibility)
if FEATURE_NAMES is None:
    FEATURE_NAMES = [
        'Year', 'Week', 'Temperature (°C)', 'Rainfall (mm)', 
        'Humidity (%)', 'AQI', 'Mosquito Density', 
        'Population Density', 'Dengue Cases Reported',
        'Latitude', 'Longitude', 'Month_Num'
    ]

# City coordinates mapping
CITY_COORDS = {
    'Mumbai': {'lat': 19.0760, 'lon': 72.8777},
    'Delhi': {'lat': 28.7041, 'lon': 77.1025},
    'Bangalore': {'lat': 12.9716, 'lon': 77.5946},
    'Hyderabad': {'lat': 17.3850, 'lon': 78.4867},
    'Chennai': {'lat': 13.0827, 'lon': 80.2707},
    'Kolkata': {'lat': 22.5726, 'lon': 88.3639},
    'Pune': {'lat': 18.5204, 'lon': 73.8567},
    'Ahmedabad': {'lat': 23.0225, 'lon': 72.5714},
    'Jaipur': {'lat': 26.9124, 'lon': 75.7873},
    'Lucknow': {'lat': 26.8467, 'lon': 80.9462},
    'Kanpur': {'lat': 26.4499, 'lon': 80.3319},
    'Nagpur': {'lat': 21.1458, 'lon': 79.0882},
    'Indore': {'lat': 22.7196, 'lon': 75.8577},
    'Thane': {'lat': 19.2183, 'lon': 72.9781},
    'Bhopal': {'lat': 23.2599, 'lon': 77.4126},
    'Visakhapatnam': {'lat': 17.6868, 'lon': 83.2185},
    'Pimpri-Chinchwad': {'lat': 18.6298, 'lon': 73.7997},
    'Patna': {'lat': 25.5941, 'lon': 85.1376},
    'Vadodara': {'lat': 22.3072, 'lon': 73.1812},
    'Ghaziabad': {'lat': 28.6692, 'lon': 77.4538},
    'Ludhiana': {'lat': 30.9010, 'lon': 75.8573},
    'Agra': {'lat': 27.1767, 'lon': 78.0081},
    'Nashik': {'lat': 19.9975, 'lon': 73.7898},
    'Faridabad': {'lat': 28.4089, 'lon': 77.3178},
    'Meerut': {'lat': 28.9845, 'lon': 77.7064},
    'Rajkot': {'lat': 22.3039, 'lon': 70.8022},
    'Kalyan-Dombivli': {'lat': 19.2403, 'lon': 73.1305},
    'Vasai-Virar': {'lat': 19.4612, 'lon': 72.7985},
    'Varanasi': {'lat': 25.3176, 'lon': 82.9739},
    'Srinagar': {'lat': 34.0837, 'lon': 74.7973}
}

# Month mapping
MONTH_MAP = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12
}

# Risk level mapping (Low=0, Moderate=1, High=2)
RISK_LABELS = {0: 'Low', 1: 'Moderate', 2: 'High'}
RISK_COLORS = {0: '#00ccff', 1: '#00ff88', 2: '#ff3366'}
RISK_ICONS = {0: '🟢', 1: '🟡', 2: '🔴'}


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': len(MODEL_REGISTRY) > 0,
        'model_type': list(MODEL_REGISTRY.keys()) if len(MODEL_REGISTRY) > 0 else None
    })


# prediction route example (select model_name optional)
@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint
    Expected JSON input:
    {
        "city": "Mumbai",
        "month": "July",
        "temperature": 32.5,
        "rainfall": 1200,
        "humidity": 85,
        "aqi": 150,
        "mosquito": 0.75,
        "population": 5000,
        "cases": 450
    }
    """
    try:
        # Get input data
        data = request.get_json()
        
        # Extract values
        city = data.get('city', 'Mumbai')
        month = data.get('month', 'July')
        temperature = float(data.get('temperature', 30))
        rainfall = float(data.get('rainfall', 1000))
        humidity = float(data.get('humidity', 80))
        aqi = float(data.get('aqi', 100))
        mosquito = float(data.get('mosquito', 0.5))
        population = float(data.get('population', 3000))
        cases = float(data.get('cases', 300))
        
        # Get city coordinates
        coords = CITY_COORDS.get(city, {'lat': 20.0, 'lon': 77.0})
        
        # Get current year and week (or use defaults)
        import datetime
        now = datetime.datetime.now()
        year = now.year
        week = now.isocalendar()[1]
        
        # Get month number
        month_num = MONTH_MAP.get(month, 7)
        
        # Build feature vector using FEATURE_NAMES and pad/match to scaler expectations
        def build_feature_vector(input_data):
            """Return a numpy array shaped (1, n_features) matching the scaler's expected input.

            It maps known input keys to common feature names and pads missing features with 0.
            """
            # simple mapping from common tokens to input keys
            key_map = {
                'year': year,
                'week': week,
                'temperature': temperature,
                'average temperature (°c)': temperature,
                'rainfall': rainfall,
                'average rainfall (mm)': rainfall,
                'humidity': humidity,
                'average humidity (%)': humidity,
                'aqi': aqi,
                'air quality index (aqi)': aqi,
                'mosquito': mosquito,
                'mosquito density': mosquito,
                'population': population,
                'population density': population,
                'cases': cases,
                'dengue cases reported': cases,
                'latitude': coords['lat'],
                'longitude': coords['lon'],
                'month_num': month_num,
                'month': month_num
            }

            # Prepare base length
            target_len = None
            if SCALER is not None:
                target_len = getattr(SCALER, 'n_features_in_', None)
            # If we have FEATURE_NAMES, use its length as target
            if target_len is None and FEATURE_NAMES is not None:
                target_len = len(FEATURE_NAMES)
            if target_len is None:
                target_len = 12

            # Build array of zeros
            vec = [0.0] * target_len

            # If we have FEATURE_NAMES, fill by matching lowercased names
            if FEATURE_NAMES is not None:
                for i, fname in enumerate(FEATURE_NAMES):
                    lname = str(fname).strip().lower()
                    # try direct mapping keys
                    if lname in key_map:
                        vec[i] = float(key_map[lname])
                    else:
                        # try to match by substring
                        for k, v in key_map.items():
                            if k in lname or lname in k:
                                try:
                                    vec[i] = float(v)
                                except Exception:
                                    vec[i] = 0.0
                                break
                        # otherwise leave 0.0 (padding)
            else:
                # fallback: use default order (12)
                base = [year, week, temperature, rainfall, humidity, aqi, mosquito, population, cases, coords['lat'], coords['lon'], month_num]
                for i in range(min(len(base), target_len)):
                    vec[i] = float(base[i])

            return np.array([vec])

        features = build_feature_vector(data)

        # CRITICAL: Scale features using the same scaler from training
        if SCALER is None:
            return jsonify({'success': False, 'error': 'Server scaler is not loaded'}), 500

        try:
            features_scaled = SCALER.transform(features)
        except Exception as e:
            print('Scaler transform failed:', e)
            return jsonify({'success': False, 'error': f'Scaler transform failed: {str(e)}'}), 500

        # Make prediction with scaled features
        if not MODEL_REGISTRY:
            return jsonify({'success': False, 'error': 'No models loaded on server'}), 500

        predictions = {}
        for name, model in MODEL_REGISTRY.items():
            try:
                prediction = model.predict(features_scaled)[0]
                # Some models may not implement predict_proba; handle gracefully
                probabilities = None
                if hasattr(model, 'predict_proba'):
                    try:
                        probabilities = model.predict_proba(features_scaled)[0]
                    except Exception:
                        probabilities = None

                # Map risk and confidence
                risk_level = RISK_LABELS.get(int(prediction), 'Unknown')
                risk_code = int(prediction)
                confidence = None
                if probabilities is not None and len(probabilities) > risk_code:
                    confidence = float(probabilities[risk_code])

                predictions[name] = {
                    'risk_level': risk_level,
                    'risk_code': risk_code,
                    'confidence': round(confidence * 100, 2) if confidence is not None else None,
                    'color': RISK_COLORS.get(risk_code),
                    'icon': RISK_ICONS.get(risk_code),
                    'probabilities': {
                        'low': round(float(probabilities[0]) * 100, 2) if probabilities is not None and len(probabilities) > 0 else None,
                        'moderate': round(float(probabilities[1]) * 100, 2) if probabilities is not None and len(probabilities) > 1 else None,
                        'high': round(float(probabilities[2]) * 100, 2) if probabilities is not None and len(probabilities) > 2 else None
                    } if probabilities is not None else None
                }
            except Exception as e:
                print(f'Prediction failed for model {name}:', e)
                predictions[name] = {'error': f'Prediction failed: {str(e)}'}
        
        # Prepare response
        response = {
            'success': True,
            'predictions': predictions,
            'input_data': {
                'city': city,
                'month': month,
                'temperature': temperature,
                'rainfall': rainfall,
                'humidity': humidity,
                'aqi': aqi,
                'mosquito_density': mosquito,
                'population_density': population,
                'cases_reported': cases
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/model-info', methods=['GET'])
def model_info():
    """Return loaded models and optional comparison results"""
    loaded = list(MODEL_REGISTRY.keys())
    comparison = None
    comp_path = os.path.join(os.path.dirname(__file__), 'results', 'model_comparison_results.csv')
    try:
        if os.path.exists(comp_path):
            comparison = pd.read_csv(comp_path).to_dict(orient='records')
    except Exception:
        comparison = None

    return jsonify({
        'loaded_models': loaded,
        'comparison': comparison,
        'features': FEATURE_NAMES,
        'supported_cities': list(CITY_COORDS.keys()),
        'risk_levels': RISK_LABELS
    })


@app.route('/api/feature-importance', methods=['GET'])
def get_feature_importance():
    """Get actual feature importance from the trained model"""
    try:
        # Get model name from query parameter or default to xgboost_model
        model_name = request.args.get('model', 'xgboost_model')
        
        # Get the model from registry
        if model_name not in MODEL_REGISTRY:
            return jsonify({'error': f'Model {model_name} not found. Available models: {list(MODEL_REGISTRY.keys())}'}), 404
        
        model = MODEL_REGISTRY[model_name]
        
        # Check if model has feature_importances_ attribute (tree-based models)
        if not hasattr(model, 'feature_importances_'):
            return jsonify({'error': f'Model {model_name} does not support feature importance'}), 400
        
        # Get feature importances from the model
        importances = model.feature_importances_
        
        # Create feature importance pairs (handle length mismatch defensively)
        feature_importance = []
        for i, feature in enumerate(FEATURE_NAMES):
            importance_value = float(importances[i]) if i < len(importances) else 0.0
            feature_importance.append({
                'feature': feature,
                'importance': importance_value
            })
        
        # Sort by importance (descending)
        feature_importance.sort(key=lambda x: x['importance'], reverse=True)
        
        return jsonify({
            'success': True,
            'feature_importance': feature_importance
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/training-stats', methods=['GET'])
def get_training_stats():
    """Get statistics from training data"""
    try:
        # Load training data
        df = pd.read_csv('data/dengue_india_weekly_with_nulls.csv')
        
        # Create Month_Num if not exists
        month_map = {
            'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
            'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
        }
        if 'Month' in df.columns and 'Month_Num' not in df.columns:
            df['Month_Num'] = df['Month'].map(month_map)
        
        # Map risk levels to numeric if needed
        risk_map = {'Low': 0, 'Moderate': 1, 'High': 2}
        if 'Outbreak Risk Level' in df.columns:
            df['Risk_Level'] = df['Outbreak Risk Level'].map(risk_map)
        
        # Calculate monthly risk trends
        monthly_risk = {}
        if 'Risk_Level' in df.columns and 'Month_Num' in df.columns:
            monthly_risk = df.groupby('Month_Num')['Risk_Level'].apply(
                lambda x: (x == 2).sum() / len(x) if len(x) > 0 else 0
            ).to_dict()
        
        # Get rainfall vs temperature correlation data
        scatter_data = {
            'low': {'rainfall': [], 'temperature': []},
            'moderate': {'rainfall': [], 'temperature': []},
            'high': {'rainfall': [], 'temperature': []}
        }
        
        if 'Risk_Level' in df.columns:
            # Sample 50 points from each risk level to avoid sending too much data
            for risk_level in [0, 1, 2]:
                risk_df = df[df['Risk_Level'] == risk_level]
                if len(risk_df) > 50:
                    risk_df = risk_df.sample(50, random_state=42)
                
                risk_key = ['low', 'moderate', 'high'][risk_level]
                scatter_data[risk_key]['rainfall'] = risk_df['Average Rainfall (mm)'].dropna().tolist()
                scatter_data[risk_key]['temperature'] = risk_df['Average Temperature (°C)'].dropna().tolist()
        
        # Get city-wise risk distribution
        city_risk = {}
        if 'City' in df.columns and 'Risk_Level' in df.columns:
            cities = df['City'].unique()[:15]  # Top 15 cities
            for city in cities:
                city_df = df[df['City'] == city]
                if len(city_df) > 0:
                    high_risk_ratio = (city_df['Risk_Level'] == 2).sum() / len(city_df)
                    city_risk[city] = float(high_risk_ratio)
        
        return jsonify({
            'success': True,
            'monthly_risk': monthly_risk,
            'scatter_data': scatter_data,
            'city_risk': city_risk,
            'total_records': len(df)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("="*60)
    print("🦟 Dengue Risk Prediction API")
    print("="*60)
    print(f"Loaded models: {list(MODEL_REGISTRY.keys())}")
    print(f"Using scaler: {'yes' if SCALER is not None else 'no'}")
    print(f"Listening on port: {port}")
    print("="*60)
    app.run(debug=True, host='0.0.0.0', port=port)
