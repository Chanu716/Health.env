"""
Dengue Risk Prediction API
Serves the trained XGBoost model for real-time predictions
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# Load the best trained model (XGBoost) and scaler
MODEL_PATH = 'models/xgboost_model.pkl'
SCALER_PATH = 'models/scaler.pkl'

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print(f"✅ Model loaded successfully from {MODEL_PATH}")
    
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    print(f"✅ Scaler loaded successfully from {SCALER_PATH}")
except Exception as e:
    print(f"❌ Error loading model/scaler: {e}")
    model = None
    scaler = None

# Feature names (must match training data order)
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
        'model_loaded': model is not None,
        'model_type': 'XGBoost' if model else None
    })


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
        if model is None or scaler is None:
            return jsonify({'error': 'Model or scaler not loaded'}), 500
        
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
        
        # Create feature array in correct order
        features = np.array([[
            year,           # Year
            week,           # Week
            temperature,    # Temperature
            rainfall,       # Rainfall
            humidity,       # Humidity
            aqi,           # AQI
            mosquito,      # Mosquito Density
            population,    # Population Density
            cases,         # Dengue Cases
            coords['lat'], # Latitude
            coords['lon'], # Longitude
            month_num      # Month_Num
        ]])
        
        # CRITICAL: Scale features using the same scaler from training
        features_scaled = scaler.transform(features)
        
        # Make prediction with scaled features
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        # Get risk details
        risk_level = RISK_LABELS[prediction]
        risk_color = RISK_COLORS[prediction]
        risk_icon = RISK_ICONS[prediction]
        confidence = float(probabilities[prediction])
        
        # Prepare response
        response = {
            'success': True,
            'prediction': {
                'risk_level': risk_level,
                'risk_code': int(prediction),
                'confidence': round(confidence * 100, 2),
                'color': risk_color,
                'icon': risk_icon
            },
            'probabilities': {
                'low': round(float(probabilities[0]) * 100, 2),
                'moderate': round(float(probabilities[1]) * 100, 2),
                'high': round(float(probabilities[2]) * 100, 2)
            },
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
    """Get model information"""
    return jsonify({
        'model_type': 'XGBoost Classifier',
        'accuracy': 0.9657,
        'precision': 0.9675,
        'recall': 0.9657,
        'f1_score': 0.9659,
        'features': FEATURE_NAMES,
        'supported_cities': list(CITY_COORDS.keys()),
        'risk_levels': RISK_LABELS
    })


@app.route('/api/feature-importance', methods=['GET'])
def get_feature_importance():
    """Get actual feature importance from the trained model"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get feature importances from XGBoost model
        importances = model.feature_importances_
        
        # Create feature importance pairs
        feature_importance = []
        for i, feature in enumerate(FEATURE_NAMES):
            feature_importance.append({
                'feature': feature,
                'importance': float(importances[i])
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
    print("="*60)
    print("🦟 Dengue Risk Prediction API")
    print("="*60)
    print(f"Model: XGBoost Classifier")
    print(f"Accuracy: 96.57%")
    print(f"Endpoint: http://localhost:5000/predict")
    print("="*60)
    app.run(debug=True, host='0.0.0.0', port=5000)
