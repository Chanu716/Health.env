# 🦟 Health.Env - Dengue Risk Prediction System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0.0-green.svg)](https://flask.palletsprojects.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-ML-orange.svg)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An intelligent machine learning system for predicting dengue outbreak risk levels across Indian cities using real-time environmental and demographic data. Features a modern glassmorphic dashboard with interactive data visualizations powered by a trained XGBoost model.

---

## 📊 Overview

Health.Env combines a Flask REST API, a trained XGBoost classifier (96.57% accuracy), and a browser-based dashboard to forecast weekly dengue outbreak risk for 30+ major Indian cities. The system processes weather, environmental, and epidemiological indicators through the same preprocessing pipeline used during training, returning probabilistic risk classifications (Low, Moderate, High) with confidence scores and actionable recommendations.

### 🏆 Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 96.57% |
| **Precision** | 96.75% |
| **Recall** | 96.57% |
| **F1 Score** | 96.59% |
| **Training Data** | 15,600 weekly records |
| **Time Period** | 2015-2023 (8 years) |
| **Cities Covered** | 30 Indian cities |

---

## ✨ Key Features

### 🤖 Machine Learning
- XGBoost Classifier with 96.57% accuracy
- Trained on 15,600+ historical dengue outbreak records
- StandardScaler preprocessing for normalized predictions
- Multi-class classification: Low, Moderate, High risk levels
- Real-time predictions via REST API endpoints

### 📊 Real Data Visualization
- **Monthly Risk Trends** - Seasonal patterns from training data
- **Temperature vs Rainfall** - Scatter plots by risk level
- **Feature Importance** - Live XGBoost model weights
- **City Risk Distribution** - City-wise statistics from 30 cities
- 5-minute intelligent data caching for optimal performance

### 🎨 Modern Dashboard
- Glassmorphic UI with dark/light theme toggle
- Responsive design (desktop → tablet → mobile)
- Interactive Plotly.js charts with zoom functionality
- Real-time predictions with confidence indicators
- Smart recommendations based on risk classification
- Offline fallback mode when API is unavailable

---

## 📁 Repository Structure

```
.
├── api.py                  # Flask REST API with prediction endpoints
├── requirements.txt        # Python dependencies
├── index.html             # Dashboard HTML structure
├── script.js              # Frontend logic and chart initialization
├── styles.css             # Glassmorphic styling
├── data/
│   ├── dengue_data_cleaned.csv               # Preprocessed data
│   ├── model_comparison_results.csv          # Model evaluation metrics
│   └── tuning_comparison.csv                 # Hyperparameter tuning results
├── models/
│   ├── xgboost_model.pkl                     # Trained XGBoost classifier
│   ├── knn_dengue_model.pkl                  
│   ├── scaler.pkl                            # StandardScaler for normalization
│   ├── feature_names.pkl                     # Feature order reference
│   └── [alternative models]
├── scripts/
│   └── compare_models.py                   # Other trained classifiers
├── notebooks/              # Jupyter notebooks for training and analysis
├── results/   
│   └── model_comparison_results.csv            # Model performance plots and metrics
├── docs/                  # Additional documentation
└── LICENSE               # MIT License
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Modern web browser (Chrome, Firefox, Edge, Safari)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Chanu716/Health.env.git
   cd Health.env
   ```

2. **Create virtual environment (recommended)**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**
   ```powershell
   pip install -r requirements.txt
   ```

### Running the Application

1. **Start the Flask API**
   ```powershell
   python api.py
   ```
   The API will start on `http://localhost:5000`

2. **Open the Dashboard**
   - Simply double-click `index.html` or
   - Drag `index.html` into your browser

3. **Make a Prediction**
   - Select a city and month
   - Adjust environmental parameters using sliders
   - Click **"Predict Risk Level"**
   - View results, confidence scores, and recommendations

---

## 🔌 API Reference

### Base URL
```
http://localhost:5000
```

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check and model status |
| `/predict` | POST | Get dengue risk prediction |
| `/model-info` | GET | Model metadata and performance metrics |
| `/api/feature-importance` | GET | XGBoost feature importance scores |
| `/api/training-stats` | GET | Training dataset statistics |

### Example: POST /predict

**Request Body:**
```json
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
```

**Response:**
```json
{
  "success": true,
  "prediction": {
    "risk_level": "High",
    "risk_code": 2,
    "confidence": 89.45,
    "color": "#ff3366",
    "icon": "🔴"
  },
  "probabilities": {
    "low": 2.31,
    "moderate": 8.24,
    "high": 89.45
  },
  "input_data": { ... }
}
```

---

## 🧪 Model Details

### Features Used (12 total)
1. Year
2. Week
3. Temperature (°C)
4. Rainfall (mm)
5. Humidity (%)
6. Air Quality Index (AQI)
7. Mosquito Density
8. Population Density
9. Dengue Cases Reported
10. Latitude
11. Longitude
12. Month Number

### Training Configuration
- **Algorithm**: XGBoost Classifier
- **Preprocessing**: StandardScaler normalization
- **Train-Test Split**: 80-20
- **Validation**: Cross-validation with stratified sampling
- **Hyperparameter Tuning**: Grid search (see `docs/HYPERPARAMETER_TUNING_GUIDE.md`)

### Supported Cities (30)
Mumbai, Delhi, Bangalore, Hyderabad, Chennai, Kolkata, Pune, Ahmedabad, Jaipur, Lucknow, Kanpur, Nagpur, Indore, Thane, Bhopal, Visakhapatnam, Pimpri-Chinchwad, Patna, Vadodara, Ghaziabad, Ludhiana, Agra, Nashik, Faridabad, Meerut, Rajkot, Kalyan-Dombivli, Vasai-Virar, Varanasi, Srinagar

---

## 🎨 Dashboard Features

### Input Controls
- **City Selector**: Dropdown with 30 major Indian cities
- **Month Selector**: All 12 months for seasonal analysis
- **Weather Sliders**: Temperature (20-45°C), Rainfall (0-3000mm), Humidity (40-100%)
- **Environmental Sliders**: AQI (0-500), Mosquito Density (0-1)
- **Demographics**: Population density and recent case numbers

### Visualizations
1. **Monthly Risk Trend** - Line chart showing seasonal dengue patterns
2. **Rainfall vs Temperature** - Scatter plot colored by risk level
3. **Feature Importance** - Horizontal bar chart from XGBoost model
4. **City Risk Distribution** - Bar chart with gradient coloring

### Theme Support
- **Dark Mode** (default): High contrast with glassmorphic elements
- **Light Mode**: Clean, minimal design for daytime use
- Theme preference saved in browser localStorage

---

## 🛠️ Development

### Project Structure
- **Backend** (`api.py`): Flask REST API serving predictions and model data
- **Frontend** (`index.html`, `script.js`, `styles.css`): Static dashboard with no build process
- **Models** (`models/`): Serialized ML artifacts
- **Data** (`data/`): Training datasets and evaluation results

### Customization

#### Change API Port
Edit `api.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5000)  # Change port here
```

Update `script.js`:
```javascript
fetch('http://localhost:5000/predict', { ... })  // Update URL
```

#### Retrain Model
1. Run training notebooks in `notebooks/`
2. Save new models to `models/` directory
3. Update performance metrics in `api.py` `/model-info` endpoint

#### Add New Cities
1. Add coordinates to `CITY_COORDS` dict in `api.py`
2. Add city option to `<select id="city">` in `index.html`
3. Retrain model with new city data

---

## 📊 Data Sources

- **Historical Dengue Data**: Weekly dengue case reports from 2015-2023
- **Weather Data**: Temperature, rainfall, humidity from meteorological departments
- **Environmental Data**: Air Quality Index (AQI) from pollution monitoring stations
- **Demographic Data**: Population density from census records
- **Entomological Data**: Mosquito density surveys from health departments

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Karri Chanikya Sri Hari Narayana Dattu**

- GitHub: [@Chanu716](https://github.com/Chanu716)
- Project: [Health.env](https://github.com/Chanu716/Health.env)

---

## 📚 Additional Resources

- [Hyperparameter Tuning Guide](docs/HYPERPARAMETER_TUNING_GUIDE.md)
- [Dashboard Documentation](docs/README_Dashboard.md)
- Model comparison results in `results/` directory
- Training notebooks in `notebooks/` directory

---

## 🙏 Acknowledgments

- XGBoost development team
- Flask and Plotly.js communities
- Indian meteorological and health departments for data access
- Open-source machine learning community

---

**Built with ❤️ for public health | Preventing dengue outbreaks through AI**
