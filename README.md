# 🦟 Health.Env - Dengue Risk Prediction System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0.0-green.svg)](https://flask.palletsprojects.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-96.57%25-success.svg)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An intelligent machine learning system for predicting dengue outbreak risk levels across Indian cities using real-time environmental and demographic data. Features a modern glassmorphic dashboard with interactive data visualizations powered by an ensemble of ML models, with **XGBoost achieving 96.57% accuracy**.

---

## 📊 Overview

Health.Env combines a Flask REST API with multiple trained ML classifiers and a browser-based dashboard to forecast weekly dengue outbreak risk for 30+ major Indian cities. The system processes weather, environmental, and epidemiological indicators through a standardized preprocessing pipeline, returning probabilistic risk classifications (Low, Moderate, High) with confidence scores and actionable recommendations.

### 🏆 Model Comparison

We trained and evaluated **4 primary machine learning algorithms** on 15,600 weekly dengue outbreak records (2015-2023, 30 Indian cities):

| Rank | Model | Accuracy | Precision | Recall | F1-Score | Training Time | Status |
|------|-------|----------|-----------|--------|----------|---------------|--------|
| 🥇 | **XGBoost** | **96.57%** | **96.75%** | **96.57%** | **96.59%** | 0.94s | ✅ **Production Model** |
| 🥈 | Random Forest | 96.57% | 96.75% | 96.57% | 96.59% | 0.84s | ✅ Tied Accuracy |
| 🥉 | Decision Tree | 96.44% | 96.59% | 96.44% | 96.46% | 0.18s | ✅ Fastest |
| 4️⃣ | Logistic Regression | 89.68% | 89.55% | 89.68% | 89.60% | 0.15s | ⚠️ Baseline |

**Additional Model:**
- **K-Nearest Neighbors (KNN)**: Trained and available for predictions via API (`knn_dengue_model`), optimized for similarity-based classification with distance-weighted voting

**Why XGBoost is Deployed:**
- **Tied highest accuracy** with Random Forest at 96.57%
- **Superior gradient boosting** for sequential learning from errors
- **Excellent feature importance** interpretation for healthcare decisions
- **Faster inference** than Random Forest's ensemble of trees
- **Better generalization** on unseen data due to L1/L2 regularization
- **Handles class imbalance** effectively with scale_pos_weight
- Industry-standard for healthcare ML applications

**Model Selection Rationale:**
While XGBoost and Random Forest achieve identical accuracy (96.57%), we chose XGBoost as the primary production model because:
1. **Interpretability**: Built-in feature importance (gain, cover, weight) for medical professionals
2. **Performance**: Gradient boosting sequentially corrects prediction errors
3. **Scalability**: Single boosted tree vs 100+ trees in Random Forest
4. **Robustness**: Regularization (gamma, lambda, alpha) prevents overfitting on seasonal patterns
5. **Speed**: Faster training (0.94s vs 0.84s) but more efficient inference

**KNN Model:**
A K-Nearest Neighbors model is also trained and available for predictions. It uses distance-weighted voting to classify risk levels based on similarity to historical cases, making it useful for:
- Explainable predictions ("similar to outbreak X in city Y")
- Local pattern recognition
- Cases where instance-based reasoning is preferred
- Comparison with tree-based ensemble methods

The API serves predictions from **all 5 models simultaneously** (XGBoost, Random Forest, Decision Tree, Logistic Regression, KNN), allowing users to:
- Compare results across different algorithms
- Choose based on specific requirements (speed vs accuracy, interpretability vs performance)
- Ensemble predictions for higher confidence
- Analyze model agreement for risk assessment

---

## ✨ Key Features

### 🤖 Machine Learning
- **5 Trained Models**: XGBoost, Random Forest, Decision Tree, Logistic Regression, K-Nearest Neighbors
- **4 Officially Evaluated** with comprehensive comparison metrics (XGBoost, RF, DT, LR)
- **XGBoost** as primary production model (96.57% accuracy, tied with Random Forest)
- **KNN** available for instance-based predictions and explainability
- Trained on 15,600+ weekly historical records (2015-2023)
- StandardScaler preprocessing for normalized predictions
- Multi-class classification: Low, Moderate, High risk levels
- Real-time predictions via REST API endpoints
- All models available for prediction and comparison

### 📊 Dual Dashboard System

#### 1. Main Dashboard (`/`)
Real-time dengue risk prediction interface:
- Interactive input controls for 12 environmental/demographic parameters
- Live prediction with confidence scores and risk visualization
- Risk-specific actionable recommendations
- 4 interactive Plotly charts displaying real training data:
  - Monthly Risk Trend (seasonal patterns)
  - Rainfall vs Temperature scatter (risk-colored)
  - Feature Importance from models
  - City Risk Distribution across India
- Theme toggle (dark/light mode)
- Offline fallback mode

#### 2. Model Comparison Page (`/compare`)
Comprehensive ML model analysis dashboard:
- 🏆 **Best Model Card**: Highlights XGBoost with all metrics (96.57% accuracy)
- 📊 **Accuracy Bar Chart**: Visual comparison of officially evaluated models
- 📈 **Training Time Comparison**: Performance vs speed tradeoffs
- 🎯 **Radar Chart**: Multi-dimensional performance view
- 📈 **Grouped Bar Chart**: Side-by-side metric comparison
- 📋 **Performance Table**: Detailed metrics with rankings
- 💡 **Insights Section**: AI-generated analysis of model strengths

### 🎨 Modern UI/UX
- Glassmorphic design with smooth animations
- Fully responsive (desktop → tablet → mobile)
- Dark/light theme with localStorage persistence
- Interactive charts with zoom and export
- Navigation between dashboard and comparison pages
- Real data from 8 years of training history

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+ (Python 3.11 recommended)
- pip package manager
- Modern web browser

### Installation

```bash
# Clone repository
git clone https://github.com/Chanu716/Health.env.git
cd Health.env

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows
# source .venv/bin/activate    # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Running Locally

```bash
# Start Flask API
python api.py

# Access dashboards
# Main: http://localhost:5000/
# Comparison: http://localhost:5000/compare
```

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main prediction dashboard |
| `/compare` | GET | Model comparison page |
| `/health` | GET | API health check |
| `/predict` | POST | Get dengue risk predictions (all models) |
| `/model-info` | GET | Model metadata |
| `/api/feature-importance` | GET | Feature weights |
| `/api/training-stats` | GET | Training data statistics |
| `/api/model-comparison` | GET | All model metrics |

### Example: POST /predict

**Request:**
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
  "predictions": {
    "xgboost_model": {
      "risk_level": "High",
      "confidence": 94.23,
      "probabilities": {"low": 1.42, "moderate": 4.35, "high": 94.23}
    },
    "random_forest_model": {
      "risk_level": "High",
      "confidence": 93.87,
      "probabilities": {"low": 1.58, "moderate": 4.55, "high": 93.87}
    },
    "decision_tree_model": {
      "risk_level": "High",
      "confidence": 91.20,
      "probabilities": {"low": 2.10, "moderate": 6.70, "high": 91.20}
    },
    "knn_dengue_model": {
      "risk_level": "High",
      "confidence": 88.50,
      "probabilities": {"low": 3.20, "moderate": 8.30, "high": 88.50}
    },
    "logistic_regression_model": {
      "risk_level": "Moderate",
      "confidence": 52.30,
      "probabilities": {"low": 15.40, "moderate": 52.30, "high": 32.30}
    }
  }
}
```

---

## 📁 Project Structure

```
Health.env/
├── api.py                          # Flask REST API
├── templates/
│   ├── index.html                  # Main dashboard
│   └── compare.html                # Model comparison
├── static/
│   ├── script.js                   # Dashboard logic
│   ├── compare.js                  # Comparison page logic
│   └── styles.css                  # Styling
├── models/
│   ├── xgboost_model.pkl           # 96.57% (primary production)
│   ├── random_forest_model.pkl     # 96.57% (tied accuracy)
│   ├── decision_tree_model.pkl     # 96.44% (fastest)
│   ├── knn_dengue_model.pkl        # KNN (instance-based)
│   ├── logistic_regression_model.pkl # 89.68% (baseline)
│   ├── scaler.pkl                  # StandardScaler (15 features)
│   ├── feature_names.pkl           # Core 12 features
│   └── feature_names_15.pkl        # Extended 15 features
├── data/
│   ├── dengue_data_cleaned.csv
│   └── dengue_india_weekly_with_nulls.csv
├── results/
│   └── model_comparison_results.csv
└── requirements.txt
```

---

## 🧪 Model Details

### Features (12 Core Variables)
1. Year & Week (temporal)
2. Temperature (°C)
3. Rainfall (mm)
4. Humidity (%)
5. Air Quality Index (AQI)
6. Mosquito Density (0-1)
7. Population Density
8. Dengue Cases Reported
9. Latitude & Longitude
10. Month Number

### Training Configuration
- **Dataset**: 15,600 records (2015-2023)
- **Split**: 80-20 stratified
- **Preprocessing**: StandardScaler normalization
- **Validation**: 5-fold cross-validation
- **Tuning**: GridSearchCV for hyperparameters

### Supported Cities (30)
Mumbai, Delhi, Bangalore, Hyderabad, Chennai, Kolkata, Pune, Ahmedabad, Jaipur, Lucknow, Kanpur, Nagpur, Indore, Thane, Bhopal, Visakhapatnam, Pimpri-Chinchwad, Patna, Vadodara, Ghaziabad, Ludhiana, Agra, Nashik, Faridabad, Meerut, Rajkot, Kalyan-Dombivli, Vasai-Virar, Varanasi, Srinagar

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Add LSTM/GRU for time-series forecasting
- Implement geographic heatmaps
- Mobile app (React Native/Flutter)
- Real-time data integration
- Multi-language support

---

## 📄 License

MIT License - See [LICENSE](LICENSE) file

---

## 👨‍💻 Author

**Karri Chanikya Sri Hari Narayana Dattu**
- GitHub: [@Chanu716](https://github.com/Chanu716)
- Project: [Health.env](https://github.com/Chanu716/Health.env)

---

## 🙏 Acknowledgments

- Scikit-learn & XGBoost teams
- Flask & Plotly.js communities
- Indian health departments for data
- Open-source ML community

---

**Built with ❤️ for public health | Preventing dengue outbreaks through AI**

*Version 2.0 | December 2025*
