# 🦟 Health.Env - Dengue Risk Prediction System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0.0-green.svg)](https://flask.palletsprojects.com/)
[![RandomForest](https://img.shields.io/badge/Random_Forest-99.11%25-success.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An intelligent machine learning system for predicting dengue outbreak risk levels across Indian cities using real-time environmental and demographic data. Features a modern glassmorphic dashboard with interactive data visualizations powered by an ensemble of ML models, with **Random Forest achieving 99.11% accuracy**.

---

## 📊 Overview

Health.Env combines a Flask REST API with multiple trained ML classifiers and a browser-based dashboard to forecast weekly dengue outbreak risk for 30+ major Indian cities. The system processes weather, environmental, and epidemiological indicators through a standardized preprocessing pipeline, returning probabilistic risk classifications (Low, Moderate, High) with confidence scores and actionable recommendations.

### 🏆 Model Comparison

We trained and evaluated **5 different machine learning algorithms** on 15,600 weekly dengue outbreak records:

| Rank | Model | Accuracy | Precision | Recall | F1-Score | Status |
|------|-------|----------|-----------|--------|----------|--------|
| 🥇 | **Random Forest** | **99.11%** | **99.12%** | **99.11%** | **99.11%** | ✅ **Best Model** |
| 🥈 | XGBoost | 97.07% | 97.19% | 97.07% | 97.08% | ✅ Production Ready |
| 🥉 | Decision Tree | 96.88% | 96.99% | 96.88% | 96.89% | ✅ Production Ready |
| 4️⃣ | K-Nearest Neighbors (KNN) | 95.32% | 95.46% | 95.32% | 95.38% | ✅ Good Performance |
| 5️⃣ | Logistic Regression | 89.98% | 89.83% | 89.98% | 89.88% | ⚠️ Baseline Model |

**Why Random Forest Wins:**
- Highest accuracy across all metrics (99.11%)
- Excellent precision and recall balance
- Robust to overfitting through ensemble learning
- Handles non-linear relationships effectively
- Low variance due to bagging technique

**Model Selection Philosophy:**
While Random Forest achieves the highest accuracy, we maintain all models for different use cases:
- **Random Forest**: Production default (best overall performance, robust predictions)
- **XGBoost**: Alternative with excellent feature importance interpretation
- **Decision Tree**: Fast inference for resource-constrained deployments
- **KNN**: Good for similarity-based predictions and explainability
- **Logistic Regression**: Baseline for comparison and simple linear relationships

The API serves predictions from all models simultaneously, allowing users to compare results or choose based on specific requirements (speed vs accuracy, interpretability vs performance, etc.).

---

## ✨ Key Features

### 🤖 Machine Learning
- **5 Trained Models** with comprehensive comparison metrics
- **Random Forest** as primary model (99.11% accuracy)
- Trained on 15,600+ historical dengue outbreak records
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
- 🏆 **Best Model Card**: Highlights Random Forest with all metrics
- 📊 **Accuracy Bar Chart**: Visual comparison of all 5 models
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
    "random_forest_model": {
      "risk_level": "High",
      "confidence": 94.23,
      "probabilities": {"low": 1.42, "moderate": 4.35, "high": 94.23}
    },
    "xgboost_model": { ... },
    "decision_tree_model": { ... },
    "knn_dengue_model": { ... },
    "logistic_regression_model": { ... }
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
│   ├── random_forest_model.pkl     # 99.11% (best)
│   ├── xgboost_model.pkl           # 97.07%
│   ├── decision_tree_model.pkl     # 96.88%
│   ├── knn_dengue_model.pkl        # 95.32%
│   ├── logistic_regression_model.pkl # 89.98%
│   └── scaler.pkl                  # Feature normalizer
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
