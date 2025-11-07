# 🦟 Dengue Risk Prediction System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0.0-green.svg)](https://flask.palletsprojects.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-ML-orange.svg)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An intelligent machine learning system for predicting dengue outbreak risk levels across Indian cities using real-time environmental and demographic data. Features a modern glassmorphic dashboard with interactive visualizations.

---

## 📊 Project Overview

This system uses an **XGBoost** machine learning model with **96.57% accuracy** to predict dengue outbreak risk levels based on weather, environmental, and epidemiological factors.

### 🏆 Model Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 96.57% |
| **Precision** | 96.75% |
| **Recall** | 96.57% |
| **F1 Score** | 96.59% |
| **Training Data** | 15,600 records |
| **Time Period** | 2015-2023 (8 years) |
| **Cities Covered** | 30 Indian cities |

---

## ✨ Key Features

### 🤖 Machine Learning Model
- **XGBoost Classifier** with 96.57% accuracy
- Trained on 15,600+ historical dengue outbreak records
- Proper StandardScaler preprocessing
- Multi-class classification: Low, Moderate, High risk
- Real-time predictions via REST API

### 📊 Real Data Visualization
- **Monthly Risk Trends** - Actual seasonal outbreak patterns from training data
- **Temperature vs Rainfall Correlation** - Real scatter plots by risk level
- **Feature Importance** - Live XGBoost model feature weights
- **City Risk Distribution** - Authentic city-wise statistics from 30 Indian cities
- All charts load real data with 5-minute caching for performance

### 🎨 Modern Dashboard
- **Glassmorphic UI** with dark/light theme toggle
- **Responsive 3-column layout** (sidebar | results | charts)
- **Interactive Plotly.js charts** with zoom functionality
- **Real-time predictions** with confidence scores
- **Personalized recommendations** based on risk level

### ⚡ Performance Optimized
- 5-minute data caching to reduce API calls
- Lazy loading of chart data
- Fallback static data for offline mode
- Minimal bundle size with CDN resources

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/Chanu716/Health.env.git
cd Health.env
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the API server**
```bash
python api.py
```
The API will start on `http://localhost:5000`

**4. Open the dashboard** (in a new terminal)
```bash
python -m http.server 8000
```

**5. Access the application**
- 🌐 Dashboard: http://localhost:8000/index.html
- ❤️ API Health: http://localhost:5000/health
- 📊 API Info: http://localhost:5000/model-info

---

## 📁 Project Structure

```
Health.env/
├── api.py                          # Flask REST API with real data endpoints
├── index.html                      # Modern glassmorphic dashboard
├── script.js                       # Frontend logic with real data loading
├── styles.css                      # Responsive CSS styling
├── requirements.txt                # Python dependencies
│
├── models/                         # Trained ML models
│   ├── xgboost_model.pkl          # Best model (96.57% accuracy)
│   ├── scaler.pkl                 # StandardScaler for preprocessing
│   ├── feature_names.pkl          # Feature order preservation
│   ├── random_forest_model.pkl    # Alternative model
│   ├── decision_tree_model.pkl    # Baseline model
│   └── logistic_regression_model.pkl
│
├── data/                           # Datasets
│   ├── dengue_data_cleaned.csv    # Cleaned training data
│   └── dengue_india_weekly_with_nulls.csv  # Raw data
│
├── notebooks/                      # Jupyter notebooks
│   ├── data_preprocessing.ipynb   # Data cleaning pipeline
│   ├── exploratory_data_analysis.ipynb  # EDA visualizations
│   └── model_training_evaluation.ipynb  # Model training & tuning
│
├── results/                        # Training results
│   ├── *.png                       # Performance visualizations
│   ├── hyperparameter_tuning_results.json
│   └── model_comparison_results.csv
│
└── docs/                           # Documentation
    ├── README_Dashboard.md
    └── HYPERPARAMETER_TUNING_GUIDE.md
```

---

## 🔧 API Endpoints

### 1. Health Check
```http
GET /health
```
Returns API status and uptime.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### 2. Predict Risk
```http
POST /predict
Content-Type: application/json
```

**Request Body:**
```json
{
  "city": "Mumbai",
  "month": "July",
  "temperature": 28.5,
  "rainfall": 150.0,
  "humidity": 85.0,
  "aqi": 120,
  "mosquito": 0.8,
  "population": 50000,
  "cases": 45
}
```

**Response:**
```json
{
  "success": true,
  "prediction": {
    "risk_level": "High",
    "risk_code": 2,
    "confidence": 87.5,
    "color": "#ff3366",
    "icon": "🔴"
  },
  "probabilities": {
    "low": 5.2,
    "moderate": 7.3,
    "high": 87.5
  }
}
```

### 3. Feature Importance
```http
GET /api/feature-importance
```
Returns real feature importance values from the trained XGBoost model.

### 4. Training Statistics
```http
GET /api/training-stats
```
Returns aggregated statistics from training data:
- Monthly risk trends (12 months)
- Temperature vs Rainfall correlation data
- City-wise risk distribution

### 5. Model Information
```http
GET /model-info
```
Returns model metadata and performance metrics.

---

## 🎯 How to Use

### Dashboard Input Fields

1. **City** - Select from 30 Indian cities
2. **Month** - Choose the month for prediction
3. **Temperature** - Average temperature (20-45°C)
4. **Rainfall** - Monthly rainfall (0-3000mm)
5. **Humidity** - Relative humidity (40-100%)
6. **AQI** - Air Quality Index (0-500)
7. **Mosquito Density** - Breeding site density (0-1)
8. **Population Density** - People per sq km
9. **Cases Reported** - Previous dengue cases

### Making Predictions

1. Fill in all input fields
2. Click "Predict Risk" button
3. View risk level:
   - 🟢 **LOW RISK** (Blue) - Routine monitoring
   - 🟡 **MODERATE RISK** (Green) - Enhanced surveillance
   - 🔴 **HIGH RISK** (Red) - Immediate action required

---

## 📊 Features

### Input Features (12 total)
- Year
- Week
- Temperature (°C)
- Rainfall (mm)
- Humidity (%)
- Air Quality Index (AQI)
- Mosquito Density
- Population Density
- Dengue Cases Reported
- Latitude
- Longitude
- Month Number

### Supported Cities (30)
Mumbai, Delhi, Bangalore, Hyderabad, Chennai, Kolkata, Pune, Ahmedabad, Jaipur, Lucknow, Kanpur, Nagpur, Indore, Thane, Bhopal, Visakhapatnam, Pimpri-Chinchwad, Patna, Vadodara, Ghaziabad, Ludhiana, Agra, Nashik, Faridabad, Meerut, Rajkot, Kalyan-Dombivli, Vasai-Virar, Varanasi, Srinagar

---

## 🔬 Data Science Workflow

### 1. Data Preprocessing
- Handled missing values and outliers
- Feature engineering (Month_Num, seasonal indicators)
- Label encoding for risk levels (Low=0, Moderate=1, High=2)
- StandardScaler normalization

### 2. Model Training
- Tested 4 algorithms: Logistic Regression, Decision Tree, Random Forest, XGBoost
- XGBoost selected for best performance
- Hyperparameter tuning with GridSearchCV
- 80-20 train-test split with stratification

### 3. Model Validation
- 5-fold cross-validation
- Confusion matrix analysis
- Feature importance visualization
- Real-world test scenarios validated

### 4. Hyperparameter Tuning
- **Method:** GridSearchCV with 5-fold cross-validation
- **Combinations tested:** 3,840 parameter combinations
- **Best XGBoost parameters:**
  - n_estimators: 200
  - max_depth: 7
  - learning_rate: 0.1
  - subsample: 0.9
  - colsample_bytree: 0.9

### Feature Importance (from trained model)
1. **Dengue Cases Reported** - 40.01%
2. **Mosquito Density** - 32.16%
3. **Rainfall** - 9.72%
4. **Week** - 8.24%
5. **Temperature** - 2.18%
6. **Population Density** - 2.04%
7. Other features - 5.65%

---

## 🛠️ Technology Stack

### Backend
- **Flask 3.0.0** - REST API framework
- **XGBoost 2.0.3** - Gradient boosting classifier
- **Scikit-learn 1.3.2** - ML utilities and preprocessing
- **Pandas 2.1.4** - Data manipulation
- **NumPy 1.26.2** - Numerical computing
- **Flask-CORS 4.0.1** - Cross-origin requests

### Frontend
- **Vanilla JavaScript (ES6)** - Frontend logic
- **Plotly.js 2.27.0** - Interactive data visualization
- **CSS Grid & Flexbox** - Responsive layout
- **Glassmorphism** - Modern UI design

### Fonts
- **Barlow** - Primary UI font
- **Comfortaa** - Display headings
- **JetBrains Mono** - Monospace numbers

---

## 🐛 Troubleshooting

### API Not Starting
```bash
# Check if port 5000 is available
netstat -ano | findstr :5000

# Try different port (edit api.py to change port)
python api.py
```

### Model Not Loading
```bash
# Verify model file exists
dir models\xgboost_model.pkl

# Check file size (should be ~500KB)
```

### CORS Errors
- Ensure Flask-CORS is installed
- Check browser console for details
- Use a local web server instead of file://

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 for Python code
- Use ESLint for JavaScript
- Add comments for complex logic
- Update documentation for new features
- Test thoroughly before submitting PR

---

## 📝 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Dataset**: Based on Indian dengue outbreak historical records (2015-2023)
- **XGBoost**: Excellent gradient boosting library
- **Plotly.js**: Beautiful interactive visualization library
- **Flask**: Lightweight and powerful web framework
- **Indian Health Departments**: For providing dengue outbreak data

---

## 📧 Contact & Support

- **GitHub**: [@Chanu716](https://github.com/Chanu716)
- **Repository**: [Health.env](https://github.com/Chanu716/Health.env)
- **Issues**: [Report a bug](https://github.com/Chanu716/Health.env/issues)

For questions, feedback, or collaboration opportunities, please open an issue on GitHub.

---

## 🎓 Future Enhancements

- [ ] Deploy on cloud (AWS/Azure/Heroku)
- [ ] Add batch prediction endpoint
- [ ] Implement user authentication
- [ ] Mobile app version
- [ ] Real-time weather API integration
- [ ] Historical prediction tracking
- [ ] Email alerts for high-risk predictions
- [ ] Model retraining pipeline
- [ ] Multi-language support

---

<div align="center">

**Made with ❤️ for better dengue outbreak prevention**

⭐ Star this repo if you found it helpful!

</div>w": 3.22,├── models/                     # Trained models3. **Temperature** - Average temperature (20-45°C)

    "moderate": 9.33,

    "high": 87.45│   ├── xgboost_model.pkl      # Main XGBoost classifier (96.57% accuracy)4. **Rainfall** - Monthly rainfall (0-3000mm)

  },

  "input_data": { ... }│   ├── scaler.pkl             # StandardScaler for preprocessing5. **Humidity** - Relative humidity (40-100%)

}

```│   ├── feature_names.pkl      # Feature order preservation6. **AQI** - Air Quality Index (0-500)



### 3. Feature Importance (Real Model Data)│   └── ...                    # Other model files7. **Mosquito Density** - Breeding site density (0-1)

```http

GET /api/feature-importance│8. **Population Density** - People per sq km

```

Returns actual feature importance from the trained XGBoost model.├── data/                       # Training datasets9. **Cases Reported** - Previous dengue cases



**Response:**│   ├── dengue_data_cleaned.csv               # Cleaned training data

```json

{│   └── dengue_india_weekly_with_nulls.csv    # Original dataset### Making Predictions

  "success": true,

  "feature_importance": [│

    {"feature": "Dengue Cases Reported", "importance": 0.40009},

    {"feature": "Mosquito Density", "importance": 0.32160},├── notebooks/                  # Jupyter notebooks1. Fill in all input fields

    {"feature": "Rainfall (mm)", "importance": 0.09720},

    ...│   ├── data_preprocessing.ipynb              # Data cleaning pipeline2. Click "Predict Risk" button

  ]

}│   ├── exploratory_data_analysis.ipynb       # EDA visualizations3. View risk level:

```

│   └── model_training_evaluation.ipynb       # Model training & tuning   - 🔴 **HIGH RISK** (Red)

### 4. Training Statistics (Real Data)

```http│   - 🟡 **MODERATE RISK** (Yellow)

GET /api/training-stats

```├── docs/                       # Documentation   - 🟢 **LOW RISK** (Green)

Returns aggregated statistics from actual training data.

│   ├── HYPERPARAMETER_TUNING_GUIDE.md

**Response:**

```json│   └── README_Dashboard.md## 🔧 API Usage

{

  "success": true,│

  "monthly_risk": {

    "1.0": 0.0, "2.0": 0.0, ..., "6.0": 0.7855, "7.0": 0.7877, ...└── results/                    # Model outputs### Endpoint: `/predict`

  },

  "scatter_data": {    ├── model_comparison_results.csv

    "low": {"rainfall": [...], "temperature": [...]},

    "moderate": {"rainfall": [...], "temperature": [...]},    └── tuning_comparison.csv**Method:** POST

    "high": {"rainfall": [...], "temperature": [...]}

  },```

  "city_risk": {

    "Kolkata": 0.4406, "Chennai": 0.3959, "Mumbai": 0.3612, ...**Request Body:**

  },

  "total_records": 15600## 🔧 API Endpoints```json

}

```{



### 5. Model Information### 1. **Health Check**  "city": "Mumbai",

```http

GET /model-info```http  "month": "July",

```

Returns model metadata and performance metrics.GET /health  "temperature": 32.5,



---```  "rainfall": 1200,



## 🎯 Model PerformanceReturns API status and uptime.  "humidity": 85,



| Metric | Score |  "aqi": 150,

|--------|-------|

| **Accuracy** | 96.57% |### 2. **Predict Risk**  "mosquito": 0.75,

| **Precision** | 96.75% |

| **Recall** | 96.57% |```http  "population": 5000,

| **F1 Score** | 96.59% |

| **Training Data** | 15,600 records |POST /predict  "cases": 450

| **Cities Covered** | 15 Indian cities |

| **Time Period** | 2015-2023 (8 years) |Content-Type: application/json}



### Feature Importance (Actual Model Values)```

From the trained XGBoost model's `feature_importances_` attribute:

{

1. **Dengue Cases Reported** - 40.01%

2. **Mosquito Density** - 32.16%  "city": "Mumbai",**Response:**

3. **Rainfall (mm)** - 9.72%

4. **Week** - 8.24%  "month": "July",```json

5. **Temperature (°C)** - 2.18%

6. **Population Density** - 2.04%  "temperature": 28.5,{

7. **Month_Num** - 1.21%

8. **Humidity (%)** - 1.08%  "rainfall": 150.0,  "success": true,

9. **Longitude** - 0.94%

10. **AQI** - 0.84%  "humidity": 85.0,  "prediction": {

11. **Latitude** - 0.81%

12. **Year** - 0.78%  "aqi": 120,    "risk_level": "High",



---  "mosquito": 0.8,    "risk_code": 2,



## 🎨 Dashboard Features  "population": 50000,    "confidence": 87.5,



### Theme Toggle  "cases": 45    "color": "#ff3366",

- **Dark Theme**: Glassmorphic design with blur effects (default)

- **Light Theme**: Neumorphic design with soft shadows}    "icon": "🔴"

- Persistent theme saved to localStorage

```  },

### Responsive Layout

- **Desktop (>1400px)**: 3-column grid (sidebar | results | charts)  "probabilities": {

- **Tablet (768px-1400px)**: 2-column layout

- **Mobile (<768px)**: Single column stack**Response:**    "low": 5.2,



### Chart Interactions```json    "moderate": 7.3,

- Click any chart to open full-screen modal with zoom

- Real-time data loading from API endpoints{    "high": 87.5

- Smooth animations and theme transitions

- Automatic color scheme adaptation  "success": true,  }



### Input Controls  "prediction": {}

- City dropdown (15 Indian cities)

- Month selector (seasonal awareness)    "risk_level": "High",```

- Temperature slider (10-45°C)

- Rainfall slider (0-300mm)    "risk_code": 2,

- Humidity slider (30-100%)

- AQI slider (0-500)    "confidence": 87.45,### Other Endpoints

- Mosquito density slider (0-1)

- Population density input    "color": "#ff3366",

- Cases reported input

    "icon": "🔴"- `GET /health` - Health check

---

  },- `GET /model-info` - Model information

## 🔬 Data Science Workflow

  "probabilities": {

### 1. Data Preprocessing

- Handled missing values using median/mode imputation    "low": 3.22,## 📊 Features

- Removed outliers using IQR method

- Feature engineering: `Month_Num` from month names    "moderate": 9.33,

- Label encoding: Low=0, Moderate=1, High=2

- StandardScaler normalization (critical for model accuracy)    "high": 87.45### Input Features (12 total)



### 2. Model Training  }- Year

- Tested 4 algorithms:

  - Logistic Regression (baseline)}- Week

  - Decision Tree

  - Random Forest```- Temperature (°C)

  - **XGBoost** (selected for best performance)

- 80-20 train-test split with stratification- Rainfall (mm)

- Hyperparameter tuning with GridSearchCV

- 5-fold cross-validation### 3. **Get Feature Importance**- Humidity (%)



### 3. Model Validation```http- Air Quality Index (AQI)

- Confusion matrix analysis

- ROC curve and AUC scoresGET /api/feature-importance- Mosquito Density

- Feature importance visualization

- Real-world test scenarios (7 diverse cases, 85.7% accuracy)```- Population Density



### 4. Critical Bug FixReturns real feature importance values from the trained XGBoost model.- Dengue Cases Reported

**Issue**: Model initially predicted 100% High risk due to missing StandardScaler in API

- Latitude

**Solution**: 

- Retrained model and saved `scaler.pkl` separately### 4. **Get Training Statistics**- Longitude

- Updated API to load and apply scaler: `scaler.transform(features)`

- Result: Proper Low/Moderate/High distribution restored```http- Month Number



---GET /api/training-stats



## 🛠️ Technology Stack```### Supported Cities (30)



### BackendReturns aggregated statistics from training data:Mumbai, Delhi, Bangalore, Hyderabad, Chennai, Kolkata, Pune, Ahmedabad, Jaipur, Lucknow, Kanpur, Nagpur, Indore, Thane, Bhopal, Visakhapatnam, Pimpri-Chinchwad, Patna, Vadodara, Ghaziabad, Ludhiana, Agra, Nashik, Faridabad, Meerut, Rajkot, Kalyan-Dombivli, Vasai-Virar, Varanasi, Srinagar

| Technology | Version | Purpose |

|------------|---------|---------|- Monthly risk trends (12 months)

| **Flask** | 3.0.0 | REST API framework |

| **XGBoost** | 2.0.3 | Gradient boosting classifier |- Temperature vs Rainfall correlation data## 🧪 Model Training

| **Scikit-learn** | 1.4.0 | ML utilities & preprocessing |

| **Pandas** | 2.1.4 | Data manipulation |- City-wise risk distribution

| **NumPy** | 1.26.3 | Numerical computing |

| **Flask-CORS** | 4.0.1 | Cross-origin requests |The system was trained on:



### Frontend### 5. **Model Information**- **Dataset:** 10+ years of dengue data (2015-2024)

| Technology | Version | Purpose |

|------------|---------|---------|```http- **Records:** 15,600+ weekly observations

| **Vanilla JavaScript** | ES6 | Frontend logic (no frameworks) |

| **Plotly.js** | 2.27.0 | Interactive data visualization |GET /model-info- **Cities:** 30 major Indian cities

| **CSS Grid & Flexbox** | CSS3 | Responsive layout |

| **Glassmorphism** | CSS | Modern UI design |```- **Features:** 12 environmental and epidemiological factors



### FontsReturns model metadata and performance metrics.

- **Barlow** - Primary UI font (clean, modern)

- **Comfortaa** - Display headings (friendly, rounded)### Hyperparameter Tuning

- **JetBrains Mono** - Monospace numbers (metrics)

## 🎯 Model Performance- **Method:** GridSearchCV with 5-fold cross-validation

---

- **Combinations tested:** 3,840 parameter combinations

## 📊 Data Sources

| Metric | Score |- **Best XGBoost parameters:**

### Training Dataset

- **Total Records**: 15,600 weekly records|--------|-------|  - n_estimators: 200

- **Time Period**: 2015-2023 (8 years)

- **Cities**: 15 major Indian cities| **Accuracy** | 96.57% |  - max_depth: 7

  - Delhi, Mumbai, Bangalore, Chennai, Kolkata

  - Hyderabad, Pune, Ahmedabad, Jaipur, Lucknow| **Precision** | 96.75% |  - learning_rate: 0.1

  - Indore, Bhopal, Nagpur, Surat, Kanpur

| **Recall** | 96.57% |  - subsample: 0.9

### Features (12 total)

1. **Year** - Temporal feature| **F1 Score** | 96.59% |  - colsample_bytree: 0.9

2. **Week** - Week number (1-52)

3. **Temperature (°C)** - Average weekly temperature

4. **Rainfall (mm)** - Average weekly rainfall

5. **Humidity (%)** - Average humidity### Feature Importance (Real Model Values)## 📈 Results

6. **AQI** - Air Quality Index

7. **Mosquito Density** - Index (0-1)1. **Dengue Cases Reported** - 40.01%

8. **Population Density** - People per sq km

9. **Dengue Cases Reported** - Weekly cases2. **Mosquito Density** - 32.16%### Model Comparison

10. **Latitude** - City coordinates

11. **Longitude** - City coordinates3. **Rainfall** - 9.72%

12. **Month_Num** - Month number (1-12)

4. **Week** - 8.24%| Model | Accuracy | Precision | Recall | F1-Score |

### Risk Distribution

- **Low Risk**: 52.2% (8,143 records)5. **Temperature** - 2.18%|-------|----------|-----------|--------|----------|

- **Moderate Risk**: 20.1% (3,136 records)

- **High Risk**: 27.7% (4,321 records)6. **Population Density** - 2.04%| **XGBoost** | **96.57%** | **96.75%** | **96.57%** | **96.59%** |



---7. Other features - 5.65%| Random Forest | 96.57% | 96.75% | 96.57% | 96.59% |



## 🤝 Contributing| Decision Tree | 96.44% | 96.59% | 96.44% | 96.46% |



Contributions are welcome! Please follow these steps:## 🎨 Dashboard Features| Logistic Regression | 89.68% | 89.55% | 89.68% | 89.60% |



1. Fork the repository

2. Create your feature branch

   ```bash### Theme Toggle## 🎨 Dashboard Features

   git checkout -b feature/AmazingFeature

   ```- **Dark Theme**: Glassmorphic design with blur effects

3. Commit your changes

   ```bash- **Light Theme**: Neumorphic design with soft shadows- **Dark/Light Theme Toggle** - Glassmorphic design

   git commit -m 'Add some AmazingFeature'

   ```- Persistent theme preference saved to localStorage- **Real-time Predictions** - Instant risk assessment

4. Push to the branch

   ```bash- **Interactive Charts** - Plotly.js visualizations

   git push origin feature/AmazingFeature

   ```### Responsive Layout- **Confidence Score** - Model certainty indicator

5. Open a Pull Request

- **Desktop (>1400px)**: 3-column grid (sidebar | results | charts)- **Risk Recommendations** - Actionable advice

### Development Guidelines

- Follow PEP 8 for Python code- **Tablet (768px-1400px)**: 2-column layout- **Responsive Design** - Mobile-friendly interface

- Use ESLint for JavaScript

- Add comments for complex logic- **Mobile (<768px)**: Single column stack- **Offline Mode** - Fallback rule-based prediction

- Update documentation for new features

- Test thoroughly before submitting PR



---### Chart Interactions## 🔒 Risk Level Mapping



## 📝 License- Click any chart to open full-screen modal



This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.- Real-time data loading from API- **0 = Low Risk** (🟢 Blue) - Routine monitoring



---- Smooth animations and transitions- **1 = Moderate Risk** (🟡 Green) - Enhanced surveillance



## 🙏 Acknowledgments- Automatic theme adaptation- **2 = High Risk** (🔴 Red) - Immediate action required



- **Dataset**: Based on Indian dengue outbreak historical records (2015-2023)

- **XGBoost**: Excellent gradient boosting library and community

- **Plotly.js**: Beautiful interactive visualization library## 🔬 Data Science Workflow## 🛠️ Technologies Used

- **Flask**: Lightweight and powerful web framework

- **Indian Health Departments**: For providing dengue outbreak data



---### 1. Data Preprocessing### Backend



## 📧 Contact & Support- Handled missing values and outliers- Python 3.13



- **GitHub**: [@Chanu716](https://github.com/Chanu716)- Feature engineering (Month_Num, seasonal indicators)- Flask - Web framework

- **Repository**: [Health.env](https://github.com/Chanu716/Health.env)

- **Issues**: [Report a bug](https://github.com/Chanu716/Health.env/issues)- Label encoding for risk levels (Low=0, Moderate=1, High=2)- XGBoost - ML model



For questions, feedback, or collaboration opportunities, please open an issue on GitHub.- StandardScaler normalization- scikit-learn - Model evaluation



---- pandas & numpy - Data processing



## 🎓 Future Enhancements### 2. Model Training



- [ ] Deploy on cloud (AWS/Azure/Heroku)- Tested 4 algorithms: Logistic Regression, Decision Tree, Random Forest, XGBoost### Frontend

- [ ] Add batch prediction endpoint

- [ ] Implement user authentication- XGBoost selected for best performance- HTML5 / CSS3

- [ ] Mobile app version

- [ ] Real-time weather API integration- Hyperparameter tuning with GridSearchCV- JavaScript (ES6+)

- [ ] Historical prediction tracking

- [ ] Email alerts for high-risk predictions- 80-20 train-test split with stratification- Plotly.js - Interactive charts

- [ ] Model retraining pipeline

- [ ] A/B testing framework- Fetch API - HTTP requests

- [ ] Multi-language support

### 3. Model Validation

---

- Cross-validation (5-fold CV)## 📝 Development Workflow

<div align="center">

- Confusion matrix analysis

**Made with ❤️ for better dengue outbreak prevention**

- Feature importance visualization1. **Data Collection** - Gathered dengue data from multiple sources

⭐ Star this repo if you found it helpful!

- Real-world test scenarios validated2. **Data Preprocessing** - Cleaned, encoded, and normalized data

</div>

3. **EDA** - Comprehensive exploratory analysis

## 🛠️ Technology Stack4. **Model Training** - Trained 4 different models

5. **Hyperparameter Tuning** - Optimized XGBoost parameters

### Backend6. **Model Evaluation** - Compared performance metrics

- **Flask 3.0.0** - REST API framework7. **API Development** - Created Flask REST API

- **XGBoost 2.0.3** - Gradient boosting classifier8. **Frontend Integration** - Built interactive dashboard

- **Scikit-learn 1.4.0** - ML utilities and preprocessing

- **Pandas 2.1.4** - Data manipulation## 🐛 Troubleshooting

- **NumPy 1.26.3** - Numerical computing

### API Not Starting

### Frontend```bash

- **Vanilla JavaScript** - No frameworks, pure JS# Check if port 5000 is available

- **Plotly.js 2.27.0** - Interactive data visualizationnetstat -ano | findstr :5000

- **CSS Grid & Flexbox** - Responsive layout

- **Glassmorphism** - Modern UI design trend# Try different port

python api.py  # Edit api.py to change port

### Fonts```

- **Barlow** - Primary UI font

- **Comfortaa** - Display headings### Model Not Loading

- **JetBrains Mono** - Monospace numbers```bash

# Verify model file exists

## 📊 Data Sourcesdir models\xgboost_model.pkl



The model is trained on real dengue outbreak data including:# Check file size (should be ~500KB)

- **15,600 weekly records** across 15 Indian cities```

- **Time period**: 2015-2023 (8 years)

- **Features**: Temperature, Rainfall, Humidity, AQI, Mosquito Density, Population, Cases### CORS Errors

- **Cities**: Mumbai, Delhi, Bangalore, Chennai, Kolkata, and 10 more- Ensure Flask-CORS is installed

- Check browser console for details

## 🤝 Contributing- Try using a local web server instead of file://



Contributions are welcome! Please feel free to submit a Pull Request.## 📚 References



1. Fork the repository- XGBoost Documentation: https://xgboost.readthedocs.io/

2. Create your feature branch (`git checkout -b feature/AmazingFeature`)- Flask Documentation: https://flask.palletsprojects.com/

3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)- Plotly.js Documentation: https://plotly.com/javascript/

4. Push to the branch (`git push origin feature/AmazingFeature`)

5. Open a Pull Request## 👥 Contributors



## 📝 LicenseBuilt as part of ML-powered dengue prediction system.



This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.## 📄 License



## 🙏 AcknowledgmentsThis project is for educational and research purposes.



- Dataset based on Indian dengue outbreak historical records---

- XGBoost documentation and community

- Plotly.js for excellent visualization library**🦟 Stay Safe, Predict Smart!**

- Flask documentation for API best practices

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Made with ❤️ for better dengue outbreak prevention**
