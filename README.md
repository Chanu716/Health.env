# 🦟 Dengue Risk Prediction System# 🦟 Dengue Risk Prediction System



[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)A complete end-to-end machine learning system for predicting dengue outbreak risk levels in Indian cities.

[![Flask](https://img.shields.io/badge/Flask-3.0.0-green.svg)](https://flask.palletsprojects.com/)

[![XGBoost](https://img.shields.io/badge/XGBoost-ML-orange.svg)](https://xgboost.readthedocs.io/)## 📊 Project Overview

[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This system uses **XGBoost** machine learning model (96.57% accuracy) to predict dengue outbreak risk levels based on weather, environmental, and epidemiological factors.

An intelligent machine learning system for predicting dengue outbreak risk levels across Indian cities using real-time environmental and demographic data. The system features a modern glassmorphic dashboard with **real-time data visualization** powered by authentic model insights.

## 🏆 Model Performance

## ✨ Key Features

- **Best Model:** XGBoost Classifier

### 🤖 **Machine Learning Model**- **Accuracy:** 96.57%

- **XGBoost Classifier** with 96.57% accuracy- **Precision:** 96.75%

- Trained on 15,600+ historical dengue outbreak records- **Recall:** 96.57%

- Proper feature scaling with StandardScaler- **F1-Score:** 96.59%

- Multi-class classification (Low, Moderate, High risk)

- Real-time predictions via REST API## 📁 Project Structure



### 📊 **Real Data Visualization**```

- **Monthly Risk Trends**: Actual seasonal patterns from training datahealth.env/

- **Temperature vs Rainfall Correlation**: Real scatter plots by risk level├── api.py                          # Flask API for model serving

- **Feature Importance**: Live XGBoost model feature weights├── index.html                      # Frontend dashboard

- **City Risk Distribution**: Authentic city-wise risk statistics├── script.js                       # Dashboard JavaScript logic

- All charts load real data from the trained model and dataset├── styles.css                      # Dashboard styling

├── requirements.txt                # Python dependencies

### 🎨 **Modern Dashboard**│

- Glassmorphic UI with dark/light theme toggle├── models/                         # Trained ML models

- Responsive 3-column layout│   ├── xgboost_model.pkl          # Best model (96.57% accuracy)

- Interactive Plotly.js charts with zoom functionality│   ├── random_forest_model.pkl    # Alternative model

- Real-time risk predictions with confidence scores│   ├── decision_tree_model.pkl    # Baseline model

- Personalized recommendations based on risk level│   └── logistic_regression_model.pkl

│

### ⚡ **Performance Optimized**├── data/                           # Datasets

- 5-minute data caching to reduce API calls│   ├── dengue_data_cleaned.csv    # Cleaned training data

- Lazy loading of chart data│   ├── dengue_india_weekly_with_nulls.csv  # Raw data

- Fallback static data for offline mode│   └── *.csv                       # Other data files

- Minimal bundle size with CDN resources│

├── notebooks/                      # Jupyter notebooks

## 🚀 Quick Start│   ├── data_preprocessing.ipynb   # Data cleaning pipeline

│   ├── exploratory_data_analysis.ipynb  # EDA visualizations

### Prerequisites│   └── model_training_evaluation.ipynb  # Model training & tuning

```bash│

Python 3.8+├── results/                        # Training results

pip (Python package manager)│   ├── *.png                       # Performance visualizations

```│   ├── hyperparameter_tuning_results.json

│   └── model_comparison_results.csv

### Installation│

└── docs/                           # Documentation

1. **Clone the repository**    ├── README_Dashboard.md

```bash    └── HYPERPARAMETER_TUNING_GUIDE.md

git clone https://github.com/yourusername/dengue-risk-prediction.git```

cd dengue-risk-prediction

```## 🚀 Quick Start



2. **Install dependencies**### 1. Install Dependencies

```bash

pip install -r requirements.txt```bash

```pip install -r requirements.txt

```

3. **Run the API server**

```bash### 2. Start the API Server

python api.py

``````bash

python api.py

4. **Open the dashboard**```

```bash

# In a new terminalThe API will start on `http://localhost:5000`

python -m http.server 8000

```### 3. Open the Dashboard



5. **Access the application**Open `index.html` in your web browser, or use a local server:

- Dashboard: http://localhost:8000/index.html

- API Health Check: http://localhost:5000/health```bash

- API Documentation: http://localhost:5000/model-info# Using Python

python -m http.server 8000

## 📁 Project Structure

# Then visit: http://localhost:8000

``````

dengue-risk-prediction/

├── api.py                      # Flask REST API with real data endpoints## 🎯 How to Use

├── index.html                  # Main dashboard HTML

├── script.js                   # Frontend logic with real data loading### Dashboard Input Fields

├── styles.css                  # Glassmorphic CSS styling

├── requirements.txt            # Python dependencies1. **City** - Select from 30 Indian cities

│2. **Month** - Choose the month for prediction

├── models/                     # Trained models3. **Temperature** - Average temperature (20-45°C)

│   ├── xgboost_model.pkl      # Main XGBoost classifier (96.57% accuracy)4. **Rainfall** - Monthly rainfall (0-3000mm)

│   ├── scaler.pkl             # StandardScaler for preprocessing5. **Humidity** - Relative humidity (40-100%)

│   ├── feature_names.pkl      # Feature order preservation6. **AQI** - Air Quality Index (0-500)

│   └── ...                    # Other model files7. **Mosquito Density** - Breeding site density (0-1)

│8. **Population Density** - People per sq km

├── data/                       # Training datasets9. **Cases Reported** - Previous dengue cases

│   ├── dengue_data_cleaned.csv               # Cleaned training data

│   └── dengue_india_weekly_with_nulls.csv    # Original dataset### Making Predictions

│

├── notebooks/                  # Jupyter notebooks1. Fill in all input fields

│   ├── data_preprocessing.ipynb              # Data cleaning pipeline2. Click "Predict Risk" button

│   ├── exploratory_data_analysis.ipynb       # EDA visualizations3. View risk level:

│   └── model_training_evaluation.ipynb       # Model training & tuning   - 🔴 **HIGH RISK** (Red)

│   - 🟡 **MODERATE RISK** (Yellow)

├── docs/                       # Documentation   - 🟢 **LOW RISK** (Green)

│   ├── HYPERPARAMETER_TUNING_GUIDE.md

│   └── README_Dashboard.md## 🔧 API Usage

│

└── results/                    # Model outputs### Endpoint: `/predict`

    ├── model_comparison_results.csv

    └── tuning_comparison.csv**Method:** POST

```

**Request Body:**

## 🔧 API Endpoints```json

{

### 1. **Health Check**  "city": "Mumbai",

```http  "month": "July",

GET /health  "temperature": 32.5,

```  "rainfall": 1200,

Returns API status and uptime.  "humidity": 85,

  "aqi": 150,

### 2. **Predict Risk**  "mosquito": 0.75,

```http  "population": 5000,

POST /predict  "cases": 450

Content-Type: application/json}

```

{

  "city": "Mumbai",**Response:**

  "month": "July",```json

  "temperature": 28.5,{

  "rainfall": 150.0,  "success": true,

  "humidity": 85.0,  "prediction": {

  "aqi": 120,    "risk_level": "High",

  "mosquito": 0.8,    "risk_code": 2,

  "population": 50000,    "confidence": 87.5,

  "cases": 45    "color": "#ff3366",

}    "icon": "🔴"

```  },

  "probabilities": {

**Response:**    "low": 5.2,

```json    "moderate": 7.3,

{    "high": 87.5

  "success": true,  }

  "prediction": {}

    "risk_level": "High",```

    "risk_code": 2,

    "confidence": 87.45,### Other Endpoints

    "color": "#ff3366",

    "icon": "🔴"- `GET /health` - Health check

  },- `GET /model-info` - Model information

  "probabilities": {

    "low": 3.22,## 📊 Features

    "moderate": 9.33,

    "high": 87.45### Input Features (12 total)

  }- Year

}- Week

```- Temperature (°C)

- Rainfall (mm)

### 3. **Get Feature Importance**- Humidity (%)

```http- Air Quality Index (AQI)

GET /api/feature-importance- Mosquito Density

```- Population Density

Returns real feature importance values from the trained XGBoost model.- Dengue Cases Reported

- Latitude

### 4. **Get Training Statistics**- Longitude

```http- Month Number

GET /api/training-stats

```### Supported Cities (30)

Returns aggregated statistics from training data:Mumbai, Delhi, Bangalore, Hyderabad, Chennai, Kolkata, Pune, Ahmedabad, Jaipur, Lucknow, Kanpur, Nagpur, Indore, Thane, Bhopal, Visakhapatnam, Pimpri-Chinchwad, Patna, Vadodara, Ghaziabad, Ludhiana, Agra, Nashik, Faridabad, Meerut, Rajkot, Kalyan-Dombivli, Vasai-Virar, Varanasi, Srinagar

- Monthly risk trends (12 months)

- Temperature vs Rainfall correlation data## 🧪 Model Training

- City-wise risk distribution

The system was trained on:

### 5. **Model Information**- **Dataset:** 10+ years of dengue data (2015-2024)

```http- **Records:** 15,600+ weekly observations

GET /model-info- **Cities:** 30 major Indian cities

```- **Features:** 12 environmental and epidemiological factors

Returns model metadata and performance metrics.

### Hyperparameter Tuning

## 🎯 Model Performance- **Method:** GridSearchCV with 5-fold cross-validation

- **Combinations tested:** 3,840 parameter combinations

| Metric | Score |- **Best XGBoost parameters:**

|--------|-------|  - n_estimators: 200

| **Accuracy** | 96.57% |  - max_depth: 7

| **Precision** | 96.75% |  - learning_rate: 0.1

| **Recall** | 96.57% |  - subsample: 0.9

| **F1 Score** | 96.59% |  - colsample_bytree: 0.9



### Feature Importance (Real Model Values)## 📈 Results

1. **Dengue Cases Reported** - 40.01%

2. **Mosquito Density** - 32.16%### Model Comparison

3. **Rainfall** - 9.72%

4. **Week** - 8.24%| Model | Accuracy | Precision | Recall | F1-Score |

5. **Temperature** - 2.18%|-------|----------|-----------|--------|----------|

6. **Population Density** - 2.04%| **XGBoost** | **96.57%** | **96.75%** | **96.57%** | **96.59%** |

7. Other features - 5.65%| Random Forest | 96.57% | 96.75% | 96.57% | 96.59% |

| Decision Tree | 96.44% | 96.59% | 96.44% | 96.46% |

## 🎨 Dashboard Features| Logistic Regression | 89.68% | 89.55% | 89.68% | 89.60% |



### Theme Toggle## 🎨 Dashboard Features

- **Dark Theme**: Glassmorphic design with blur effects

- **Light Theme**: Neumorphic design with soft shadows- **Dark/Light Theme Toggle** - Glassmorphic design

- Persistent theme preference saved to localStorage- **Real-time Predictions** - Instant risk assessment

- **Interactive Charts** - Plotly.js visualizations

### Responsive Layout- **Confidence Score** - Model certainty indicator

- **Desktop (>1400px)**: 3-column grid (sidebar | results | charts)- **Risk Recommendations** - Actionable advice

- **Tablet (768px-1400px)**: 2-column layout- **Responsive Design** - Mobile-friendly interface

- **Mobile (<768px)**: Single column stack- **Offline Mode** - Fallback rule-based prediction



### Chart Interactions## 🔒 Risk Level Mapping

- Click any chart to open full-screen modal

- Real-time data loading from API- **0 = Low Risk** (🟢 Blue) - Routine monitoring

- Smooth animations and transitions- **1 = Moderate Risk** (🟡 Green) - Enhanced surveillance

- Automatic theme adaptation- **2 = High Risk** (🔴 Red) - Immediate action required



## 🔬 Data Science Workflow## 🛠️ Technologies Used



### 1. Data Preprocessing### Backend

- Handled missing values and outliers- Python 3.13

- Feature engineering (Month_Num, seasonal indicators)- Flask - Web framework

- Label encoding for risk levels (Low=0, Moderate=1, High=2)- XGBoost - ML model

- StandardScaler normalization- scikit-learn - Model evaluation

- pandas & numpy - Data processing

### 2. Model Training

- Tested 4 algorithms: Logistic Regression, Decision Tree, Random Forest, XGBoost### Frontend

- XGBoost selected for best performance- HTML5 / CSS3

- Hyperparameter tuning with GridSearchCV- JavaScript (ES6+)

- 80-20 train-test split with stratification- Plotly.js - Interactive charts

- Fetch API - HTTP requests

### 3. Model Validation

- Cross-validation (5-fold CV)## 📝 Development Workflow

- Confusion matrix analysis

- Feature importance visualization1. **Data Collection** - Gathered dengue data from multiple sources

- Real-world test scenarios validated2. **Data Preprocessing** - Cleaned, encoded, and normalized data

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
