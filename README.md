# Wind Turbine Energy Output Prediction

## 🌬️ Weather-Based Prediction: A Next-Generation Approach to Renewable Energy Management

This project predicts the energy output of wind turbines based on weather conditions using machine learning. It helps energy companies, grid operators, and wind farm managers optimize energy production, plan maintenance, and integrate renewable energy efficiently.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Web Application](#web-application)
- [API Documentation](#api-documentation)
- [Technical Details](#technical-details)
- [Use Cases](#use-cases)

---

## 🎯 Overview

The project analyzes historical data of weather conditions and energy output to train machine learning models that can predict wind turbine energy generation. This enables:

1. **Energy Production Forecasting** - Predict energy output based on weather forecasts
2. **Maintenance Planning** - Schedule maintenance during low wind activity periods
3. **Grid Integration** - Balance the grid by predicting wind energy availability

---

## ✨ Features

- **Comprehensive Data Analysis** - Complete EDA with visualizations
- **Multiple ML Models** - Linear Regression, Random Forest, Decision Tree
- **Model Comparison** - Automatic selection of best-performing model
- **Web Interface** - User-friendly Flask application
- **REST API** - JSON-based prediction endpoint
- **Real-time Predictions** - Instant energy output forecasts
- **Responsive Design** - Modern, mobile-friendly UI

---

## 📁 Project Structure

```
intern-project/
│
├── Wind_mill_model.ipynb          # Main Jupyter notebook for model training
├── T1.csv                          # Wind turbine dataset
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
│
└── Flask/                          # Web application
    ├── windApp.py                  # Flask server
    ├── power_prediction.sav        # Trained model (generated)
    ├── scaler.sav                  # Feature scaler (generated)
    ├── feature_names.pkl           # Feature names (generated)
    │
    ├── templates/
    │   └── index.html              # Web interface
    │
    └── static/
        └── (images/styles)         # Static assets
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Jupyter Notebook or JupyterLab

### Step 1: Clone or Download the Project

```bash
cd e:\intern-project
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Verify Installation

```bash
python -c "import pandas, numpy, sklearn, flask; print('✓ All dependencies installed!')"
```

---

## 📊 Usage

### Complete Workflow

1. **Train the Model** (using Jupyter Notebook)
2. **Run the Web Application** (using Flask)
3. **Make Predictions** (via Web UI or API)

---

## 🧠 Model Training

### Step 1: Open Jupyter Notebook

```bash
jupyter notebook Wind_mill_model.ipynb
```

### Step 2: Run All Cells

Execute all cells in the notebook sequentially. The notebook will:

1. Load and preprocess the `T1.csv` dataset
2. Perform exploratory data analysis
3. Train three regression models:
   - Linear Regression
   - Random Forest Regression
   - Decision Tree Regression
4. Compare model performance
5. Save the best model to `Flask/power_prediction.sav`
6. Save the scaler to `Flask/scaler.sav`
7. Save feature names to `Flask/feature_names.pkl`

### Step 3: Review Results

The notebook provides:
- Correlation heatmaps
- Distribution plots
- Model comparison charts
- Actual vs Predicted visualizations
- Residual analysis
- Performance metrics (R², MAE, RMSE)

---

## 🌐 Web Application

### Start the Flask Server

```bash
cd Flask
python windApp.py
```

The server will start on `http://127.0.0.1:5000`

### Access the Web Interface

Open your browser and navigate to:
```
http://127.0.0.1:5000
```

### Make Predictions

1. Enter the required weather parameters:
   - Wind Speed (m/s)
   - Wind Direction (degrees)
   - Theoretical Power (kW)
   - (Other features based on your dataset)

2. Click "Predict Energy Output"

3. View the predicted power output in kilowatts (kW)

---

## 🔌 API Documentation

### Prediction Endpoint

**URL:** `/api/predict`  
**Method:** `POST`  
**Content-Type:** `application/json`

#### Request Body

```json
{
  "Wind_Speed": 12.5,
  "Wind_Direction": 180.0,
  "Theoretical_Power": 1500.0
}
```

#### Response

```json
{
  "prediction": 1425.67,
  "unit": "kW",
  "success": true
}
```

#### Error Response

```json
{
  "error": "Error message",
  "success": false
}
```

### Health Check Endpoint

**URL:** `/health`  
**Method:** `GET`

#### Response

```json
{
  "status": "healthy",
  "model_loaded": true,
  "scaler_loaded": true,
  "features": ["Wind_Speed", "Wind_Direction", "Theoretical_Power"]
}
```

### Example cURL Request

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"Wind_Speed": 12.5, "Wind_Direction": 180.0, "Theoretical_Power": 1500.0}'
```

---

## 🔬 Technical Details

### Dataset

The `T1.csv` dataset contains:
- **Wind Speed** - Speed of wind in m/s
- **Wind Direction** - Direction of wind in degrees
- **Theoretical Power** - Calculated theoretical power output
- **Actual Power** - Real measured power output (target variable)

### Data Preprocessing

1. **Missing Value Handling** - Median imputation for numerical features
2. **Feature Scaling** - StandardScaler normalization
3. **Train-Test Split** - 80% training, 20% testing
4. **Correlation Analysis** - Identify key predictive features

### Machine Learning Models

#### 1. Linear Regression
- Simple, interpretable baseline model
- Fast training and prediction
- Good for linear relationships

#### 2. Random Forest Regression
- Ensemble of decision trees
- Handles non-linear relationships
- Provides feature importance
- Parameters: 100 estimators, max_depth=20

#### 3. Decision Tree Regression
- Single tree model
- Captures non-linear patterns
- Parameters: max_depth=15

### Model Evaluation Metrics

- **R² Score** - Coefficient of determination (higher is better)
- **MAE** - Mean Absolute Error (lower is better)
- **RMSE** - Root Mean Squared Error (lower is better)

### Model Selection

The best model is automatically selected based on the highest Test R² Score.

---

## 💼 Use Cases

### Scenario 1: Energy Production Forecasting

**Challenge:** Energy companies need to forecast production for grid management.

**Solution:** Use weather forecasts as input to predict energy output, enabling:
- Better energy distribution planning
- Optimized pricing strategies
- Reduced energy waste

### Scenario 2: Maintenance Planning

**Challenge:** Wind farm operators need to minimize downtime.

**Solution:** Predict low-wind periods for maintenance scheduling:
- Schedule maintenance during predicted low output
- Maximize turbine availability during high-wind periods
- Reduce revenue loss from downtime

### Scenario 3: Grid Integration

**Challenge:** Grid operators need to balance renewable and traditional energy.

**Solution:** Predict wind energy availability to:
- Adjust output from other energy sources
- Prevent grid instability
- Optimize renewable energy utilization

---

## 📈 Performance Expectations

Based on typical wind turbine datasets:

- **R² Score:** 0.85 - 0.95 (85-95% variance explained)
- **MAE:** 50 - 150 kW (depending on turbine capacity)
- **RMSE:** 75 - 200 kW

*Note: Actual performance depends on data quality and feature engineering.*

---

## 🛠️ Troubleshooting

### Model Files Not Found

**Error:** "Model not loaded. Please train the model first."

**Solution:** Run the Jupyter notebook to generate model files:
```bash
jupyter notebook Wind_mill_model.ipynb
```

### Import Errors

**Error:** "ModuleNotFoundError: No module named 'sklearn'"

**Solution:** Install dependencies:
```bash
pip install -r requirements.txt
```

### Port Already in Use

**Error:** "Address already in use"

**Solution:** Change the port in `windApp.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Change to 5001
```

---

## 🔮 Future Enhancements

- [ ] Real-time weather API integration
- [ ] Historical prediction tracking
- [ ] Multi-turbine farm predictions
- [ ] Advanced ensemble models (XGBoost, LightGBM)
- [ ] Automated model retraining pipeline
- [ ] Dashboard with analytics
- [ ] Mobile application

---

## 📝 License

This project is created for educational and research purposes.

---

## 👥 Contributors

- Data Science Team
- Machine Learning Engineers
- Web Development Team

---

## 📧 Contact

For questions or support, please contact the project maintainers.

---

## 🙏 Acknowledgments

- Wind turbine dataset providers
- Open-source ML community
- Flask framework developers
- Scikit-learn contributors

---

**Built with ❤️ for a sustainable energy future**
