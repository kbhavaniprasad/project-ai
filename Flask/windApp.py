from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load the trained model, scaler, and feature names
MODEL_PATH = 'power_prediction.sav'
SCALER_PATH = 'scaler.sav'
FEATURES_PATH = 'feature_names.pkl'

# Global variables to store loaded objects
model = None
scaler = None
feature_names = None

def load_model_components():
    """Load model, scaler, and feature names"""
    global model, scaler, feature_names
    
    try:
        if os.path.exists(MODEL_PATH):
            model = pickle.load(open(MODEL_PATH, 'rb'))
            print("✓ Model loaded successfully")
        else:
            print(f"⚠ Model file not found: {MODEL_PATH}")
            
        if os.path.exists(SCALER_PATH):
            scaler = pickle.load(open(SCALER_PATH, 'rb'))
            print("✓ Scaler loaded successfully")
        else:
            print(f"⚠ Scaler file not found: {SCALER_PATH}")
            
        if os.path.exists(FEATURES_PATH):
            feature_names = pickle.load(open(FEATURES_PATH, 'rb'))
            print(f"✓ Feature names loaded: {feature_names}")
        else:
            print(f"⚠ Feature names file not found: {FEATURES_PATH}")
            
    except Exception as e:
        print(f"Error loading model components: {str(e)}")

# Load model components when app starts
load_model_components()

@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html', 
                         feature_names=feature_names if feature_names else [],
                         prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        if model is None or scaler is None:
            return render_template('index.html', 
                                 feature_names=feature_names if feature_names else [],
                                 prediction=None,
                                 error="Model not loaded. Please train the model first using the Jupyter notebook.")
        
        # Get form data
        if request.method == 'POST':
            # Collect input features from form
            input_features = []
            feature_values = {}
            
            if feature_names:
                for feature in feature_names:
                    value = float(request.form.get(feature, 0))
                    input_features.append(value)
                    feature_values[feature] = value
            else:
                # Fallback: get all numeric inputs
                for key in request.form:
                    if key != 'submit':
                        input_features.append(float(request.form.get(key, 0)))
            
            # Convert to numpy array and reshape
            input_array = np.array(input_features).reshape(1, -1)
            
            # Scale the input
            input_scaled = scaler.transform(input_array)
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            
            # Round to 2 decimal places
            prediction = round(prediction, 2)
            
            return render_template('index.html', 
                                 feature_names=feature_names if feature_names else [],
                                 prediction=prediction,
                                 input_values=feature_values if feature_names else None)
    
    except Exception as e:
        return render_template('index.html', 
                             feature_names=feature_names if feature_names else [],
                             prediction=None,
                             error=f"Error making prediction: {str(e)}")

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions (JSON)"""
    try:
        if model is None or scaler is None:
            return jsonify({
                'error': 'Model not loaded',
                'success': False
            }), 500
        
        # Get JSON data
        data = request.get_json()
        
        # Extract features
        if feature_names:
            input_features = [float(data.get(feature, 0)) for feature in feature_names]
        else:
            input_features = list(data.values())
        
        # Convert to numpy array and reshape
        input_array = np.array(input_features).reshape(1, -1)
        
        # Scale the input
        input_scaled = scaler.transform(input_array)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        return jsonify({
            'prediction': round(float(prediction), 2),
            'unit': 'kW',
            'success': True
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 400

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'features': feature_names if feature_names else []
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🌬️  WIND TURBINE ENERGY PREDICTION SERVER")
    print("="*60)
    print("Server starting on http://127.0.0.1:5000")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
