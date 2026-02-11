"""
Wind Turbine Energy Output Prediction - Training Script
This script executes the complete ML pipeline from the Jupyter notebook
"""

# Data manipulation and analysis
import pandas as pd
import numpy as np

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Machine Learning Models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

# Model Evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Model Persistence
import pickle

# Warnings
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("="*60)
print("WIND TURBINE ENERGY PREDICTION - MODEL TRAINING")
print("="*60)
print("\n✓ All libraries imported successfully!\n")

# ============================================================================
# STEP 1: Load Dataset
# ============================================================================
print("STEP 1: Loading Dataset...")
# Load CSV without parsing dates to avoid errors
df = pd.read_csv('T1.csv', low_memory=False)

# Drop Date/Time column if it exists (not useful for prediction)
if 'Date/Time' in df.columns:
    df = df.drop(columns=['Date/Time'])
    print("✓ Dropped 'Date/Time' column (not needed for prediction)")

print(f"✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"\nColumn names: {df.columns.tolist()}")
print(f"\nFirst 5 rows:")
print(df.head())

# ============================================================================
# STEP 2: Check for Null Values
# ============================================================================
print("\n" + "="*60)
print("STEP 2: Checking for Null Values...")
null_counts = df.isnull().sum()
print(null_counts)
if null_counts.sum() == 0:
    print("✓ No null values found!")
else:
    print(f"⚠ Found {null_counts.sum()} null values")

# ============================================================================
# STEP 3: Handle Missing Data
# ============================================================================
print("\n" + "="*60)
print("STEP 3: Handling Missing Data...")
original_shape = df.shape

# Fill with median for numerical columns
for col in df.select_dtypes(include=[np.number]).columns:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)
        print(f"✓ Filled {col} with median value")

# Fill with mode for categorical columns
for col in df.select_dtypes(include=['object']).columns:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)
        print(f"✓ Filled {col} with mode value")

print(f"✓ Data shape maintained: {df.shape}")

# ============================================================================
# STEP 4: Statistical Summary
# ============================================================================
print("\n" + "="*60)
print("STEP 4: Statistical Summary")
print("="*60)
print(df.describe())

# ============================================================================
# STEP 5: Correlation Analysis
# ============================================================================
print("\n" + "="*60)
print("STEP 5: Correlation Analysis...")
correlation_matrix = df.corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Save correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            fmt='.2f', square=True, linewidths=1)
plt.title('Correlation Heatmap - Wind Turbine Features', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Correlation heatmap saved as 'correlation_heatmap.png'")
plt.close()

# ============================================================================
# STEP 6: Feature Engineering
# ============================================================================
print("\n" + "="*60)
print("STEP 6: Feature Engineering...")

# Define target column (last column by default)
target_column = df.columns[-1]
print(f"Target Variable: {target_column}")

# Select features and target
X = df.drop(columns=[target_column])
y = df[target_column]

print(f"Feature Variables: {X.columns.tolist()}")
print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Handle categorical columns if any
categorical_cols = X.select_dtypes(include=['object']).columns
if len(categorical_cols) > 0:
    print(f"\n✓ Encoding categorical columns: {categorical_cols.tolist()}")
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
else:
    print("✓ No categorical columns found - all features are numerical")

# ============================================================================
# STEP 7: Feature Scaling
# ============================================================================
print("\n" + "="*60)
print("STEP 7: Feature Scaling...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
print("✓ Feature scaling completed using StandardScaler")

# ============================================================================
# STEP 8: Train-Test Split
# ============================================================================
print("\n" + "="*60)
print("STEP 8: Train-Test Split...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print(f"✓ Training set: {X_train.shape[0]} samples ({(X_train.shape[0]/len(df))*100:.1f}%)")
print(f"✓ Testing set: {X_test.shape[0]} samples ({(X_test.shape[0]/len(df))*100:.1f}%)")

# ============================================================================
# STEP 9: Model Training - Linear Regression
# ============================================================================
print("\n" + "="*60)
print("STEP 9: Training Linear Regression Model...")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

y_pred_lr_train = lr_model.predict(X_train)
y_pred_lr_test = lr_model.predict(X_test)

lr_train_r2 = r2_score(y_train, y_pred_lr_train)
lr_test_r2 = r2_score(y_test, y_pred_lr_test)
lr_mae = mean_absolute_error(y_test, y_pred_lr_test)
lr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr_test))

print("LINEAR REGRESSION PERFORMANCE:")
print(f"  Training R² Score: {lr_train_r2:.4f}")
print(f"  Testing R² Score: {lr_test_r2:.4f}")
print(f"  MAE: {lr_mae:.4f}")
print(f"  RMSE: {lr_rmse:.4f}")

# ============================================================================
# STEP 10: Model Training - Random Forest
# ============================================================================
print("\n" + "="*60)
print("STEP 10: Training Random Forest Model...")
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

y_pred_rf_train = rf_model.predict(X_train)
y_pred_rf_test = rf_model.predict(X_test)

rf_train_r2 = r2_score(y_train, y_pred_rf_train)
rf_test_r2 = r2_score(y_test, y_pred_rf_test)
rf_mae = mean_absolute_error(y_test, y_pred_rf_test)
rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf_test))

print("RANDOM FOREST PERFORMANCE:")
print(f"  Training R² Score: {rf_train_r2:.4f}")
print(f"  Testing R² Score: {rf_test_r2:.4f}")
print(f"  MAE: {rf_mae:.4f}")
print(f"  RMSE: {rf_rmse:.4f}")

# ============================================================================
# STEP 11: Model Training - Decision Tree
# ============================================================================
print("\n" + "="*60)
print("STEP 11: Training Decision Tree Model...")
dt_model = DecisionTreeRegressor(
    random_state=42,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2
)
dt_model.fit(X_train, y_train)

y_pred_dt_train = dt_model.predict(X_train)
y_pred_dt_test = dt_model.predict(X_test)

dt_train_r2 = r2_score(y_train, y_pred_dt_train)
dt_test_r2 = r2_score(y_test, y_pred_dt_test)
dt_mae = mean_absolute_error(y_test, y_pred_dt_test)
dt_rmse = np.sqrt(mean_squared_error(y_test, y_pred_dt_test))

print("DECISION TREE PERFORMANCE:")
print(f"  Training R² Score: {dt_train_r2:.4f}")
print(f"  Testing R² Score: {dt_test_r2:.4f}")
print(f"  MAE: {dt_mae:.4f}")
print(f"  RMSE: {dt_rmse:.4f}")

# ============================================================================
# STEP 12: Model Comparison
# ============================================================================
print("\n" + "="*60)
print("STEP 12: Model Comparison")
print("="*60)

model_comparison = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest', 'Decision Tree'],
    'Train R² Score': [lr_train_r2, rf_train_r2, dt_train_r2],
    'Test R² Score': [lr_test_r2, rf_test_r2, dt_test_r2],
    'MAE': [lr_mae, rf_mae, dt_mae],
    'RMSE': [lr_rmse, rf_rmse, dt_rmse]
})

print(model_comparison.to_string(index=False))

# Identify best model
best_model_idx = model_comparison['Test R² Score'].idxmax()
best_model_name = model_comparison.loc[best_model_idx, 'Model']
best_r2 = model_comparison.loc[best_model_idx, 'Test R² Score']

print("\n" + "="*60)
print(f"🏆 BEST MODEL: {best_model_name}")
print(f"   Test R² Score: {best_r2:.4f}")
print("="*60)

# Select best model
if best_model_name == 'Linear Regression':
    best_model = lr_model
elif best_model_name == 'Random Forest':
    best_model = rf_model
else:
    best_model = dt_model

# ============================================================================
# STEP 13: Save Model
# ============================================================================
print("\n" + "="*60)
print("STEP 13: Saving Model...")

# Save the best model
model_filename = 'Flask/power_prediction.sav'
pickle.dump(best_model, open(model_filename, 'wb'))
print(f"✓ Best model ({best_model_name}) saved as '{model_filename}'")

# Save the scaler
scaler_filename = 'Flask/scaler.sav'
pickle.dump(scaler, open(scaler_filename, 'wb'))
print(f"✓ Scaler saved as '{scaler_filename}'")

# Save feature names
feature_names_filename = 'Flask/feature_names.pkl'
pickle.dump(X.columns.tolist(), open(feature_names_filename, 'wb'))
print(f"✓ Feature names saved as '{feature_names_filename}'")

# ============================================================================
# STEP 14: Test Saved Model
# ============================================================================
print("\n" + "="*60)
print("STEP 14: Testing Saved Model...")

loaded_model = pickle.load(open(model_filename, 'rb'))
loaded_scaler = pickle.load(open(scaler_filename, 'rb'))

# Test with a sample
sample_idx = 0
sample_input = X_test.iloc[sample_idx:sample_idx+1]
actual_output = y_test.iloc[sample_idx]
predicted_output = loaded_model.predict(sample_input)[0]

print("\nSAMPLE PREDICTION TEST:")
print(f"  Actual Power Output: {actual_output:.2f}")
print(f"  Predicted Power Output: {predicted_output:.2f}")
print(f"  Prediction Error: {abs(actual_output - predicted_output):.2f}")
print(f"  Accuracy: {(1 - abs(actual_output - predicted_output) / actual_output) * 100:.2f}%")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*60)
print("✅ MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("="*60)
print(f"Dataset: {df.shape[0]} samples, {df.shape[1]} features")
print(f"Best Model: {best_model_name}")
print(f"Test R² Score: {best_r2:.4f}")
print(f"\nModel files saved in Flask/ directory:")
print(f"  - {model_filename}")
print(f"  - {scaler_filename}")
print(f"  - {feature_names_filename}")
print("\n🚀 Ready to run Flask application!")
print("   Run: cd Flask && python windApp.py")
print("="*60)
