"""
Australian Bushfire ML Analysis - Model Training Script
Author: Vinh
Course: MGSC 7310 - Tulane University
Description: Train XGBoost models for fire intensity prediction and risk classification
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
DATA_PATH = 'data/processed/fires_for_xgboost.csv'
MODELS_DIR = 'results/models'
FIGURES_DIR = 'results/figures'
METRICS_DIR = 'results/metrics'

# Create directories if they don't exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

print("=" * 70)
print("AUSTRALIAN BUSHFIRE ML ANALYSIS - MODEL TRAINING")
print("=" * 70)
print()

# ============================================================================
# PART 1: LOAD AND PREPARE DATA
# ============================================================================

print("Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"✓ Loaded {len(df):,} fire observations with {df.shape[1]} features")
print()

# Display basic statistics
print("Dataset Summary:")
print(f"  Date range: {df['month_num'].min()} to {df['month_num'].max()}")
print(f"  FRP range: {df['frp'].min():.1f} to {df['frp'].max():.1f} MW")
print(f"  High-risk events: {df['high_risk_event'].sum():,} ({100*df['high_risk_event'].mean():.2f}%)")
print()

# ============================================================================
# PART 2: ENCODE CATEGORICAL VARIABLES
# ============================================================================

print("Encoding categorical variables...")

# Create label encoders for categorical features
label_encoders = {}
categorical_cols = ['source', 'satellite', 'daynight']

for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        print(f"  ✓ Encoded {col}: {len(le.classes_)} unique values")

print()

# ============================================================================
# PART 3: DEFINE FEATURES
# ============================================================================

# Define feature columns (25 features total)
feature_cols = [
    # Weather features (10)
    'max_temp', 'min_temp', 'temp_range', 'avg_temp',
    'rainfall_mm', 'days_since_rain', 'rain_sum_7day', 'rain_sum_30day',
    'extreme_heat', 'dry_day',
    
    # Fire characteristics (4)
    'brightness', 'scan', 'track', 'confidence_numeric',
    
    # Population features (3)
    'population_density', 'in_populated_area', 'near_urban',
    
    # Temporal features (5)
    'month_num', 'week', 'acq_time',
    
    # Spatial features (2)
    'latitude', 'longitude',
    
    # Source information (3)
    'source_encoded', 'satellite_encoded', 'daynight_encoded'
]

# Verify all features exist
missing_features = [f for f in feature_cols if f not in df.columns]
if missing_features:
    print(f"WARNING: Missing features: {missing_features}")
    feature_cols = [f for f in feature_cols if f in df.columns]

print(f"Using {len(feature_cols)} features for modeling")
print()

# ============================================================================
# PART 4: PREPARE DATA FOR REGRESSION (FIRE INTENSITY PREDICTION)
# ============================================================================

print("=" * 70)
print("MODEL 1: FIRE INTENSITY REGRESSION")
print("=" * 70)
print()

# Prepare features and target
X = df[feature_cols]
y = df['frp']  # Fire Radiative Power (MW)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42
)

print(f"Training set: {len(X_train):,} samples")
print(f"Test set: {len(X_test):,} samples")
print()

# ============================================================================
# PART 5: TRAIN XGBOOST REGRESSOR
# ============================================================================

print("Training XGBoost Regressor...")

# Create and train model
model_reg = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

model_reg.fit(X_train, y_train)
print("✓ Model training complete")
print()

# ============================================================================
# PART 6: EVALUATE REGRESSION MODEL
# ============================================================================

print("Evaluating regression model...")

# Make predictions
y_pred = model_reg.predict(X_test)

# Calculate metrics
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print("REGRESSION RESULTS:")
print(f"  R² Score: {r2:.4f} ({r2*100:.2f}% variance explained)")
print(f"  RMSE: {rmse:.2f} MW")
print(f"  MAE: {mae:.2f} MW")
print()

# Get feature importance
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': model_reg.feature_importances_
}).sort_values('importance', ascending=False)

print("TOP 10 IMPORTANT FEATURES:")
for i, row in importance_df.head(10).iterrows():
    print(f"  {row['feature']:25s}: {row['importance']:.4f} ({row['importance']*100:.1f}%)")
print()

# Save regression metrics
reg_metrics = pd.DataFrame({
    'metric': ['R2', 'RMSE', 'MAE'],
    'value': [r2, rmse, mae]
})
reg_metrics.to_csv(f"{METRICS_DIR}/regression_metrics.csv", index=False)
print(f"✓ Saved regression metrics to {METRICS_DIR}/regression_metrics.csv")

# Save feature importance
importance_df.to_csv(f"{METRICS_DIR}/feature_importance_regression.csv", index=False)
print(f"✓ Saved feature importance to {METRICS_DIR}/feature_importance_regression.csv")
print()

# Save regression model
model_reg.save_model(f"{MODELS_DIR}/xgboost_frp_regression.json")
print(f"✓ Saved model to {MODELS_DIR}/xgboost_frp_regression.json")
print()

# ============================================================================
# PART 7: PREPARE DATA FOR CLASSIFICATION (HIGH-RISK IDENTIFICATION)
# ============================================================================

print("=" * 70)
print("MODEL 2: HIGH-RISK EVENT CLASSIFICATION")
print("=" * 70)
print()

# Prepare classification target
y_class = df['high_risk_event'].astype(int)

# Check class distribution
class_counts = y_class.value_counts()
print("Class Distribution:")
print(f"  Normal fires: {class_counts[0]:,} ({100*class_counts[0]/len(y_class):.2f}%)")
print(f"  High-risk fires: {class_counts[1]:,} ({100*class_counts[1]/len(y_class):.2f}%)")
print(f"  Imbalance ratio: {class_counts[0]/class_counts[1]:.1f}:1")
print()

# Calculate scale_pos_weight for imbalanced data
scale_pos_weight = class_counts[0] / class_counts[1]
print(f"Using scale_pos_weight: {scale_pos_weight:.1f}")
print()

# Train-test split with stratification
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_class,
    test_size=0.2,
    random_state=42,
    stratify=y_class  # Maintain class distribution
)

print(f"Training set: {len(X_train_c):,} samples")
print(f"  - Normal: {(y_train_c == 0).sum():,}")
print(f"  - High-risk: {(y_train_c == 1).sum():,}")
print(f"Test set: {len(X_test_c):,} samples")
print(f"  - Normal: {(y_test_c == 0).sum():,}")
print(f"  - High-risk: {(y_test_c == 1).sum():,}")
print()

# ============================================================================
# PART 8: TRAIN XGBOOST CLASSIFIER
# ============================================================================

print("Training XGBoost Classifier...")

# Create and train classifier with class weighting
model_clf = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,  # Handle class imbalance
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
)

model_clf.fit(X_train_c, y_train_c)
print("✓ Model training complete")
print()

# ============================================================================
# PART 9: EVALUATE CLASSIFICATION MODEL
# ============================================================================

print("Evaluating classification model...")

# Make predictions
y_pred_c = model_clf.predict(X_test_c)
y_pred_proba = model_clf.predict_proba(X_test_c)[:, 1]

# Calculate metrics
from sklearn.metrics import confusion_matrix, accuracy_score

accuracy = accuracy_score(y_test_c, y_pred_c)
precision = precision_score(y_test_c, y_pred_c)
recall = recall_score(y_test_c, y_pred_c)
f1 = f1_score(y_test_c, y_pred_c)
roc_auc = roc_auc_score(y_test_c, y_pred_proba)

print("CLASSIFICATION RESULTS:")
print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"  Recall: {recall:.4f} ({recall*100:.2f}%)")
print(f"  F1 Score: {f1:.4f} ({f1*100:.2f}%)")
print(f"  ROC-AUC: {roc_auc:.4f}")
print()

# Confusion matrix
cm = confusion_matrix(y_test_c, y_pred_c)
tn, fp, fn, tp = cm.ravel()

print("CONFUSION MATRIX:")
print(f"                 Predicted")
print(f"             Normal  High-Risk")
print(f"  Normal     {tn:6,}  {fp:6,}")
print(f"  High-Risk  {fn:6,}  {tp:6,}")
print()
print("INTERPRETATION:")
print(f"  True Negatives:  {tn:,} (correctly identified normal)")
print(f"  False Positives: {fp:,} (false alarms)")
print(f"  False Negatives: {fn:,} (missed dangerous fires) ⚠️")
print(f"  True Positives:  {tp:,} (correctly identified high-risk)")
print()

# Get feature importance for classification
importance_clf_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': model_clf.feature_importances_
}).sort_values('importance', ascending=False)

print("TOP 10 IMPORTANT FEATURES (CLASSIFICATION):")
for i, row in importance_clf_df.head(10).iterrows():
    print(f"  {row['feature']:25s}: {row['importance']:.4f} ({row['importance']*100:.1f}%)")
print()

# Save classification metrics
clf_metrics = pd.DataFrame({
    'metric': ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC'],
    'value': [accuracy, precision, recall, f1, roc_auc]
})
clf_metrics.to_csv(f"{METRICS_DIR}/classification_metrics.csv", index=False)
print(f"✓ Saved classification metrics to {METRICS_DIR}/classification_metrics.csv")

# Save confusion matrix
cm_df = pd.DataFrame(cm, 
                     index=['Actual_Normal', 'Actual_HighRisk'],
                     columns=['Predicted_Normal', 'Predicted_HighRisk'])
cm_df.to_csv(f"{METRICS_DIR}/confusion_matrix.csv")
print(f"✓ Saved confusion matrix to {METRICS_DIR}/confusion_matrix.csv")

# Save feature importance
importance_clf_df.to_csv(f"{METRICS_DIR}/feature_importance_classification.csv", index=False)
print(f"✓ Saved feature importance to {METRICS_DIR}/feature_importance_classification.csv")
print()

# Save classification model
model_clf.save_model(f"{MODELS_DIR}/xgboost_highrisk_classification.json")
print(f"✓ Saved model to {MODELS_DIR}/xgboost_highrisk_classification.json")
print()

# ============================================================================
# PART 10: SUMMARY
# ============================================================================

print("=" * 70)
print("TRAINING COMPLETE - SUMMARY")
print("=" * 70)
print()

print("MODEL 1 - FIRE INTENSITY REGRESSION:")
print(f"  R² = {r2:.4f} ({r2*100:.1f}% variance explained)")
print(f"  RMSE = {rmse:.2f} MW")
print(f"  Top predictor: {importance_df.iloc[0]['feature']} ({importance_df.iloc[0]['importance']*100:.1f}%)")
print()

print("MODEL 2 - HIGH-RISK CLASSIFICATION:")
print(f"  F1 Score = {f1:.4f} ({f1*100:.1f}%)")
print(f"  Recall = {recall:.4f} (caught {tp}/{tp+fn} high-risk fires)")
print(f"  Precision = {precision:.4f} ({tp}/{tp+fp} alerts correct)")
print(f"  Top predictor: {importance_clf_df.iloc[0]['feature']} ({importance_clf_df.iloc[0]['importance']*100:.1f}%)")
print()

print("FILES SAVED:")
print(f"  Models: {MODELS_DIR}/")
print(f"  Metrics: {METRICS_DIR}/")
print()

print("=" * 70)
print("✓ ALL MODELS TRAINED AND SAVED SUCCESSFULLY!")
print("=" * 70)
print()
print("Next steps:")
print("  1. Run evaluate_model.py for detailed evaluation")
print("  2. Use predict.py for making new predictions")
