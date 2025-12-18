"""
Australian Bushfire ML Analysis - Model Evaluation Script
Author: Vinh
Course: MGSC 7310 - Tulane University
Description: Evaluate trained models and create visualizations
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error,
                            roc_auc_score, roc_curve, f1_score, precision_score, 
                            recall_score, confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Configuration
DATA_PATH = 'data/processed/fires_for_xgboost.csv'
MODELS_DIR = 'results/models'
FIGURES_DIR = 'results/figures'

# Create figures directory
os.makedirs(FIGURES_DIR, exist_ok=True)

print("=" * 70)
print("AUSTRALIAN BUSHFIRE ML ANALYSIS - MODEL EVALUATION")
print("=" * 70)
print()

# ============================================================================
# PART 1: LOAD DATA AND MODELS
# ============================================================================

print("Loading data and models...")

# Load data
df = pd.read_csv(DATA_PATH)

# Encode categorical variables
label_encoders = {}
for col in ['source', 'satellite', 'daynight']:
    if col in df.columns:
        le = LabelEncoder()
        df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

# Define features
feature_cols = [
    'max_temp', 'min_temp', 'temp_range', 'avg_temp',
    'rainfall_mm', 'days_since_rain', 'rain_sum_7day', 'rain_sum_30day',
    'extreme_heat', 'dry_day',
    'brightness', 'scan', 'track', 'confidence_numeric',
    'population_density', 'in_populated_area', 'near_urban',
    'month_num', 'week', 'acq_time',
    'latitude', 'longitude',
    'source_encoded', 'satellite_encoded', 'daynight_encoded'
]

# Prepare data
X = df[feature_cols]
y_reg = df['frp']
y_class = df['high_risk_event'].astype(int)

# Split data (same as training)
X_train, X_test, y_train_reg, y_test_reg = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_class, test_size=0.2, random_state=42, stratify=y_class
)

# Load models
model_reg = xgb.XGBRegressor()
model_reg.load_model(f"{MODELS_DIR}/xgboost_frp_regression.json")

model_clf = xgb.XGBClassifier()
model_clf.load_model(f"{MODELS_DIR}/xgboost_highrisk_classification.json")

print("✓ Data and models loaded")
print()

# ============================================================================
# PART 2: REGRESSION MODEL VISUALIZATIONS
# ============================================================================

print("Creating regression visualizations...")

# Predictions
y_pred_reg = model_reg.predict(X_test)

# Figure 1: Actual vs Predicted
plt.figure(figsize=(10, 8))
plt.scatter(y_test_reg, y_pred_reg, alpha=0.3, s=20)
plt.plot([y_test_reg.min(), y_test_reg.max()], 
         [y_test_reg.min(), y_test_reg.max()], 
         'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual FRP (MW)', fontsize=12, fontweight='bold')
plt.ylabel('Predicted FRP (MW)', fontsize=12, fontweight='bold')
plt.title('Fire Intensity Prediction: Actual vs Predicted', 
          fontsize=14, fontweight='bold', pad=20)

# Add R² text
r2 = r2_score(y_test_reg, y_pred_reg)
plt.text(0.05, 0.95, f'R² = {r2:.4f}', 
         transform=plt.gca().transAxes,
         fontsize=14, fontweight='bold',
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/xgb_actual_vs_predicted.png", dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: xgb_actual_vs_predicted.png")

# Figure 2: Regression Feature Importance
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': model_reg.feature_importances_
}).sort_values('importance', ascending=False).head(15)

plt.figure(figsize=(10, 8))
plt.barh(range(len(importance_df)), importance_df['importance'])
plt.yticks(range(len(importance_df)), importance_df['feature'])
plt.xlabel('Importance', fontsize=12, fontweight='bold')
plt.title('Top 15 Features - Fire Intensity Prediction', 
          fontsize=14, fontweight='bold', pad=20)
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/xgb_feature_importance_regression.png", dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: xgb_feature_importance_regression.png")

# ============================================================================
# PART 3: CLASSIFICATION MODEL VISUALIZATIONS
# ============================================================================

print("Creating classification visualizations...")

# Predictions
y_pred_c = model_clf.predict(X_test_c)
y_pred_proba = model_clf.predict_proba(X_test_c)[:, 1]

# Figure 3: Confusion Matrix
cm = confusion_matrix(y_test_c, y_pred_c)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'High-Risk'],
            yticklabels=['Normal', 'High-Risk'],
            cbar_kws={'label': 'Count'})
plt.ylabel('Actual', fontsize=12, fontweight='bold')
plt.xlabel('Predicted', fontsize=12, fontweight='bold')
plt.title('Confusion Matrix - High-Risk Classification', 
          fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/xgb_confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: xgb_confusion_matrix.png")

# Figure 4: ROC Curve
fpr, tpr, thresholds = roc_curve(y_test_c, y_pred_proba)
roc_auc = roc_auc_score(y_test_c, y_pred_proba)

plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
plt.ylabel('True Positive Rate (Recall)', fontsize=12, fontweight='bold')
plt.title('ROC Curve - High-Risk Fire Classification', 
          fontsize=14, fontweight='bold', pad=20)
plt.legend(loc="lower right", fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/xgb_roc_curve.png", dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: xgb_roc_curve.png")

# Figure 5: Classification Feature Importance
importance_clf_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': model_clf.feature_importances_
}).sort_values('importance', ascending=False).head(15)

plt.figure(figsize=(10, 8))
plt.barh(range(len(importance_clf_df)), importance_clf_df['importance'], color='coral')
plt.yticks(range(len(importance_clf_df)), importance_clf_df['feature'])
plt.xlabel('Importance', fontsize=12, fontweight='bold')
plt.title('Top 15 Features - High-Risk Classification', 
          fontsize=14, fontweight='bold', pad=20)
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/xgb_feature_importance_classification.png", dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ Saved: xgb_feature_importance_classification.png")

# ============================================================================
# PART 4: DETAILED METRICS REPORT
# ============================================================================

print()
print("=" * 70)
print("DETAILED EVALUATION RESULTS")
print("=" * 70)
print()

# Regression metrics
rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
mae = mean_absolute_error(y_test_reg, y_pred_reg)

print("MODEL 1: FIRE INTENSITY REGRESSION")
print(f"  R² Score: {r2:.4f} ({r2*100:.2f}% variance explained)")
print(f"  RMSE: {rmse:.2f} MW")
print(f"  MAE: {mae:.2f} MW")
print()
print("  Top 5 Predictive Features:")
for i, row in importance_df.head(5).iterrows():
    print(f"    {i+1}. {row['feature']:20s}: {row['importance']*100:.1f}%")
print()

# Classification metrics
accuracy = (y_test_c == y_pred_c).mean()
precision = precision_score(y_test_c, y_pred_c)
recall = recall_score(y_test_c, y_pred_c)
f1 = f1_score(y_test_c, y_pred_c)

tn, fp, fn, tp = cm.ravel()

print("MODEL 2: HIGH-RISK CLASSIFICATION")
print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"  Recall: {recall:.4f} ({recall*100:.2f}%)")
print(f"  F1 Score: {f1:.4f} ({f1*100:.2f}%)")
print(f"  ROC-AUC: {roc_auc:.4f}")
print()
print("  Confusion Matrix Breakdown:")
print(f"    True Negatives:  {tn:,} (correct normal predictions)")
print(f"    False Positives: {fp:,} (false alarms)")
print(f"    False Negatives: {fn:,} (missed dangerous fires) ⚠️")
print(f"    True Positives:  {tp:,} (correct high-risk predictions)")
print()
print(f"  Performance: Caught {tp} out of {tp+fn} high-risk fires ({recall*100:.1f}%)")
print()
print("  Top 5 Predictive Features:")
for i, row in importance_clf_df.head(5).iterrows():
    print(f"    {i+1}. {row['feature']:20s}: {row['importance']*100:.1f}%")
print()

# ============================================================================
# PART 5: CLASSIFICATION REPORT
# ============================================================================

print("DETAILED CLASSIFICATION REPORT:")
print(classification_report(y_test_c, y_pred_c, 
                           target_names=['Normal', 'High-Risk'],
                           digits=4))

print("=" * 70)
print("✓ EVALUATION COMPLETE")
print("=" * 70)
print()
print(f"All visualizations saved to: {FIGURES_DIR}/")
print()
print("Visualizations created:")
print("  1. xgb_actual_vs_predicted.png - Regression accuracy")
print("  2. xgb_feature_importance_regression.png - Intensity predictors")
print("  3. xgb_confusion_matrix.png - Classification performance")
print("  4. xgb_roc_curve.png - ROC analysis")
print("  5. xgb_feature_importance_classification.png - Risk predictors")
