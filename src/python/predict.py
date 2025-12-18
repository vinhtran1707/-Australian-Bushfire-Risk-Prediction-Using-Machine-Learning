"""
Australian Bushfire ML Analysis - Prediction Script
Author: Vinh
Course: MGSC 7310 - Tulane University
Description: Make predictions on new fire detections using trained models
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import os

# Configuration
MODELS_DIR = 'results/models'

# Load models
model_reg = xgb.XGBRegressor()
model_reg.load_model(f"{MODELS_DIR}/xgboost_frp_regression.json")

model_clf = xgb.XGBClassifier()
model_clf.load_model(f"{MODELS_DIR}/xgboost_highrisk_classification.json")

print("=" * 70)
print("AUSTRALIAN BUSHFIRE ML ANALYSIS - PREDICTION SYSTEM")
print("=" * 70)
print()
print("✓ Models loaded successfully")
print()

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def predict_fire(fire_data):
    """
    Predict fire intensity and risk level for a new fire detection
    
    Parameters:
    -----------
    fire_data : dict
        Dictionary containing fire attributes with keys matching feature names
        
    Returns:
    --------
    dict : Predictions including intensity (MW) and risk classification
    """
    
    # Define required features
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
    
    # Create DataFrame
    fire_df = pd.DataFrame([fire_data])
    
    # Ensure all features are present
    missing_features = set(feature_cols) - set(fire_df.columns)
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Select features in correct order
    X = fire_df[feature_cols]
    
    # Predict intensity
    predicted_frp = model_reg.predict(X)[0]
    
    # Predict risk
    predicted_risk = model_clf.predict(X)[0]
    predicted_risk_proba = model_clf.predict_proba(X)[0, 1]
    
    # Classify intensity level
    if predicted_frp < 50:
        intensity_level = "LOW"
    elif predicted_frp < 100:
        intensity_level = "MODERATE"
    elif predicted_frp < 500:
        intensity_level = "HIGH"
    else:
        intensity_level = "EXTREME"
    
    # Format results
    results = {
        'predicted_frp_mw': round(predicted_frp, 2),
        'intensity_level': intensity_level,
        'is_high_risk': bool(predicted_risk),
        'risk_probability': round(predicted_risk_proba * 100, 2),
        'recommendation': get_recommendation(predicted_frp, predicted_risk, 
                                            fire_data.get('population_density', 0))
    }
    
    return results


def get_recommendation(frp, is_high_risk, population_density):
    """Generate emergency response recommendation"""
    
    if is_high_risk:
        return {
            'priority': 'URGENT',
            'actions': [
                'Dispatch resources immediately',
                f'Alert residents in affected area (~{int(population_density)} people/km²)',
                'Prepare evacuation routes',
                'Establish incident command'
            ]
        }
    elif frp > 100:
        return {
            'priority': 'HIGH',
            'actions': [
                'Monitor closely',
                'Pre-position resources',
                'Issue fire warnings to nearby areas'
            ]
        }
    elif frp > 50:
        return {
            'priority': 'MODERATE',
            'actions': [
                'Standard monitoring protocol',
                'Track fire progression',
                'Alert nearby fire stations'
            ]
        }
    else:
        return {
            'priority': 'LOW',
            'actions': [
                'Routine satellite monitoring',
                'Log in fire database'
            ]
        }

# ============================================================================
# EXAMPLE PREDICTIONS
# ============================================================================

def run_example_predictions():
    """Run predictions on example scenarios"""
    
    print("=" * 70)
    print("EXAMPLE PREDICTIONS")
    print("=" * 70)
    print()
    
    # Example 1: Urban High-Risk Fire
    print("SCENARIO 1: Urban Fire During Drought")
    print("-" * 70)
    
    fire1 = {
        'max_temp': 42.0,
        'min_temp': 25.0,
        'temp_range': 17.0,
        'avg_temp': 33.5,
        'rainfall_mm': 0.0,
        'days_since_rain': 45,
        'rain_sum_7day': 0.0,
        'rain_sum_30day': 0.0,
        'extreme_heat': 1,
        'dry_day': 1,
        'brightness': 355.0,
        'scan': 1.5,
        'track': 1.2,
        'confidence_numeric': 92,
        'population_density': 850.0,
        'in_populated_area': 1,
        'near_urban': 1,
        'month_num': 1,
        'week': 2,
        'acq_time': 1500,
        'latitude': -35.3,
        'longitude': 149.2,
        'source_encoded': 1,
        'satellite_encoded': 1,
        'daynight_encoded': 0
    }
    
    result1 = predict_fire(fire1)
    print_prediction_result(fire1, result1)
    print()
    
    # Example 2: Remote Low-Intensity Fire
    print("SCENARIO 2: Remote Low-Intensity Fire")
    print("-" * 70)
    
    fire2 = {
        'max_temp': 28.0,
        'min_temp': 15.0,
        'temp_range': 13.0,
        'avg_temp': 21.5,
        'rainfall_mm': 0.0,
        'days_since_rain': 10,
        'rain_sum_7day': 5.0,
        'rain_sum_30day': 15.0,
        'extreme_heat': 0,
        'dry_day': 1,
        'brightness': 315.0,
        'scan': 3.2,
        'track': 1.8,
        'confidence_numeric': 65,
        'population_density': 0.5,
        'in_populated_area': 0,
        'near_urban': 0,
        'month_num': 9,
        'week': 37,
        'acq_time': 1100,
        'latitude': -30.5,
        'longitude': 145.2,
        'source_encoded': 0,
        'satellite_encoded': 0,
        'daynight_encoded': 0
    }
    
    result2 = predict_fire(fire2)
    print_prediction_result(fire2, result2)
    print()
    
    # Example 3: Suburban Moderate Fire
    print("SCENARIO 3: Suburban Fire")
    print("-" * 70)
    
    fire3 = {
        'max_temp': 35.0,
        'min_temp': 20.0,
        'temp_range': 15.0,
        'avg_temp': 27.5,
        'rainfall_mm': 0.0,
        'days_since_rain': 20,
        'rain_sum_7day': 0.0,
        'rain_sum_30day': 8.0,
        'extreme_heat': 0,
        'dry_day': 1,
        'brightness': 330.0,
        'scan': 2.5,
        'track': 1.6,
        'confidence_numeric': 78,
        'population_density': 120.0,
        'in_populated_area': 1,
        'near_urban': 1,
        'month_num': 11,
        'week': 46,
        'acq_time': 1400,
        'latitude': -33.9,
        'longitude': 151.1,
        'source_encoded': 1,
        'satellite_encoded': 0,
        'daynight_encoded': 0
    }
    
    result3 = predict_fire(fire3)
    print_prediction_result(fire3, result3)
    print()


def print_prediction_result(fire_data, result):
    """Print formatted prediction results"""
    
    print("INPUT:")
    print(f"  Location: ({fire_data['latitude']:.2f}, {fire_data['longitude']:.2f})")
    print(f"  Temperature: {fire_data['max_temp']:.1f}°C")
    print(f"  Drought: {fire_data['days_since_rain']} days without rain")
    print(f"  Population: {fire_data['population_density']:.1f} people/km²")
    print(f"  Brightness: {fire_data['brightness']:.1f} K")
    print()
    
    print("PREDICTIONS:")
    print(f"  Predicted Intensity: {result['predicted_frp_mw']:.1f} MW ({result['intensity_level']})")
    print(f"  Risk Classification: {'HIGH-RISK ⚠️' if result['is_high_risk'] else 'Normal ✓'}")
    print(f"  Risk Probability: {result['risk_probability']:.1f}%")
    print()
    
    print("RECOMMENDATION:")
    print(f"  Priority: {result['recommendation']['priority']}")
    print("  Actions:")
    for action in result['recommendation']['actions']:
        print(f"    • {action}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Run example predictions
    run_example_predictions()
    
    print()
    print("=" * 70)
    print("✓ PREDICTION EXAMPLES COMPLETE")
    print("=" * 70)
    print()
    print("To make your own predictions:")
    print("  1. Import this module: from predict import predict_fire")
    print("  2. Create a fire_data dictionary with required features")
    print("  3. Call: result = predict_fire(fire_data)")
    print()
