"""
Unit Tests for Australian Bushfire ML Analysis - Python Module
Author: Vinh
Course: MGSC 7310 - Tulane University

Tests for model loading, predictions, and data validation.
Run with: pytest test_predictions.py
"""

import unittest
import pandas as pd
import numpy as np
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'python'))

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class TestModelLoading(unittest.TestCase):
    """Test model file loading and basic properties"""
    
    @unittest.skipUnless(XGBOOST_AVAILABLE, "XGBoost not installed")
    def test_regression_model_exists(self):
        """Test that regression model file exists"""
        model_path = 'results/models/xgboost_frp_regression.json'
        self.assertTrue(
            os.path.exists(model_path),
            f"Regression model file not found at {model_path}"
        )
    
    @unittest.skipUnless(XGBOOST_AVAILABLE, "XGBoost not installed")
    def test_classification_model_exists(self):
        """Test that classification model file exists"""
        model_path = 'results/models/xgboost_highrisk_classification.json'
        self.assertTrue(
            os.path.exists(model_path),
            f"Classification model file not found at {model_path}"
        )
    
    @unittest.skipUnless(XGBOOST_AVAILABLE, "XGBoost not installed")
    def test_load_regression_model(self):
        """Test loading regression model"""
        model_path = 'results/models/xgboost_frp_regression.json'
        if os.path.exists(model_path):
            model = xgb.XGBRegressor()
            model.load_model(model_path)
            self.assertEqual(model.n_estimators, 100)
            self.assertEqual(model.max_depth, 6)
    
    @unittest.skipUnless(XGBOOST_AVAILABLE, "XGBoost not installed")
    def test_load_classification_model(self):
        """Test loading classification model"""
        model_path = 'results/models/xgboost_highrisk_classification.json'
        if os.path.exists(model_path):
            model = xgb.XGBClassifier()
            model.load_model(model_path)
            self.assertEqual(model.n_estimators, 100)
            self.assertEqual(model.max_depth, 6)


class TestDataValidation(unittest.TestCase):
    """Test data validation and feature requirements"""
    
    def test_required_features(self):
        """Test that all required features are defined"""
        required_features = [
            'max_temp', 'min_temp', 'temp_range', 'avg_temp',
            'rainfall_mm', 'days_since_rain', 'rain_sum_7day', 'rain_sum_30day',
            'extreme_heat', 'dry_day',
            'brightness', 'scan', 'track', 'confidence_numeric',
            'population_density', 'in_populated_area', 'near_urban',
            'month_num', 'week', 'acq_time',
            'latitude', 'longitude',
            'source_encoded', 'satellite_encoded', 'daynight_encoded'
        ]
        self.assertEqual(len(required_features), 25)
    
    def test_feature_data_types(self):
        """Test that sample data has correct types"""
        sample_data = {
            'max_temp': 38.5,
            'min_temp': 22.0,
            'brightness': 340.5,
            'population_density': 180.5,
            'extreme_heat': 0,
            'in_populated_area': 1
        }
        
        # Test numeric features
        self.assertIsInstance(sample_data['max_temp'], (int, float))
        self.assertIsInstance(sample_data['brightness'], (int, float))
        
        # Test binary features
        self.assertIn(sample_data['extreme_heat'], [0, 1])
        self.assertIn(sample_data['in_populated_area'], [0, 1])
    
    def test_feature_ranges(self):
        """Test that features are within reasonable ranges"""
        sample_data = {
            'max_temp': 38.5,
            'min_temp': 22.0,
            'brightness': 340.5,
            'population_density': 180.5,
            'latitude': -33.87,
            'longitude': 151.21,
            'confidence_numeric': 85
        }
        
        # Temperature ranges (Australia)
        self.assertGreaterEqual(sample_data['max_temp'], -10)
        self.assertLessEqual(sample_data['max_temp'], 60)
        
        # Latitude range (Australia)
        self.assertGreaterEqual(sample_data['latitude'], -45)
        self.assertLessEqual(sample_data['latitude'], -10)
        
        # Longitude range (Australia)
        self.assertGreaterEqual(sample_data['longitude'], 110)
        self.assertLessEqual(sample_data['longitude'], 160)
        
        # Confidence range
        self.assertGreaterEqual(sample_data['confidence_numeric'], 0)
        self.assertLessEqual(sample_data['confidence_numeric'], 100)


class TestPredictions(unittest.TestCase):
    """Test prediction functionality"""
    
    @unittest.skipUnless(XGBOOST_AVAILABLE, "XGBoost not installed")
    def test_regression_prediction_shape(self):
        """Test regression prediction returns correct shape"""
        model_path = 'results/models/xgboost_frp_regression.json'
        
        if os.path.exists(model_path):
            model = xgb.XGBRegressor()
            model.load_model(model_path)
            
            # Create sample input
            X_sample = np.random.rand(5, 25)  # 5 samples, 25 features
            predictions = model.predict(X_sample)
            
            self.assertEqual(predictions.shape, (5,))
            self.assertTrue(all(isinstance(x, (np.floating, float)) for x in predictions))
    
    @unittest.skipUnless(XGBOOST_AVAILABLE, "XGBoost not installed")
    def test_classification_prediction_shape(self):
        """Test classification prediction returns correct shape"""
        model_path = 'results/models/xgboost_highrisk_classification.json'
        
        if os.path.exists(model_path):
            model = xgb.XGBClassifier()
            model.load_model(model_path)
            
            # Create sample input
            X_sample = np.random.rand(5, 25)  # 5 samples, 25 features
            predictions = model.predict(X_sample)
            probabilities = model.predict_proba(X_sample)
            
            self.assertEqual(predictions.shape, (5,))
            self.assertEqual(probabilities.shape, (5, 2))
            self.assertTrue(all(x in [0, 1] for x in predictions))
    
    @unittest.skipUnless(XGBOOST_AVAILABLE, "XGBoost not installed")
    def test_prediction_output_ranges(self):
        """Test that predictions are within reasonable ranges"""
        model_path = 'results/models/xgboost_frp_regression.json'
        
        if os.path.exists(model_path):
            model = xgb.XGBRegressor()
            model.load_model(model_path)
            
            # Create realistic sample input
            X_sample = np.random.rand(10, 25) * 100  # Scale to reasonable range
            predictions = model.predict(X_sample)
            
            # FRP should be positive
            self.assertTrue(all(x >= 0 for x in predictions))
            
            # FRP should be reasonable (not astronomical)
            self.assertTrue(all(x < 10000 for x in predictions))


class TestPredictFunction(unittest.TestCase):
    """Test the predict_fire function from predict.py"""
    
    def test_predict_function_exists(self):
        """Test that predict function can be imported"""
        try:
            from predict import predict_fire
            self.assertTrue(callable(predict_fire))
        except ImportError:
            self.skipTest("predict.py not in path")
    
    def test_predict_function_with_valid_input(self):
        """Test predict_fire with valid input data"""
        try:
            from predict import predict_fire
            
            valid_fire = {
                'max_temp': 38.5, 'min_temp': 22.0, 'temp_range': 16.5,
                'avg_temp': 30.25, 'rainfall_mm': 0.0, 'days_since_rain': 35,
                'rain_sum_7day': 0.0, 'rain_sum_30day': 2.5,
                'extreme_heat': 0, 'dry_day': 1,
                'brightness': 340.5, 'scan': 2.8, 'track': 1.5,
                'confidence_numeric': 85, 'population_density': 180.5,
                'in_populated_area': 1, 'near_urban': 1,
                'month_num': 12, 'week': 50, 'acq_time': 1430,
                'latitude': -33.87, 'longitude': 151.21,
                'source_encoded': 1, 'satellite_encoded': 1,
                'daynight_encoded': 0
            }
            
            if os.path.exists('results/models/xgboost_frp_regression.json'):
                result = predict_fire(valid_fire)
                
                # Check result structure
                self.assertIn('predicted_frp_mw', result)
                self.assertIn('intensity_level', result)
                self.assertIn('is_high_risk', result)
                self.assertIn('risk_probability', result)
                
                # Check types
                self.assertIsInstance(result['predicted_frp_mw'], (int, float))
                self.assertIsInstance(result['is_high_risk'], bool)
                self.assertIsInstance(result['risk_probability'], (int, float))
                
                # Check ranges
                self.assertGreaterEqual(result['predicted_frp_mw'], 0)
                self.assertGreaterEqual(result['risk_probability'], 0)
                self.assertLessEqual(result['risk_probability'], 100)
            else:
                self.skipTest("Models not available")
                
        except ImportError:
            self.skipTest("predict.py not available")


class TestDataFiles(unittest.TestCase):
    """Test that required data files exist"""
    
    def test_processed_data_directory(self):
        """Test that processed data directory exists"""
        self.assertTrue(
            os.path.exists('data/processed'),
            "Processed data directory not found"
        )
    
    def test_results_directories(self):
        """Test that results directories exist"""
        self.assertTrue(os.path.exists('results'))
        self.assertTrue(os.path.exists('results/models'))
        self.assertTrue(os.path.exists('results/figures'))


def run_tests():
    """Run all tests and print summary"""
    print("="*70)
    print("RUNNING UNIT TESTS - Australian Bushfire ML Analysis")
    print("="*70)
    print()
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestModelLoading))
    suite.addTests(loader.loadTestsFromTestCase(TestDataValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestPredictions))
    suite.addTests(loader.loadTestsFromTestCase(TestPredictFunction))
    suite.addTests(loader.loadTestsFromTestCase(TestDataFiles))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print()
    print("="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("="*70)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
