import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from datetime import datetime, timedelta

# Import the function to test
from Models.ModelTraining import CatBoostBinaryClassifierTraining
from catboost import CatBoostClassifier

class TestModelTraining(unittest.TestCase):
    
    def setUp(self):
        """Create synthetic data for testing"""
        # Create date range for time series data
        dates = pd.date_range(start='2022-01-01', periods=1000, freq='H')
        
        # Create synthetic features
        np.random.seed(42)
        self.X = pd.DataFrame({
            'Close': np.random.randn(1000).cumsum() + 100,  # Price-like feature
            'Volume': np.abs(np.random.randn(1000)) * 1000,  # Volume-like feature
            'Feature1': np.random.randn(1000),
            'Feature2': np.random.randn(1000),
            'Feature3': np.random.randn(1000)
        }, index=dates)
        
        # Create target variable (binary classification: 0 or 1)
        # For testing purposes, we'll make it somewhat predictable from features
        signal = (self.X['Feature1'] > 0) & (self.X['Feature2'] > 0)
        noise = np.random.random(1000) > 0.8  # Add some noise
        self.y = pd.Series((signal ^ noise).astype(int), index=dates)
        
        # Create temporary directory for model outputs
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up temporary files"""
        shutil.rmtree(self.temp_dir)
    
    def test_standard_cross_validation(self):
        """Test the function with standard TimeSeriesSplit cross-validation"""
        model_params = {
            'iterations': 100,  # Reduced for faster testing
            'learning_rate': 0.1,
            'verbose': False
        }
        
        tscv_params = {
            'n_splits': 3,
            'test_size': 100
        }
        
        models, metrics, feature_importances = CatBoostBinaryClassifierTraining(
            X=self.X,
            y=self.y,
            model_name="test_model",
            model_params=model_params,
            tscv_params=tscv_params,
            walk_forward_CV=False,
            output_dir=self.temp_dir,
            experiment_name="test_run"
        )
        
        # Assertions to verify the function works as expected
        self.assertIsInstance(models, list)
        self.assertTrue(len(models) > 0)
        self.assertIsInstance(models[0], CatBoostClassifier)
        self.assertIsInstance( metrics, pd.DataFrame)
        self.assertIsInstance( feature_importances, list )
        self.assertEqual( feature_importances[0].shape[0], self.X.shape[1] )
        
        # Check if metrics dataframe has expected columns
        expected_columns = [
            "fold",
            "train_start",
            "train_end",
            "test_start",
            "test_end",
            "Positive Cases",
            "Predicted Positives",
            "Precision",
            "Recall",
        ]
        for col in expected_columns:
            self.assertIn(col, metrics.columns)
    
        
if __name__ == '__main__':
    unittest.main()