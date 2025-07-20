"""
Model training module for McDonald's financial prediction
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
import joblib
import os
from datetime import datetime
import json
import logging

# Add parent directory to path for imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from preprocessor.preprocess import McDonaldsDataProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class McDonaldsModelTrainer:
    """Model trainer for McDonald's financial prediction"""
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        
    def get_models(self) -> dict:
        """Get dictionary of models to train"""
        return {
            'linear_regression': LinearRegression(),
            'ridge_regression': Ridge(alpha=1.0),
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                random_state=42
            )
        }
    
    def train_single_model(self, model, X_train: np.ndarray, y_train: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray, model_name: str) -> dict:
        """Train a single model and return metrics"""
        logger.info(f"Training {model_name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        # Cross-validation score
        try:
            cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='r2')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
        except:
            cv_mean = 0.0
            cv_std = 0.0
        
        metrics = {
            'model_name': model_name,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'cv_r2_mean': cv_mean,
            'cv_r2_std': cv_std
        }
        
        logger.info(f"{model_name} - Test R2: {test_r2:.4f}, Test MSE: {test_mse:.4f}")
        
        return model, metrics
    
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Train all models and return results"""
        logger.info("Starting model training...")
        
        models = self.get_models()
        results = {}
        
        for model_name, model in models.items():
            try:
                # Train model
                trained_model, metrics = self.train_single_model(
                    model, X_train, y_train, X_test, y_test, model_name
                )
                
                self.models[model_name] = trained_model
                results[model_name] = metrics
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                continue
        
        # Find best model
        if results:
            best_model_name = max(results.keys(), key=lambda k: results[k]['test_r2'])
            self.best_model = self.models[best_model_name]
            self.best_model_name = best_model_name
            
            logger.info(f"Best model: {best_model_name} (R2: {results[best_model_name]['test_r2']:.4f})")
        
        return results
    
    def save_models(self, save_dir: str, save_all: bool = False):
        """Save trained models"""
        os.makedirs(save_dir, exist_ok=True)
        
        if save_all:
            # Save all models
            for model_name, model in self.models.items():
                model_path = os.path.join(save_dir, f"{model_name}.pkl")
                joblib.dump(model, model_path)
                logger.info(f"Saved {model_name} to {model_path}")
        
        # Always save best model
        if self.best_model is not None:
            best_model_path = os.path.join(save_dir, "best_model.pkl")
            joblib.dump(self.best_model, best_model_path)
            
            # Save model metadata
            metadata = {
                'best_model_name': self.best_model_name,
                'training_timestamp': datetime.now().isoformat(),
                'model_type': type(self.best_model).__name__
            }
            
            metadata_path = os.path.join(save_dir, "model_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Saved best model ({self.best_model_name}) to {best_model_path}")


def main():
    """Main training pipeline"""
    # Initialize processor and trainer
    processor = McDonaldsDataProcessor()
    trainer = McDonaldsModelTrainer()
    
    # Load and process data
    logger.info("Loading and processing data...")
    raw_df = processor.load_raw_data('../data/mc-donalds.csv')
    clean_df = processor.clean_and_restructure(raw_df)
    pivot_df = processor.create_pivot_features(clean_df)
    engineered_df = processor.engineer_features(pivot_df)
    
    # Prepare for ML
    X, y = processor.prepare_ml_data(engineered_df)
    
    # Split data
    X_train, X_test, y_train, y_test = processor.split_data(X, y)
    
    # Scale features
    X_train_scaled, X_test_scaled = processor.scale_features(X_train, X_test)
    
    # Train models
    logger.info("Training models...")
    results = trainer.train_all_models(X_train_scaled, y_train, X_test_scaled, y_test)
    
    # Save models and preprocessor
    os.makedirs('../models', exist_ok=True)
    trainer.save_models('../models', save_all=True)
    processor.save_preprocessor('../models/preprocessor')
    
    # Save training results
    results_path = '../models/training_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    print(f"Best Model: {trainer.best_model_name}")
    print(f"Target Variable: {processor.target_name}")
    print(f"Number of Features: {len(processor.feature_names)}")
    print(f"Training Samples: {X_train_scaled.shape[0]}")
    print(f"Test Samples: {X_test_scaled.shape[0]}")
    
    print("\nModel Performance:")
    for model_name, metrics in results.items():
        print(f"{model_name:20s} - R2: {metrics['test_r2']:.4f}, MSE: {metrics['test_mse']:.4f}")
    
    print(f"\nModels saved to: ../models/")
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()