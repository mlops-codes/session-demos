import argparse
import os
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)

import mlflow
import mlflow.sklearn


def parse_args():
    parser = argparse.ArgumentParser(description="Train a classification model")
    
    # Data arguments
    parser.add_argument("--train_data", type=str, help="Path to training data")
    parser.add_argument("--test_data", type=str, help="Path to test data")
    
    # Model arguments
    parser.add_argument("--model_type", type=str, default="random_forest",
                       choices=["random_forest", "logistic_regression", "svm"],
                       help="Type of model to train")
    
    # Hyperparameters
    parser.add_argument("--n_estimators", type=int, default=100,
                       help="Number of estimators for Random Forest")
    parser.add_argument("--max_depth", type=int, default=10,
                       help="Maximum depth for Random Forest")
    parser.add_argument("--C", type=float, default=1.0,
                       help="Regularization parameter for Logistic Regression and SVM")
    parser.add_argument("--random_state", type=int, default=42,
                       help="Random state for reproducibility")
    
    # Output arguments
    parser.add_argument("--model_output", type=str, default="./outputs",
                       help="Directory to save the trained model")
    
    return parser.parse_args()


def load_data(train_path, test_path):
    """Load training and test data"""
    print(f"Loading training data from: {train_path}")
    train_df = pd.read_csv(train_path)
    
    print(f"Loading test data from: {test_path}")
    test_df = pd.read_csv(test_path)
    
    return train_df, test_df


def prepare_data(train_df, test_df):
    """Prepare features and targets"""
    # Separate features and target
    X_train = train_df.drop('target', axis=1)
    y_train = train_df['target']
    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def create_model(model_type, **kwargs):
    """Create model based on type and parameters"""
    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', 10),
            random_state=kwargs.get('random_state', 42)
        )
    elif model_type == "logistic_regression":
        return LogisticRegression(
            C=kwargs.get('C', 1.0),
            random_state=kwargs.get('random_state', 42),
            max_iter=1000
        )
    elif model_type == "svm":
        return SVC(
            C=kwargs.get('C', 1.0),
            random_state=kwargs.get('random_state', 42),
            probability=True
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Evaluate model performance"""
    # Cross-validation on training data
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    metrics = {
        'cv_accuracy_mean': cv_scores.mean(),
        'cv_accuracy_std': cv_scores.std(),
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'test_precision': precision_score(y_test, y_test_pred),
        'test_recall': recall_score(y_test, y_test_pred),
        'test_f1': f1_score(y_test, y_test_pred),
        'test_auc': roc_auc_score(y_test, y_test_proba)
    }
    
    return metrics, y_test_pred


def main():
    # Parse arguments
    args = parse_args()
    
    # Start MLflow run
    mlflow.start_run()
    
    try:
        print("Starting model training...")
        
        # Load data
        train_df, test_df = load_data(args.train_data, args.test_data)
        print(f"Loaded {len(train_df)} training samples and {len(test_df)} test samples")
        
        # Prepare data
        X_train, X_test, y_train, y_test, scaler = prepare_data(train_df, test_df)
        
        # Create model
        model_params = {
            'n_estimators': args.n_estimators,
            'max_depth': args.max_depth,
            'C': args.C,
            'random_state': args.random_state
        }
        
        model = create_model(args.model_type, **model_params)
        print(f"Created {args.model_type} model")
        
        # Train model
        print("Training model...")
        model.fit(X_train, y_train)
        
        # Evaluate model
        metrics, y_pred = evaluate_model(model, X_train, X_test, y_train, y_test)
        
        # Log parameters and metrics to MLflow
        mlflow.log_param("model_type", args.model_type)
        for param, value in model_params.items():
            mlflow.log_param(param, value)
        
        for metric, value in metrics.items():
            mlflow.log_metric(metric, value)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Create output directory
        os.makedirs(args.model_output, exist_ok=True)
        
        # Save model and scaler
        model_path = os.path.join(args.model_output, "model.pkl")
        scaler_path = os.path.join(args.model_output, "scaler.pkl")
        
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        
        # Save metrics
        metrics_path = os.path.join(args.model_output, "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save model metadata
        metadata = {
            'model_type': args.model_type,
            'training_date': datetime.now().isoformat(),
            'parameters': model_params,
            'metrics': metrics,
            'data_shape': {
                'train_samples': len(train_df),
                'test_samples': len(test_df),
                'features': X_train.shape[1]
            }
        }
        
        metadata_path = os.path.join(args.model_output, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nTraining completed successfully!")
        print(f"Model saved to: {model_path}")
        print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
        print(f"Test F1 Score: {metrics['test_f1']:.4f}")
        print(f"Test AUC: {metrics['test_auc']:.4f}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        raise
    
    finally:
        mlflow.end_run()


if __name__ == "__main__":
    main()