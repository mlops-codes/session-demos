import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def train_and_log_model(model_type="random_forest", **model_params):
    """Train a model and log everything to MLflow"""
    
    with mlflow.start_run(run_name=f"{model_type}_experiment"):
        # Load and prepare data
        iris = load_iris()
        X = pd.DataFrame(iris.data, columns=iris.feature_names)
        y = iris.target
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Log dataset info
        mlflow.log_param("dataset", "iris")
        mlflow.log_param("n_samples", len(X))
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)
        
        # Create model based on type
        if model_type == "random_forest":
            model = RandomForestClassifier(**model_params, random_state=42)
        elif model_type == "logistic_regression":
            model = LogisticRegression(**model_params, random_state=42, max_iter=1000)
        elif model_type == "svm":
            model = SVC(**model_params, random_state=42, probability=True)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Log model parameters
        mlflow.log_param("model_type", model_type)
        for param, value in model_params.items():
            mlflow.log_param(param, value)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # Create and log confusion matrix plot
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=iris.target_names, 
                   yticklabels=iris.target_names)
        plt.title(f'Confusion Matrix - {model_type}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        plt.close()
        
        # Create feature importance plot (for tree-based models)
        if hasattr(model, 'feature_importances_'):
            plt.figure(figsize=(10, 6))
            feature_importance = pd.DataFrame({
                'feature': iris.feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            sns.barplot(data=feature_importance, x='importance', y='feature')
            plt.title(f'Feature Importance - {model_type}')
            plt.tight_layout()
            plt.savefig("feature_importance.png")
            mlflow.log_artifact("feature_importance.png")
            plt.close()
        
        # Log the model
        mlflow.sklearn.log_model(
            model, 
            "model",
            registered_model_name=f"iris_{model_type}"
        )
        
        # Log predictions as artifact
        predictions_df = pd.DataFrame({
            'true_label': y_test,
            'predicted_label': y_pred,
            'prediction_probability': np.max(y_pred_proba, axis=1)
        })
        predictions_df.to_csv("predictions.csv", index=False)
        mlflow.log_artifact("predictions.csv")
        
        print(f"✅ {model_type} experiment logged successfully!")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Run ID: {mlflow.active_run().info.run_id}")
        
        return model, accuracy

def run_comparison_experiments():
    """Run multiple experiments with different models for comparison"""
    
    # Set experiment
    mlflow.set_experiment("iris_model_comparison")
    
    print("🚀 Starting MLflow experiment comparison...")
    
    experiments = [
        ("random_forest", {"n_estimators": 100, "max_depth": 5}),
        ("random_forest", {"n_estimators": 200, "max_depth": 10}),
        ("logistic_regression", {"C": 1.0, "solver": "lbfgs"}),
        ("logistic_regression", {"C": 0.1, "solver": "lbfgs"}),
        ("svm", {"C": 1.0, "kernel": "rbf"}),
        ("svm", {"C": 1.0, "kernel": "linear"}),
    ]
    
    results = []
    for model_type, params in experiments:
        print(f"\n📊 Running {model_type} with params: {params}")
        model, accuracy = train_and_log_model(model_type, **params)
        results.append((model_type, params, accuracy))
    
    print("\n📈 Experiment Results Summary:")
    print("-" * 60)
    for model_type, params, accuracy in results:
        print(f"{model_type:20} | {str(params):25} | {accuracy:.4f}")
    
    print(f"\n🎯 Best model: {max(results, key=lambda x: x[2])}")
    
    return results

if __name__ == "__main__":
    # Clean up any existing artifacts
    import os
    for file in ["confusion_matrix.png", "feature_importance.png", "predictions.csv"]:
        if os.path.exists(file):
            os.remove(file)
    
    # Run experiments
    results = run_comparison_experiments()
    
    print("\n" + "="*60)
    print("🎉 All experiments completed!")
    print("💡 View results in MLflow UI:")
    print("   mlflow ui --backend-store-uri ./mlruns")
    print("   Then open: http://localhost:5000")
    print("="*60)