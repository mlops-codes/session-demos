import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import itertools
from datetime import datetime

class HyperparameterSweep:
    def __init__(self):
        self.datasets = {
            'iris': load_iris()
        }
    
    def prepare_data(self, dataset_name):
        """Prepare dataset for training"""
        dataset = self.datasets[dataset_name]
        X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
        y = dataset.target
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        return X_train, X_test, y_train, y_test, dataset
    
    def random_forest_sweep(self):
        """Comprehensive Random Forest hyperparameter sweep"""
        mlflow.set_experiment("random_forest_param_sweep")
        
        # Parameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False]
        }
        
        datasets_to_test = ['iris']
        
        print("🔍 Starting Random Forest hyperparameter sweep...")
        
        # Generate all combinations (limit to reasonable number)
        param_combinations = list(itertools.product(*param_grid.values()))
        param_names = list(param_grid.keys())
        
        # Limit combinations for demo (take first 20 for iris-only)
        param_combinations = param_combinations[:20]
        
        best_results = {}
        
        dataset_name = 'iris'
        print(f"\n📊 Testing on {dataset_name} dataset...")
        X_train, X_test, y_train, y_test, dataset = self.prepare_data(dataset_name)
        
        best_accuracy = 0
        best_params = None
        
        for i, param_values in enumerate(param_combinations):
                params = dict(zip(param_names, param_values))
                
                with mlflow.start_run(run_name=f"rf_{dataset_name}_{i}"):
                    # Log dataset info
                    mlflow.log_param("dataset", dataset_name)
                    mlflow.log_param("n_samples", len(X_train) + len(X_test))
                    mlflow.log_param("n_features", X_train.shape[1])
                    mlflow.log_param("n_classes", len(np.unique(y_train)))
                    
                    # Log model parameters
                    mlflow.log_param("model_type", "random_forest")
                    for param, value in params.items():
                        mlflow.log_param(param, value)
                    
                    try:
                        # Train model
                        model = RandomForestClassifier(**params, random_state=42)
                        model.fit(X_train, y_train)
                        
                        # Evaluate
                        y_pred = model.predict(X_test)
                        accuracy = accuracy_score(y_test, y_pred)
                        
                        # Cross-validation score
                        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
                        cv_mean = cv_scores.mean()
                        cv_std = cv_scores.std()
                        
                        # Log metrics
                        mlflow.log_metric("test_accuracy", accuracy)
                        mlflow.log_metric("cv_accuracy_mean", cv_mean)
                        mlflow.log_metric("cv_accuracy_std", cv_std)
                        mlflow.log_metric("train_accuracy", model.score(X_train, y_train))
                        
                        # Log model complexity metrics
                        mlflow.log_metric("n_estimators_used", params['n_estimators'])
                        if params['max_depth'] is not None:
                            mlflow.log_metric("max_depth_used", params['max_depth'])
                        
                        # Conditional logging based on performance
                        if accuracy > 0.9:
                            mlflow.log_param("high_performance", True)
                            mlflow.log_metric("performance_category", 3)  # High
                        elif accuracy > 0.8:
                            mlflow.log_param("high_performance", False)
                            mlflow.log_metric("performance_category", 2)  # Medium
                        else:
                            mlflow.log_param("high_performance", False)
                            mlflow.log_metric("performance_category", 1)  # Low
                        
                        # Log feature importance if model performs well
                        if accuracy > 0.85:
                            feature_importance = pd.DataFrame({
                                'feature': dataset.feature_names,
                                'importance': model.feature_importances_
                            }).sort_values('importance', ascending=False)
                            
                            feature_importance.to_csv("feature_importance.csv", index=False)
                            mlflow.log_artifact("feature_importance.csv")
                        
                        # Track best model for this dataset
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_params = params.copy()
                        
                        if i % 5 == 0:
                            print(f"   Completed {i+1}/{len(param_combinations)} combinations")
                    
                    except Exception as e:
                        mlflow.log_param("error", str(e))
                        mlflow.log_metric("test_accuracy", 0.0)
                        print(f"   Error with params {params}: {e}")
            
        best_results[dataset_name] = (best_accuracy, best_params)
        print(f"   Best {dataset_name} accuracy: {best_accuracy:.4f}")
        
        return best_results
    
    def conditional_parameter_logging(self):
        """Demo conditional parameter logging based on runtime decisions"""
        mlflow.set_experiment("conditional_param_logging")
        
        print("\n🎛️  Demonstrating conditional parameter logging...")
        
        techniques = [
            {"name": "basic", "use_regularization": False, "use_feature_selection": False},
            {"name": "regularized", "use_regularization": True, "use_feature_selection": False},
            {"name": "feature_selected", "use_regularization": False, "use_feature_selection": True},
            {"name": "full_pipeline", "use_regularization": True, "use_feature_selection": True},
        ]
        
        X_train, X_test, y_train, y_test, dataset = self.prepare_data('iris')
        
        for technique in techniques:
            with mlflow.start_run(run_name=f"conditional_{technique['name']}"):
                # Always log basic info
                mlflow.log_param("dataset", "iris")
                mlflow.log_param("technique", technique['name'])
                mlflow.log_param("timestamp", datetime.now().isoformat())
                
                # Conditional logging: regularization parameters
                if technique['use_regularization']:
                    C_value = np.random.choice([0.1, 1.0, 10.0])
                    solver = np.random.choice(['lbfgs', 'liblinear'])
                    
                    mlflow.log_param("regularization_enabled", True)
                    mlflow.log_param("C", C_value)
                    mlflow.log_param("solver", solver)
                    
                    model = LogisticRegression(C=C_value, solver=solver, random_state=42, max_iter=1000)
                else:
                    mlflow.log_param("regularization_enabled", False)
                    model = LogisticRegression(random_state=42, max_iter=1000)
                
                # Conditional logging: feature selection
                if technique['use_feature_selection']:
                    from sklearn.feature_selection import SelectKBest, f_classif
                    
                    k_features = np.random.choice([2, 3, 4])
                    selector = SelectKBest(f_classif, k=k_features)
                    
                    mlflow.log_param("feature_selection_enabled", True)
                    mlflow.log_param("n_features_selected", k_features)
                    mlflow.log_param("selection_method", "f_classif")
                    
                    X_train_selected = selector.fit_transform(X_train, y_train)
                    X_test_selected = selector.transform(X_test)
                    
                    # Log selected features
                    selected_features = [dataset.feature_names[i] for i in selector.get_support(indices=True)]
                    mlflow.log_param("selected_features", str(selected_features))
                    
                    train_data, test_data = X_train_selected, X_test_selected
                else:
                    mlflow.log_param("feature_selection_enabled", False)
                    train_data, test_data = X_train, X_test
                
                # Train and evaluate
                model.fit(train_data, y_train)
                y_pred = model.predict(test_data)
                accuracy = accuracy_score(y_test, y_pred)
                
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("n_features_used", train_data.shape[1])
                
                # Conditional logging: model artifacts (only for good models)
                if accuracy > 0.9:
                    mlflow.sklearn.log_model(model, "high_performance_model")
                    mlflow.log_param("model_saved", True)
                else:
                    mlflow.log_param("model_saved", False)
                
                print(f"   {technique['name']:15} | Accuracy: {accuracy:.4f} | Features: {train_data.shape[1]}")

def main():
    """Main function to run parameter management demos"""
    print("🎯 MLflow Demo 2: Parameter Management & Hyperparameter Sweeps")
    print("=" * 70)
    
    sweep = HyperparameterSweep()
    
    # Run hyperparameter sweep
    print("\n📈 Part 1: Hyperparameter Grid Search")
    best_results = sweep.random_forest_sweep()
    
    print("\n🏆 Best results summary:")
    for dataset, (accuracy, params) in best_results.items():
        print(f"  {dataset:15} | Accuracy: {accuracy:.4f} | Best params: {params}")
    
    # Run conditional logging demo
    print("\n🎛️  Part 2: Conditional Parameter Logging")
    sweep.conditional_parameter_logging()
    
    print("\n" + "=" * 70)
    print("✅ Demo completed! Check MLflow UI for detailed results:")
    print("   Experiments created:")
    print("   - random_forest_param_sweep")
    print("   - conditional_param_logging")
    print("\n💡 Run: mlflow ui --backend-store-uri ./mlruns")
    print("=" * 70)

if __name__ == "__main__":
    main()