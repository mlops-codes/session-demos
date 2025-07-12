import mlflow
import mlflow.sklearn
import mlflow.projects
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import json
import joblib
from datetime import datetime
import yaml

class MLflowPipeline:
    def __init__(self, pipeline_name="end_to_end_pipeline"):
        self.pipeline_name = pipeline_name
        self.client = MlflowClient()
        self.pipeline_run_id = None
        self.artifacts_dir = "pipeline_artifacts"
        os.makedirs(self.artifacts_dir, exist_ok=True)

    mlflow.set_tracking_uri("https://dagshub.com/frontenddevtrainer/MLOps.mlflow")
        
    def stage_1_data_preparation(self):
        """Stage 1: Data Generation and Preparation"""
        print("📊 Stage 1: Data Preparation")
        
        with mlflow.start_run(run_name=f"{self.pipeline_name}_data_prep", nested=True):
            # Get environment parameters for data generation
            n_samples = int(os.getenv("N_SAMPLES", "2000"))
            
            # Generate synthetic loan approval dataset
            np.random.seed(42)  # For reproducibility
            
            # Generate features
            data = {
                'income': np.random.normal(50000, 20000, n_samples).clip(15000, 200000),
                'credit_score': np.random.normal(650, 100, n_samples).clip(300, 850),
                'loan_amount': np.random.normal(150000, 75000, n_samples).clip(10000, 500000),
                'employment_length': np.random.randint(0, 30, n_samples),
                'debt_to_income': np.random.uniform(0, 0.8, n_samples),
                'number_of_credit_lines': np.random.randint(1, 20, n_samples),
                'education_level': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples, p=[0.3, 0.4, 0.25, 0.05]),
                'property_type': np.random.choice(['Apartment', 'House', 'Condo'], n_samples, p=[0.4, 0.5, 0.1]),
                'loan_purpose': np.random.choice(['Home Purchase', 'Refinance', 'Home Improvement'], n_samples, p=[0.6, 0.3, 0.1])
            }
            
            df = pd.DataFrame(data)
            
            # Create target variable based on logical rules
            approval_prob = (
                0.3 * (df['credit_score'] - 300) / (850 - 300) +  # Credit score influence
                0.2 * (df['income'] - 15000) / (200000 - 15000) +  # Income influence
                0.2 * (1 - df['debt_to_income']) +  # Lower debt-to-income is better
                0.15 * (df['employment_length'] / 30) +  # Employment stability
                0.1 * (df['education_level'].map({'High School': 0.2, 'Bachelor': 0.5, 'Master': 0.8, 'PhD': 1.0})) +  # Education
                0.05 * (df['property_type'].map({'Apartment': 0.3, 'House': 0.8, 'Condo': 0.6}))  # Property type
            )
            
            # Add some noise and create binary target
            approval_prob += np.random.normal(0, 0.1, n_samples)
            df['loan_approved'] = (approval_prob > 0.6).astype(int)
            
            # Encode categorical variables
            le_education = LabelEncoder()
            le_property = LabelEncoder()
            le_purpose = LabelEncoder()
            
            df['education_level_encoded'] = le_education.fit_transform(df['education_level'])
            df['property_type_encoded'] = le_property.fit_transform(df['property_type'])
            df['loan_purpose_encoded'] = le_purpose.fit_transform(df['loan_purpose'])
            
            # Select features for modeling
            feature_names = ['income', 'credit_score', 'loan_amount', 'employment_length', 
                           'debt_to_income', 'number_of_credit_lines', 'education_level_encoded',
                           'property_type_encoded', 'loan_purpose_encoded']
            
            X = df[feature_names]
            y = df['loan_approved']
            
            # Log dataset parameters
            mlflow.log_param("n_samples", len(X))
            mlflow.log_param("n_features", X.shape[1])
            mlflow.log_param("n_classes", len(np.unique(y)))
            mlflow.log_param("data_generation_seed", 42)
            mlflow.log_param("dataset_type", "loan_approval")
            mlflow.log_param("approval_rate", float(y.mean()))
            
            # Data quality checks
            missing_values = X.isnull().sum().sum()
            duplicate_rows = X.duplicated().sum()
            
            mlflow.log_metric("missing_values", missing_values)
            mlflow.log_metric("duplicate_rows", duplicate_rows)
            
            # Class distribution (0=Denied, 1=Approved)
            class_dist = y.value_counts().to_dict()
            for class_id, count in class_dist.items():
                label = "approved" if class_id == 1 else "denied"
                mlflow.log_metric(f"loans_{label}_count", count)
            
            # Save raw data
            raw_data_path = f"{self.artifacts_dir}/raw_data.csv"
            combined_data = df  # Save full dataset with original categorical columns
            combined_data.to_csv(raw_data_path, index=False)
            mlflow.log_artifact(raw_data_path, "raw_data")
            
            # Data validation report
            validation_report = {
                "timestamp": datetime.now().isoformat(),
                "dataset_shape": X.shape,
                "missing_values": int(missing_values),
                "duplicate_rows": int(duplicate_rows),
                "class_distribution": class_dist,
                "feature_names": feature_names,
                "data_quality_score": 1.0 if missing_values == 0 and duplicate_rows == 0 else 0.8
            }
            
            with open(f"{self.artifacts_dir}/data_validation_report.json", 'w') as f:
                json.dump(validation_report, f, indent=2)
            mlflow.log_artifact(f"{self.artifacts_dir}/data_validation_report.json", "validation")
            
            mlflow.log_metric("data_quality_score", validation_report["data_quality_score"])
            
            print(f"   ✅ Generated loan approval dataset: {X.shape}")
            print(f"   ✅ Approval rate: {y.mean():.2%}")
            print(f"   ✅ Data quality score: {validation_report['data_quality_score']}")
            
            return X, y, validation_report
    
    def stage_2_preprocessing(self, X, y):
        """Stage 2: Data Preprocessing"""
        print("🔄 Stage 2: Data Preprocessing")
        
        with mlflow.start_run(run_name=f"{self.pipeline_name}_preprocessing", nested=True):
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Log split parameters
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("stratify", True)
            mlflow.log_param("random_state", 42)
            mlflow.log_metric("train_samples", len(X_train))
            mlflow.log_metric("test_samples", len(X_test))
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(
                scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
            
            # Log preprocessing parameters
            mlflow.log_param("scaling_method", "StandardScaler")
            mlflow.log_param("feature_scaling", True)
            
            # Save preprocessing artifacts
            scaler_path = f"{self.artifacts_dir}/scaler.pkl"
            joblib.dump(scaler, scaler_path)
            mlflow.log_artifact(scaler_path, "preprocessing")
            
            # Save processed datasets
            X_train_scaled.to_csv(f"{self.artifacts_dir}/X_train_processed.csv", index=False)
            X_test_scaled.to_csv(f"{self.artifacts_dir}/X_test_processed.csv", index=False)
            y_train.to_csv(f"{self.artifacts_dir}/y_train.csv", index=False)
            y_test.to_csv(f"{self.artifacts_dir}/y_test.csv", index=False)
            
            mlflow.log_artifact(f"{self.artifacts_dir}/X_train_processed.csv", "processed_data")
            mlflow.log_artifact(f"{self.artifacts_dir}/X_test_processed.csv", "processed_data")
            mlflow.log_artifact(f"{self.artifacts_dir}/y_train.csv", "processed_data")
            mlflow.log_artifact(f"{self.artifacts_dir}/y_test.csv", "processed_data")
            
            # Preprocessing statistics
            preprocessing_stats = {
                "original_features": X.shape[1],
                "processed_features": X_train_scaled.shape[1],
                "train_mean": X_train_scaled.mean().to_dict(),
                "train_std": X_train_scaled.std().to_dict(),
                "scaling_applied": True
            }
            
            with open(f"{self.artifacts_dir}/preprocessing_stats.json", 'w') as f:
                json.dump(preprocessing_stats, f, indent=2)
            mlflow.log_artifact(f"{self.artifacts_dir}/preprocessing_stats.json", "preprocessing")
            
            print(f"   ✅ Train set: {X_train_scaled.shape}")
            print(f"   ✅ Test set: {X_test_scaled.shape}")
            
            return X_train_scaled, X_test_scaled, y_train, y_test, scaler
    
    def stage_3_model_training(self, X_train, X_test, y_train, y_test):
        """Stage 3: Model Training and Selection"""
        print("🏗️  Stage 3: Model Training")
        
        models_performance = []
        
        # Train multiple models
        models_config = [
            {
                "name": "random_forest",
                "model": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
                "params": {"n_estimators": 100, "max_depth": 10}
            },
            {
                "name": "random_forest_tuned",
                "model": RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42),
                "params": {"n_estimators": 200, "max_depth": 15}
            },
            {
                "name": "logistic_regression",
                "model": LogisticRegression(C=1.0, random_state=42, max_iter=1000),
                "params": {"C": 1.0, "max_iter": 1000}
            }
        ]
        
        for model_config in models_config:
            with mlflow.start_run(run_name=f"{self.pipeline_name}_train_{model_config['name']}", nested=True):
                # Log model parameters
                mlflow.log_param("model_type", model_config["name"])
                for param_name, param_value in model_config["params"].items():
                    mlflow.log_param(param_name, param_value)
                
                # Train model
                model = model_config["model"]
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                
                # Calculate metrics
                train_accuracy = accuracy_score(y_train, y_pred_train)
                test_accuracy = accuracy_score(y_test, y_pred_test)
                test_precision = precision_score(y_test, y_pred_test, average='weighted')
                test_recall = recall_score(y_test, y_pred_test, average='weighted')
                test_f1 = f1_score(y_test, y_pred_test, average='weighted')
                
                # Log metrics
                mlflow.log_metric("train_accuracy", train_accuracy)
                mlflow.log_metric("test_accuracy", test_accuracy)
                mlflow.log_metric("test_precision", test_precision)
                mlflow.log_metric("test_recall", test_recall)
                mlflow.log_metric("test_f1", test_f1)
                mlflow.log_metric("overfitting_score", train_accuracy - test_accuracy)
                
                # Save model
                model_path = f"{self.artifacts_dir}/model_{model_config['name']}.pkl"
                joblib.dump(model, model_path)
                mlflow.log_artifact(model_path, "models")
                
                # Log model to registry
                mlflow.sklearn.log_model(
                    model, 
                    f"model_{model_config['name']}",
                    registered_model_name=f"loan_approval_pipeline_{model_config['name']}"
                )
                
                models_performance.append({
                    "name": model_config["name"],
                    "test_accuracy": test_accuracy,
                    "test_f1": test_f1,
                    "model": model,
                    "run_id": mlflow.active_run().info.run_id
                })
                
                print(f"   ✅ {model_config['name']}: Accuracy={test_accuracy:.4f}, F1={test_f1:.4f}")
        
        # Select best model
        best_model_info = max(models_performance, key=lambda x: x["test_accuracy"])
        
        with mlflow.start_run(run_name=f"{self.pipeline_name}_model_selection", nested=True):
            mlflow.log_param("selection_criteria", "test_accuracy")
            mlflow.log_param("best_model", best_model_info["name"])
            mlflow.log_metric("best_accuracy", best_model_info["test_accuracy"])
            
            # Save model selection report
            selection_report = {
                "timestamp": datetime.now().isoformat(),
                "selection_criteria": "test_accuracy",
                "models_evaluated": len(models_performance),
                "best_model": best_model_info["name"],
                "best_accuracy": best_model_info["test_accuracy"],
                "all_models_performance": [
                    {k: v for k, v in model.items() if k != "model"} 
                    for model in models_performance
                ]
            }
            
            with open(f"{self.artifacts_dir}/model_selection_report.json", 'w') as f:
                json.dump(selection_report, f, indent=2)
            mlflow.log_artifact(f"{self.artifacts_dir}/model_selection_report.json", "model_selection")
        
        print(f"   🏆 Best model: {best_model_info['name']} (Accuracy: {best_model_info['test_accuracy']:.4f})")
        
        return best_model_info, models_performance
    
    def stage_4_evaluation(self, best_model_info, X_test, y_test):
        """Stage 4: Model Evaluation and Validation"""
        print("📊 Stage 4: Model Evaluation")
        
        with mlflow.start_run(run_name=f"{self.pipeline_name}_evaluation", nested=True):
            model = best_model_info["model"]
            
            # Comprehensive evaluation
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            # Detailed metrics
            from sklearn.metrics import classification_report, confusion_matrix
            
            # Classification report
            class_report = classification_report(y_test, y_pred, output_dict=True)
            
            # Log per-class metrics (0=Denied, 1=Approved)
            for class_id in ['0', '1']:
                if class_id in class_report:
                    label = "approved" if class_id == '1' else "denied"
                    mlflow.log_metric(f"loans_{label}_precision", class_report[class_id]['precision'])
                    mlflow.log_metric(f"loans_{label}_recall", class_report[class_id]['recall'])
                    mlflow.log_metric(f"loans_{label}_f1", class_report[class_id]['f1-score'])
            
            # Overall metrics
            mlflow.log_metric("macro_avg_precision", class_report['macro avg']['precision'])
            mlflow.log_metric("macro_avg_recall", class_report['macro avg']['recall'])
            mlflow.log_metric("macro_avg_f1", class_report['macro avg']['f1-score'])
            mlflow.log_metric("weighted_avg_f1", class_report['weighted avg']['f1-score'])
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # Model confidence analysis
            confidence_scores = np.max(y_pred_proba, axis=1)
            mlflow.log_metric("avg_prediction_confidence", confidence_scores.mean())
            mlflow.log_metric("min_prediction_confidence", confidence_scores.min())
            mlflow.log_metric("low_confidence_predictions", (confidence_scores < 0.7).sum())
            
            # Save evaluation artifacts
            evaluation_report = {
                "timestamp": datetime.now().isoformat(),
                "model_name": best_model_info["name"],
                "test_samples": len(X_test),
                "classification_report": class_report,
                "confusion_matrix": cm.tolist(),
                "confidence_analysis": {
                    "avg_confidence": float(confidence_scores.mean()),
                    "min_confidence": float(confidence_scores.min()),
                    "low_confidence_count": int((confidence_scores < 0.7).sum())
                }
            }
            
            with open(f"{self.artifacts_dir}/evaluation_report.json", 'w') as f:
                json.dump(evaluation_report, f, indent=2)
            mlflow.log_artifact(f"{self.artifacts_dir}/evaluation_report.json", "evaluation")
            
            # Save predictions
            predictions_df = pd.DataFrame({
                'true_label': y_test,
                'predicted_label': y_pred,
                'confidence': confidence_scores,
                'correct': y_test == y_pred
            })
            predictions_df.to_csv(f"{self.artifacts_dir}/predictions.csv", index=False)
            mlflow.log_artifact(f"{self.artifacts_dir}/predictions.csv", "evaluation")
            
            mlflow.log_param("evaluation_passed", True)
            
            print(f"   ✅ Evaluation completed")
            print(f"   ✅ Average confidence: {confidence_scores.mean():.3f}")
            
            return evaluation_report
    
    def stage_5_deployment_prep(self, best_model_info, scaler):
        """Stage 5: Deployment Preparation"""
        print("🚀 Stage 5: Deployment Preparation")
        
        with mlflow.start_run(run_name=f"{self.pipeline_name}_deployment", nested=True):
            model_name = f"loan_approval_pipeline_best_model"
            
            # Create deployment package
            deployment_dir = f"{self.artifacts_dir}/deployment_package"
            os.makedirs(deployment_dir, exist_ok=True)
            
            # Save final model and preprocessor
            final_model_path = f"{deployment_dir}/final_model.pkl"
            final_scaler_path = f"{deployment_dir}/final_scaler.pkl"
            
            joblib.dump(best_model_info["model"], final_model_path)
            joblib.dump(scaler, final_scaler_path)
            
            # Create model metadata
            model_metadata = {
                "model_name": model_name,
                "model_type": best_model_info["name"],
                "pipeline_run_id": self.pipeline_run_id,
                "training_timestamp": datetime.now().isoformat(),
                "performance": {
                    "accuracy": best_model_info["test_accuracy"],
                    "f1_score": best_model_info["test_f1"]
                },
                "deployment_ready": True,
                "artifacts": {
                    "model": "final_model.pkl",
                    "scaler": "final_scaler.pkl"
                }
            }
            
            with open(f"{deployment_dir}/model_metadata.json", 'w') as f:
                json.dump(model_metadata, f, indent=2)
            
            # Create inference script template
            inference_script = '''
import joblib
import numpy as np
import pandas as pd

class ModelInference:
    def __init__(self, model_path, scaler_path):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
    
    def predict(self, features):
        """Make prediction on new data"""
        # Ensure features is a DataFrame
        if not isinstance(features, pd.DataFrame):
            features = pd.DataFrame(features)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Make prediction
        prediction = self.model.predict(features_scaled)
        prediction_proba = self.model.predict_proba(features_scaled)
        
        return {
            "prediction": prediction.tolist(),
            "probabilities": prediction_proba.tolist(),
            "confidence": np.max(prediction_proba, axis=1).tolist()
        }

# Example usage:
# inference = ModelInference("final_model.pkl", "final_scaler.pkl")
# result = inference.predict([[50000, 700, 120000, 5, 0.35, 8, 1, 1, 0]])  # Loan approval features
'''
            
            with open(f"{deployment_dir}/inference.py", 'w') as f:
                f.write(inference_script)
            
            # Log deployment artifacts
            mlflow.log_artifact(deployment_dir, "deployment_package")
            
            # Register final model
            mlflow.sklearn.log_model(
                best_model_info["model"],
                "final_model",
                registered_model_name=model_name
            )
            
            mlflow.log_param("deployment_ready", True)
            mlflow.log_param("deployment_timestamp", datetime.now().isoformat())
            
            print(f"   ✅ Deployment package created")
            print(f"   ✅ Model registered: {model_name}")
            
            return model_metadata
    
    def run_complete_pipeline(self):
        """Run the complete end-to-end pipeline"""
        print("🎯 Starting End-to-End MLflow Pipeline")
        print("=" * 60)
        
        mlflow.set_experiment(self.pipeline_name)
        
        with mlflow.start_run(run_name=f"{self.pipeline_name}_complete"):
            self.pipeline_run_id = mlflow.active_run().info.run_id
            
            # Log pipeline metadata
            mlflow.log_param("pipeline_name", self.pipeline_name)
            mlflow.log_param("pipeline_version", "1.0")
            mlflow.log_param("start_timestamp", datetime.now().isoformat())
            
            try:
                # Stage 1: Data Preparation
                X, y, validation_report = self.stage_1_data_preparation()
                
                # Stage 2: Preprocessing
                X_train, X_test, y_train, y_test, scaler = self.stage_2_preprocessing(X, y)
                
                # Stage 3: Model Training
                best_model_info, all_models = self.stage_3_model_training(X_train, X_test, y_train, y_test)
                
                # Stage 4: Evaluation
                evaluation_report = self.stage_4_evaluation(best_model_info, X_test, y_test)
                
                # Stage 5: Deployment Preparation
                deployment_metadata = self.stage_5_deployment_prep(best_model_info, scaler)
                
                # Log pipeline summary
                mlflow.log_param("pipeline_status", "completed")
                mlflow.log_param("end_timestamp", datetime.now().isoformat())
                mlflow.log_metric("final_model_accuracy", best_model_info["test_accuracy"])
                mlflow.log_metric("pipeline_stages_completed", 5)
                
                # Create pipeline summary
                pipeline_summary = {
                    "pipeline_name": self.pipeline_name,
                    "run_id": self.pipeline_run_id,
                    "status": "completed",
                    "stages": {
                        "data_preparation": "completed",
                        "preprocessing": "completed", 
                        "model_training": "completed",
                        "evaluation": "completed",
                        "deployment_prep": "completed"
                    },
                    "best_model": best_model_info["name"],
                    "final_accuracy": best_model_info["test_accuracy"],
                    "data_quality_score": validation_report["data_quality_score"],
                    "deployment_ready": True
                }
                
                with open(f"{self.artifacts_dir}/pipeline_summary.json", 'w') as f:
                    json.dump(pipeline_summary, f, indent=2)
                mlflow.log_artifact(f"{self.artifacts_dir}/pipeline_summary.json")
                
                print("\n" + "=" * 60)
                print("✅ Pipeline completed successfully!")
                print(f"🏆 Best model: {best_model_info['name']}")
                print(f"📊 Final accuracy: {best_model_info['test_accuracy']:.4f}")
                print(f"🚀 Deployment ready: {deployment_metadata['deployment_ready']}")
                print(f"📋 Pipeline run ID: {self.pipeline_run_id}")
                
                return pipeline_summary
                
            except Exception as e:
                mlflow.log_param("pipeline_status", "failed")
                mlflow.log_param("error_message", str(e))
                print(f"❌ Pipeline failed: {e}")
                raise

def main():
    """Main function to run the complete pipeline"""
    pipeline = MLflowPipeline("complete_ml_pipeline")
    
    try:
        summary = pipeline.run_complete_pipeline()
        
        print("\n💡 Pipeline Features Demonstrated:")
        print("   ✓ Multi-stage pipeline with nested runs")
        print("   ✓ Data quality validation and versioning")
        print("   ✓ Automated preprocessing and artifact logging")
        print("   ✓ Multi-model training and selection")
        print("   ✓ Comprehensive model evaluation")
        print("   ✓ Deployment package preparation")
        print("   ✓ Model registry integration")
        print("   ✓ Complete audit trail")
        
        print("\n🔍 Check MLflow UI for:")
        print("   - Experiment: 'complete_ml_pipeline'")
        print("   - Nested runs for each pipeline stage")
        print("   - Registered models in Models tab")
        print("   - Deployment artifacts")
        
    except Exception as e:
        print(f"❌ Pipeline execution failed: {e}")

if __name__ == "__main__":
    main()