import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pandas as pd
import numpy as np
import time
from datetime import datetime
import requests
import json

class ModelRegistryDemo:
    def __init__(self):
        self.client = MlflowClient()
        self.model_name = "iris_classifier"
        
    def train_model_version(self, model_type="random_forest", version_suffix="v1"):
        """Train a model and return run info"""
        print(f"🏗️  Training {model_type} model ({version_suffix})...")
        
        # Load data
        iris = load_iris()
        X = pd.DataFrame(iris.data, columns=iris.feature_names)
        y = iris.target
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        with mlflow.start_run(run_name=f"{model_type}_{version_suffix}") as run:
            # Log parameters
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("version", version_suffix)
            mlflow.log_param("dataset", "iris")
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))
            
            # Train model
            if model_type == "random_forest":
                model = RandomForestClassifier(
                    n_estimators=100 if version_suffix == "v1" else 200,
                    max_depth=5 if version_suffix == "v1" else 10,
                    random_state=42
                )
                mlflow.log_param("n_estimators", model.n_estimators)
                mlflow.log_param("max_depth", model.max_depth)
            else:
                model = LogisticRegression(
                    C=1.0 if version_suffix == "v1" else 0.1,
                    random_state=42,
                    max_iter=1000
                )
                mlflow.log_param("C", model.C)
            
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            
            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            
            # Log model with registry
            mlflow.sklearn.log_model(
                model,
                "model",
                registered_model_name=self.model_name
            )
            
            print(f"   ✅ {model_type} {version_suffix} trained - Accuracy: {accuracy:.4f}")
            
            return run.info.run_id, accuracy, model
    
    def register_models(self):
        """Train and register multiple model versions"""
        mlflow.set_experiment("model_registry_demo")
        
        print("🚀 Training and registering multiple model versions...")
        
        # Train different model versions
        models_info = []
        
        # Version 1: Basic Random Forest
        run_id_1, acc_1, _ = self.train_model_version("random_forest", "v1")
        models_info.append(("v1", "random_forest", run_id_1, acc_1))
        
        # Version 2: Improved Random Forest
        run_id_2, acc_2, _ = self.train_model_version("random_forest", "v2")
        models_info.append(("v2", "random_forest", run_id_2, acc_2))
        
        # Version 3: Logistic Regression
        run_id_3, acc_3, _ = self.train_model_version("logistic_regression", "v3")
        models_info.append(("v3", "logistic_regression", run_id_3, acc_3))
        
        return models_info
    
    def manage_model_stages(self, models_info):
        """Demonstrate model stage transitions"""
        print("\n🔄 Managing model lifecycle stages...")
        
        # Get registered model
        try:
            registered_model = self.client.get_registered_model(self.model_name)
            print(f"📋 Model '{self.model_name}' found with {len(registered_model.latest_versions)} versions")
        except Exception as e:
            print(f"❌ Error getting registered model: {e}")
            return
        
        # Get all model versions
        model_versions = self.client.search_model_versions(f"name='{self.model_name}'")
        
        if len(model_versions) < 3:
            print("⚠️  Not enough model versions found. Please run register_models() first.")
            return
        
        # Sort by version number
        model_versions = sorted(model_versions, key=lambda v: int(v.version))
        
        print(f"\n📊 Found {len(model_versions)} model versions:")
        for mv in model_versions:
            print(f"   Version {mv.version}: {mv.current_stage}")
        
        # Promote best performing model to Staging
        best_version = max(models_info, key=lambda x: x[3])  # Get highest accuracy
        best_run_id = best_version[2]
        
        # Find the version corresponding to the best run
        best_model_version = None
        for mv in model_versions:
            if mv.run_id == best_run_id:
                best_model_version = mv
                break
        
        if best_model_version:
            print(f"\n⬆️  Promoting version {best_model_version.version} to Staging (accuracy: {best_version[3]:.4f})")
            self.client.transition_model_version_stage(
                name=self.model_name,
                version=best_model_version.version,
                stage="Staging",
                archive_existing_versions=False
            )
            
            # Add description
            self.client.update_model_version(
                name=self.model_name,
                version=best_model_version.version,
                description=f"Best performing model with {best_version[3]:.4f} accuracy. "
                           f"Model type: {best_version[1]}. Promoted on {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            )
            
            # Wait a moment for transition
            time.sleep(2)
            
            # Simulate testing in staging
            print("🧪 Simulating staging tests...")
            time.sleep(1)
            
            # If tests pass, promote to Production
            if best_version[3] > 0.9:  # Good accuracy threshold
                print("✅ Staging tests passed! Promoting to Production...")
                self.client.transition_model_version_stage(
                    name=self.model_name,
                    version=best_model_version.version,
                    stage="Production",
                    archive_existing_versions=True
                )
                
                # Add production annotation
                self.client.set_model_version_tag(
                    name=self.model_name,
                    version=best_model_version.version,
                    key="validation_status",
                    value="passed"
                )
                self.client.set_model_version_tag(
                    name=self.model_name,
                    version=best_model_version.version,
                    key="deployment_date",
                    value=datetime.now().isoformat()
                )
            else:
                print("❌ Staging tests failed. Model needs improvement.")
        
        # Archive older versions
        for mv in model_versions:
            if mv.version != best_model_version.version and mv.current_stage == "None":
                print(f"📦 Archiving version {mv.version}")
                self.client.transition_model_version_stage(
                    name=self.model_name,
                    version=mv.version,
                    stage="Archived"
                )
    
    def demonstrate_model_serving(self):
        """Demonstrate model serving capabilities"""
        print("\n🌐 Demonstrating model serving...")
        
        # Get production model
        try:
            production_versions = self.client.get_latest_versions(
                self.model_name, 
                stages=["Production"]
            )
            
            if not production_versions:
                print("⚠️  No production model found. Skipping serving demo.")
                return
            
            production_version = production_versions[0]
            print(f"🎯 Using production model version {production_version.version}")
            
            # Load the model
            model_uri = f"models:/{self.model_name}/Production"
            model = mlflow.sklearn.load_model(model_uri)
            
            # Create sample prediction
            sample_data = np.array([[5.1, 3.5, 1.4, 0.2]])  # Setosa sample
            prediction = model.predict(sample_data)
            prediction_proba = model.predict_proba(sample_data)
            
            print(f"📊 Sample prediction:")
            print(f"   Input: {sample_data[0]}")
            print(f"   Prediction: {prediction[0]}")
            print(f"   Probabilities: {prediction_proba[0]}")
            
            # Create serving instructions
            serving_info = {
                "model_name": self.model_name,
                "model_version": production_version.version,
                "model_uri": model_uri,
                "sample_input": sample_data.tolist(),
                "sample_output": {
                    "prediction": int(prediction[0]),
                    "probabilities": prediction_proba[0].tolist()
                },
                "serving_instructions": {
                    "local_serving": f"mlflow models serve -m {model_uri} -p 1234",
                    "docker_serving": f"mlflow models build-docker -m {model_uri} -n iris-model",
                    "cloud_serving": "Deploy using MLflow cloud integrations"
                }
            }
            
            # Save serving info
            with open("model_serving_info.json", "w") as f:
                json.dump(serving_info, f, indent=2)
            
            print(f"\n🔧 Serving instructions saved to model_serving_info.json")
            print(f"💡 To serve locally: mlflow models serve -m {model_uri} -p 1234")
            
        except Exception as e:
            print(f"❌ Error in serving demo: {e}")
    
    def show_audit_trail(self):
        """Display model audit trail"""
        print("\n📋 Model Registry Audit Trail:")
        print("=" * 50)
        
        try:
            # Get model versions with their history
            model_versions = self.client.search_model_versions(f"name='{self.model_name}'")
            
            for mv in sorted(model_versions, key=lambda v: int(v.version)):
                print(f"\n🏷️  Version {mv.version}:")
                print(f"   Stage: {mv.current_stage}")
                print(f"   Created: {datetime.fromtimestamp(mv.creation_timestamp/1000)}")
                print(f"   Run ID: {mv.run_id}")
                print(f"   Source: {mv.source}")
                
                if mv.description:
                    print(f"   Description: {mv.description}")
                
                # Get tags
                if hasattr(mv, 'tags') and mv.tags:
                    print(f"   Tags: {mv.tags}")
            
            print("\n📈 Model Performance Summary:")
            # Get run details for each version
            for mv in model_versions:
                try:
                    run = self.client.get_run(mv.run_id)
                    accuracy = run.data.metrics.get('accuracy', 'N/A')
                    model_type = run.data.params.get('model_type', 'Unknown')
                    print(f"   Version {mv.version} ({model_type}): {accuracy}")
                except:
                    continue
                    
        except Exception as e:
            print(f"❌ Error retrieving audit trail: {e}")

def main():
    """Main function to run model registry demo"""
    print("🎯 MLflow Demo 4: Model Registry & Lifecycle Management")
    print("=" * 60)
    
    demo = ModelRegistryDemo()
    
    # Step 1: Train and register models
    models_info = demo.register_models()
    
    # Step 2: Manage model stages
    demo.manage_model_stages(models_info)
    
    # Step 3: Demonstrate serving
    demo.demonstrate_model_serving()
    
    # Step 4: Show audit trail
    demo.show_audit_trail()
    
    print("\n" + "=" * 60)
    print("✅ Model Registry demo completed!")
    print("\n💡 Key features demonstrated:")
    print("   ✓ Model registration and versioning")
    print("   ✓ Stage transitions (None → Staging → Production → Archived)")
    print("   ✓ Model metadata and descriptions")
    print("   ✓ Model serving preparation")
    print("   ✓ Audit trail and lineage tracking")
    
    print("\n🔍 In MLflow UI (Models tab) you can:")
    print("   - View all registered models")
    print("   - Compare model versions")
    print("   - Manage stage transitions")
    print("   - View model lineage")
    print("   - Download model artifacts")
    
    print("\n🚀 Next steps:")
    print("   1. Check the Models tab in MLflow UI")
    print("   2. Try serving the production model")
    print("   3. Experiment with stage transitions")
    print("=" * 60)

if __name__ == "__main__":
    main()