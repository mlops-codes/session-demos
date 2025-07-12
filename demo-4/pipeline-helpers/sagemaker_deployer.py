#!/usr/bin/env python3
"""
SageMaker Deployment Script
Handles model packaging and deployment to AWS SageMaker
"""
import boto3
import json
import os
import tarfile
import shutil
from datetime import datetime
import sagemaker
from sagemaker.sklearn.model import SKLearnModel
from sagemaker import get_execution_role


def create_sagemaker_inference_script():
    """Create the SageMaker inference script"""
    return '''
import joblib
import numpy as np
import pandas as pd
import os

def model_fn(model_dir):
    """Load model and scaler"""
    model = joblib.load(os.path.join(model_dir, "final_model.pkl"))
    scaler = joblib.load(os.path.join(model_dir, "final_scaler.pkl"))
    return {"model": model, "scaler": scaler}

def input_fn(request_body, request_content_type):
    """Parse input data"""
    if request_content_type == "application/json":
        import json
        data = json.loads(request_body)
        return np.array(data["instances"])
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model_dict):
    """Make predictions"""
    model = model_dict["model"]
    scaler = model_dict["scaler"]
    
    # Scale input data
    scaled_data = scaler.transform(input_data)
    
    # Make predictions
    predictions = model.predict(scaled_data)
    probabilities = model.predict_proba(scaled_data)
    
    return {
        "predictions": predictions.tolist(),
        "probabilities": probabilities.tolist()
    }

def output_fn(prediction, content_type):
    """Format output"""
    if content_type == "application/json":
        import json
        return json.dumps(prediction)
    else:
        raise ValueError(f"Unsupported content type: {content_type}")
'''


def create_model_package():
    """Create model.tar.gz for SageMaker"""
    print("📦 Creating SageMaker model package...")
    
    # Create model directory structure
    os.makedirs("model", exist_ok=True)
    
    # Copy model artifacts
    shutil.copy("deployment-package/pipeline_artifacts/deployment_package/final_model.pkl", "model/")
    shutil.copy("deployment-package/pipeline_artifacts/deployment_package/final_scaler.pkl", "model/")
    
    # Create inference script for SageMaker
    inference_code = create_sagemaker_inference_script()
    
    with open("model/inference.py", "w") as f:
        f.write(inference_code)
    
    # Create model.tar.gz
    with tarfile.open("model.tar.gz", "w:gz") as tar:
        tar.add("model", arcname=".")
    
    print("✅ Model package created: model.tar.gz")
    return "model.tar.gz"


def deploy_to_sagemaker():
    """Deploy model to SageMaker"""
    print("🚀 Deploying to SageMaker...")
    
    # Load deployment manifest
    with open("deployment-package/deployment_manifest.json", "r") as f:
        manifest = json.load(f)
    
    # Create model package
    model_package = create_model_package()
    
    # Initialize SageMaker session
    session = sagemaker.Session()
    role = os.getenv("SAGEMAKER_ROLE", get_execution_role())
    
    # Upload model to S3
    bucket = session.default_bucket()
    model_s3_key = f"loan-approval-model/{manifest['pipeline_id']}/model.tar.gz"
    model_s3_uri = session.upload_data(
        model_package, 
        bucket=bucket, 
        key_prefix=f"loan-approval-model/{manifest['pipeline_id']}"
    )
    
    print(f"📤 Model uploaded to: {model_s3_uri}")
    
    # Create SageMaker model
    model_name = f"loan-approval-{manifest['pipeline_id']}"
    endpoint_name = f"loan-approval-endpoint-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    sklearn_model = SKLearnModel(
        model_data=model_s3_uri,
        role=role,
        entry_point="inference.py",
        framework_version="1.0-1",
        py_version="py3",
        name=model_name
    )
    
    # Deploy to endpoint
    print(f"🚀 Creating SageMaker endpoint: {endpoint_name}")
    predictor = sklearn_model.deploy(
        initial_instance_count=1,
        instance_type="ml.t2.medium",
        endpoint_name=endpoint_name
    )
    
    print(f"✅ Model deployed successfully!")
    print(f"📍 Endpoint name: {endpoint_name}")
    print(f"🔗 Endpoint ARN: arn:aws:sagemaker:{session.boto_region_name}:{session.account_id()}:endpoint/{endpoint_name}")
    
    # Test the endpoint
    test_data = [[50000, 700, 120000, 5, 0.35, 8, 1, 1, 0]]  # Loan approval features
    result = predictor.predict({"instances": test_data})
    print(f"🧪 Test prediction: {result}")
    
    # Save deployment info
    deployment_info = {
        "endpoint_name": endpoint_name,
        "model_name": model_name,
        "model_s3_uri": model_s3_uri,
        "deployment_timestamp": datetime.now().isoformat(),
        "pipeline_id": manifest["pipeline_id"],
        "model_accuracy": manifest["model_accuracy"],
        "test_prediction": result
    }
    
    with open("sagemaker_deployment_info.json", "w") as f:
        json.dump(deployment_info, f, indent=2)
    
    return deployment_info


def main():
    """Main deployment function"""
    try:
        info = deploy_to_sagemaker()
        print(f"\n🎉 SageMaker deployment completed successfully!")
        return 0
    except Exception as e:
        print(f"❌ SageMaker deployment failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())